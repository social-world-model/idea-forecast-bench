from __future__ import annotations

import dataclasses

import pytest

import live_idea_bench.similarity as similarity_module
from live_idea_bench.config import SimilarityConfig, load_similarity_config
from live_idea_bench.models import IdeaPrediction, MatchResult, PaperRecord
from live_idea_bench.similarity import compute_similarity, evaluate_predictions, is_match, score_prediction_list


@pytest.fixture(autouse=True)
def _force_hybrid_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests exercise the lexical matcher and metric math, not the
    embedding engine. The shipped default is ``engine: embedding`` (Voyage,
    needs a key), so pin the engine to ``hybrid`` to keep them hermetic and
    key-free regardless of the default in similarity.yaml."""

    def _hybrid_config(*args, **kwargs):
        cfg = load_similarity_config(*args, **kwargs)
        return dataclasses.replace(cfg, engine="hybrid")

    monkeypatch.setattr(similarity_module, "load_similarity_config", _hybrid_config)


def _paper(
    paper_id: str,
    month: str,
    title: str,
    summary: str,
    *,
    published_date: str,
) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        month=month,
        summary=summary,
        keywords=title.lower().split(),
        source_path=f"/fake/{paper_id}.md",
        published_date=published_date,
    )


def _prediction(rank: int, title: str, rationale: str) -> IdeaPrediction:
    return IdeaPrediction(rank=rank, title=title, rationale=rationale, approach=rationale)


def test_evaluate_predictions_uses_one_to_one_matching_for_duplicate_future_hits() -> None:
    train = [_paper("train-1", "2024-01", "Old baseline", "old baseline methods", published_date="2024-01-01")]
    future = [
        _paper(
            "future-1",
            "2024-02",
            "Graph agents for retrieval",
            "graph agents for retrieval tasks",
            published_date="2024-02-15",
        ),
        _paper(
            "future-2",
            "2024-03",
            "Diffusion video benchmark",
            "video diffusion benchmark release",
            published_date="2024-03-01",
        ),
    ]
    predictions = [
        _prediction(1, "Graph agents for retrieval", "graph agents retrieval tasks"),
        _prediction(2, "Graph agents for retrieval v2", "graph agents retrieval tasks"),
    ]

    result = evaluate_predictions(
        predictions=predictions,
        train_papers=train,
        future_papers=future,
        k=2,
        cutoff_date="2024-02-01",
        future_end_date="2024-03-31",
    )

    assert result.hit_at_k == 1.0
    assert result.matched_paper_ids == ["future-1"]
    assert result.precision_at_k == 0.5
    assert result.mrr == 1.0
    assert result.duplicate_rate == 0.5
    assert 0.0 < result.lead_time <= 1.0


def test_llm_judge_raises_on_unparseable_score(monkeypatch) -> None:
    """The LLM-judge eval path must fail loud on a score it cannot parse rather
    than silently scoring 0.0 (which marks a real match as a miss). It must also
    parse bold/lowercase/leading-dot score formats."""
    from live_idea_bench.config import Config

    monkeypatch.setattr(similarity_module, "create_client", lambda m: (object(), m))
    cfg = SimilarityConfig(engine="llm", llm_match_threshold=0.7,
                           system_prompt="s", user_prompt_template="{idea}|{context}")
    rt = Config()

    def _reply(text):
        monkeypatch.setattr(similarity_module, "get_response_from_llm", lambda **_k: (text, []))
        return similarity_module._llm_similarity("idea", "ctx", cfg, rt)

    # Broadened parsing: bold / lowercase / leading dot all succeed.
    assert _reply("**Score:** 0.9\nReasoning: x").score == pytest.approx(0.9)
    assert _reply("score: 0.42").score == pytest.approx(0.42)
    assert _reply("Score: .8").score == pytest.approx(0.8)
    # No parseable score -> raise, not silent 0.0.
    with pytest.raises(ValueError, match="no parseable"):
        _reply("I think these are quite related but won't give a number.")


def test_hybrid_is_match_reuses_match_result_components() -> None:
    """is_match (hybrid) must read MatchResult.semantic/keyword rather than
    recompute, so the match decision and the sort score use the same numbers.
    A result whose keyword>=threshold but semantic<threshold must still match,
    and a result missing the components must fall back to recompute."""
    cfg = SimilarityConfig(engine="hybrid", semantic_threshold=0.5, keyword_threshold=0.3)

    # compute_similarity populates semantic+keyword on the result.
    res = compute_similarity("graph retrieval agents", "graph retrieval agents for planning", cfg)
    assert res.semantic is not None and res.keyword is not None

    # Stored-component path: keyword above threshold, semantic below -> match.
    forced = MatchResult(score=0.9, engine_name="hybrid", semantic=0.1, keyword=0.4)
    assert is_match(forced, "x", "y", cfg) is True
    # Both below threshold -> no match, even with a high (irrelevant) score.
    forced_low = MatchResult(score=0.9, engine_name="hybrid", semantic=0.1, keyword=0.1)
    assert is_match(forced_low, "x", "y", cfg) is False
    # Missing components -> recompute fallback still works (identical strings match).
    legacy = MatchResult(score=0.0, engine_name="hybrid")
    assert is_match(legacy, "same text here", "same text here", cfg) is True


def test_coverage_and_recall_diverge_when_future_exceeds_k() -> None:
    """coverage_at_k uses |future| as denominator; recall_at_k uses min(k,|future|).
    With |future| > k they must diverge: coverage is depressed by the large pool,
    recall is a true [0,1] hit-rate over what the top-k could reach."""
    train = [_paper("train-1", "2024-01", "Old", "old text", published_date="2024-01-01")]
    # 4 future papers, only 1 lexically matchable; k=1.
    future = [
        _paper("f-1", "2024-02", "Neural retrieval", "neural retrieval methods", published_date="2024-02-15"),
        _paper("f-2", "2024-02", "Protein folding", "protein folding simulation", published_date="2024-02-16"),
        _paper("f-3", "2024-02", "Climate model", "climate ocean modeling", published_date="2024-02-17"),
        _paper("f-4", "2024-02", "Robotics grasp", "robot grasp planning", published_date="2024-02-18"),
    ]
    predictions = [_prediction(1, "Neural retrieval", "neural retrieval methods")]

    scored = score_prediction_list(
        predictions=predictions, train_papers=train, future_papers=future, k=1,
        cutoff_date="2024-02-01", future_end_date="2024-03-31",
    )
    ev = scored.evaluation
    assert ev.matched_paper_ids == ["f-1"]
    assert ev.coverage_at_k == pytest.approx(1 / 4)        # matched / |future|
    assert ev.recall_at_k == pytest.approx(1 / 1)          # matched / min(k, |future|)
    assert ev.coverage_at_k < ev.recall_at_k
    assert 0.0 <= ev.coverage_at_k <= 1.0
    assert 0.0 <= ev.recall_at_k <= 1.0


def test_weighted_metrics_without_popularity_weights_default_to_zero() -> None:
    """When no popularity_weights passed, weighted metrics default to 0.0 (opt-in)."""
    train = [_paper("train-1", "2024-01", "Old paper", "old paper text", published_date="2024-01-01")]
    future = [
        _paper("future-1", "2024-02", "Neural retrieval", "neural retrieval methods", published_date="2024-02-15")
    ]
    predictions = [_prediction(1, "Neural retrieval", "neural retrieval methods")]

    scored = score_prediction_list(
        predictions=predictions,
        train_papers=train,
        future_papers=future,
        k=1,
        cutoff_date="2024-02-01",
        future_end_date="2024-03-31",
    )

    assert scored.evaluation.weighted_hit_at_k == 0.0
    assert scored.evaluation.weighted_precision_at_k == 0.0
    assert scored.evaluation.weighted_mrr == 0.0
    assert scored.evaluation.popularity_recall_at_k == 0.0


def test_weighted_metrics_with_popularity_weights_computes_correctly() -> None:
    """When popularity_weights are provided, all 4 weighted metrics are computed."""
    train = [_paper("train-1", "2024-01", "Old paper", "old paper text", published_date="2024-01-01")]
    future = [
        _paper("future-1", "2024-02", "Neural retrieval", "neural retrieval methods", published_date="2024-02-15"),
        _paper("future-2", "2024-03", "Graph attention", "graph attention networks", published_date="2024-03-01"),
    ]
    predictions = [_prediction(1, "Neural retrieval", "neural retrieval methods")]
    popularity_weights = {"future-1": 0.8, "future-2": 0.3}

    scored = score_prediction_list(
        predictions=predictions,
        train_papers=train,
        future_papers=future,
        k=1,
        cutoff_date="2024-02-01",
        future_end_date="2024-03-31",
        popularity_weights=popularity_weights,
    )

    # future-1 matches with weight 0.8
    assert scored.evaluation.weighted_hit_at_k == pytest.approx(0.8)
    assert scored.evaluation.weighted_precision_at_k == pytest.approx(0.8)  # 0.8/1
    assert scored.evaluation.weighted_mrr == pytest.approx(0.8)  # 1/1 * 0.8
    # popularity_recall = 0.8 / (0.8 + 0.3) ≈ 0.727
    assert scored.evaluation.popularity_recall_at_k == pytest.approx(0.8 / 1.1, rel=1e-3)


def test_popular_match_scores_higher_weighted_mrr_than_obscure_match() -> None:
    """Matching a popular paper yields higher weighted_mrr than matching an obscure one."""
    train = [_paper("train-1", "2024-01", "Baseline", "baseline text", published_date="2024-01-01")]
    future_popular = [
        _paper("pop-1", "2024-02", "Transformer attention", "transformer attention mechanisms", published_date="2024-02-15")
    ]
    future_obscure = [
        _paper("obs-1", "2024-02", "Transformer attention", "transformer attention mechanisms", published_date="2024-02-15")
    ]
    predictions = [_prediction(1, "Transformer attention", "transformer attention mechanisms")]

    scored_popular = score_prediction_list(
        predictions=predictions,
        train_papers=train,
        future_papers=future_popular,
        k=1,
        popularity_weights={"pop-1": 1.0},
    )
    scored_obscure = score_prediction_list(
        predictions=predictions,
        train_papers=train,
        future_papers=future_obscure,
        k=1,
        popularity_weights={"obs-1": 0.1},
    )

    assert scored_popular.evaluation.weighted_mrr > scored_obscure.evaluation.weighted_mrr


def test_popularity_recall_at_k_accounts_for_total_popularity_mass() -> None:
    """popularity_recall = matched weight sum / total weight sum across all future papers."""
    train = [_paper("train-1", "2024-01", "Baseline", "baseline text", published_date="2024-01-01")]
    future = [
        _paper("f-1", "2024-02", "Big impact paper", "big impact methods", published_date="2024-02-15"),
        _paper("f-2", "2024-03", "Tiny unknown paper", "tiny unknown ideas", published_date="2024-03-01"),
    ]
    # Only predict the unpopular one
    predictions = [_prediction(1, "Tiny unknown paper", "tiny unknown ideas")]
    popularity_weights = {"f-1": 1.0, "f-2": 0.1}

    scored = score_prediction_list(
        predictions=predictions,
        train_papers=train,
        future_papers=future,
        k=1,
        popularity_weights=popularity_weights,
    )

    # popularity_recall = 0.1 / (1.0 + 0.1) ≈ 0.0909
    assert scored.evaluation.popularity_recall_at_k == pytest.approx(0.1 / 1.1, rel=1e-3)
    # But hit_at_k is still 1.0 (matched something)
    assert scored.evaluation.hit_at_k == 1.0


def test_matched_paper_popularity_stored_in_match_detail() -> None:
    """PredictionMatchDetail should store the popularity of the matched paper."""
    train = [_paper("train-1", "2024-01", "Baseline", "baseline", published_date="2024-01-01")]
    future = [_paper("f-1", "2024-02", "Diffusion models", "diffusion models review", published_date="2024-02-15")]
    predictions = [_prediction(1, "Diffusion models", "diffusion models review")]
    popularity_weights = {"f-1": 0.75}

    scored = score_prediction_list(
        predictions=predictions,
        train_papers=train,
        future_papers=future,
        k=1,
        popularity_weights=popularity_weights,
    )

    assert len(scored.matches) == 1
    assert scored.matches[0].is_match is True
    assert scored.matches[0].matched_paper_popularity == pytest.approx(0.75)


def test_score_prediction_list_exposes_match_details_and_unmatched_papers() -> None:
    train = [_paper("train-1", "2024-01", "Old baseline", "old baseline methods", published_date="2024-01-01")]
    future = [
        _paper(
            "future-1",
            "2024-02",
            "Retrieval agents",
            "retrieval agents for planning",
            published_date="2024-02-15",
        )
    ]
    predictions = [_prediction(1, "Retrieval agents", "retrieval agents for planning")]

    scored = score_prediction_list(
        predictions=predictions,
        train_papers=train,
        future_papers=future,
        k=1,
        cutoff_date="2024-02-01",
        future_end_date="2024-03-31",
    )

    assert scored.evaluation.matched_paper_ids == ["future-1"]
    assert len(scored.matches) == 1
    assert scored.matches[0].paper_id == "future-1"
    assert scored.matches[0].is_match is True
    assert scored.unmatched_future_paper_ids == []
