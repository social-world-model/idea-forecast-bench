from __future__ import annotations

from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.strategy.predictor_llm import PredictorLLMStrategy


def _paper(paper_id: str, month: str, title: str, keywords: list[str]) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        month=month,
        summary="summary",
        keywords=keywords,
        source_path=f"/fake/{paper_id}.md",
    )


def test_predictor_llm_strategy_generate_happy_path(monkeypatch) -> None:
    import live_idea_bench.strategy.predictor_llm as predictor_module

    calls: dict[str, object] = {}

    def _fake_generate_predictions(**kwargs):  # type: ignore[no-untyped-def]
        calls["generate_kwargs"] = kwargs
        return [
            IdeaPrediction(
                rank=1,
                title="Causal adapters for sparse domains",
                rationale="Improve transfer with causal bottlenecks.",
                approach="Train sparse adapters with causal regularization.",
                score=0.86,
                key_terms=["causal", "adapter"],
                confidence=0.86,
            ),
            IdeaPrediction(
                rank=2,
                title="Continual retrieval alignment",
                rationale="Current retrievers drift over time. Jointly train memory and retriever updates.",
                approach="Update retriever memory jointly with continual supervision.",
                score=0.8,
                key_terms=["retrieval", "continual-learning"],
                confidence=0.8,
            ),
        ]

    monkeypatch.setattr(
        predictor_module,
        "generate_predictions",
        _fake_generate_predictions,
    )

    strategy = PredictorLLMStrategy(
        model_name="gpt-4o-mini",
        predictor_config="predictor.yaml",
        similarity_config="similarity.yaml",
    )
    train_papers = [
        _paper("p1", "2024-04", "Title A", ["causal", "adapter"]),
        _paper("p2", "2024-05", "Title B", ["retrieval", "alignment"]),
    ]

    predictions = strategy.generate(
        train_papers=train_papers,
        cutoff_month="2024-06",
        top_k=2,
    )

    assert len(predictions) == 2
    assert all(isinstance(item, IdeaPrediction) for item in predictions)
    assert [item.rank for item in predictions] == [1, 2]
    assert predictions[0].key_terms == ["causal", "adapter"]
    assert predictions[1].confidence == 0.8
    assert predictions[0].approach
    assert predictions[0].score == 0.86

    generate_kwargs = calls["generate_kwargs"]
    assert isinstance(generate_kwargs, dict)
    assert generate_kwargs["model_name"] == "gpt-4o-mini"
    assert generate_kwargs["predictor_config_path"] == "predictor.yaml"
    assert generate_kwargs["temperature"] is None
    assert generate_kwargs["cutoff_month"] == "2024-06"
    assert generate_kwargs["top_k"] == 2
    # fail-loud: the strategy must disable the heuristic template fallback.
    assert generate_kwargs["fallback_to_heuristic"] is False


def test_generate_predictions_fails_empty_not_heuristic_by_default(monkeypatch) -> None:
    """When the LLM never yields parseable ideas, generate_predictions must
    default to returning [] (fail-loud) — NOT silently fabricate lexical
    template ideas via the heuristic generator."""
    import live_idea_bench.predictor as predictor_module

    def _empty_llm(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("predictor LLM output could not be parsed")

    heuristic_called = {"hit": False}

    def _track_heuristic(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        heuristic_called["hit"] = True
        return [IdeaPrediction(rank=1, title="lexical", rationale="r")]

    monkeypatch.setattr(predictor_module, "_llm_predictions", _empty_llm)
    monkeypatch.setattr(predictor_module, "_heuristic_predictions", _track_heuristic)

    predictions = predictor_module.generate_predictions(
        train_papers=[_paper("p1", "2024-04", "Title A", ["a"])],
        cutoff_month="2024-06",
        top_k=3,
    )
    assert predictions == []
    assert heuristic_called["hit"] is False


def test_predictor_llm_strategy_generate_malformed_output_returns_empty(monkeypatch) -> None:
    import live_idea_bench.strategy.predictor_llm as predictor_module

    monkeypatch.setattr(
        predictor_module,
        "generate_predictions",
        lambda **_kwargs: [],
    )

    strategy = PredictorLLMStrategy(
        model_name="gpt-4o-mini",
        predictor_config="predictor.yaml",
        similarity_config="similarity.yaml",
    )

    predictions = strategy.generate(
        train_papers=[_paper("p1", "2024-04", "Title A", ["a"])],
        cutoff_month="2024-06",
        top_k=3,
    )
    assert predictions == []
