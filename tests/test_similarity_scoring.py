from __future__ import annotations

from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.similarity import evaluate_predictions, score_prediction_list


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
