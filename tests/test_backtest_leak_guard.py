from __future__ import annotations

from live_idea_bench.backtest import generate_at_cutoff, split_train_future_by_cutoff
from live_idea_bench.models import PaperRecord
from live_idea_bench.strategy.keyword_trend import KeywordTrendStrategy


def test_same_month_future_paper_not_in_train_for_cutoff_date() -> None:
    papers = [
        PaperRecord(
            paper_id="p-aug",
            title="Aug paper",
            month="2024-08",
            summary="old",
            keywords=["stable_term"],
            source_path="/fake/p-aug.md",
            published_date="2024-08-25",
        ),
        PaperRecord(
            paper_id="p-sep-01",
            title="Sep first",
            month="2024-09",
            summary="on cutoff day",
            keywords=["visible_term"],
            source_path="/fake/p-sep-01.md",
            published_date="2024-09-01",
        ),
        PaperRecord(
            paper_id="p-sep-22",
            title="Sep late",
            month="2024-09",
            summary="future in same month",
            keywords=["leaked_future_term"],
            source_path="/fake/p-sep-22.md",
            published_date="2024-09-22",
        ),
    ]

    strategy = KeywordTrendStrategy(recent_months=3, min_keyword_freq=1)
    predictions = generate_at_cutoff(
        papers=papers,
        strategy=strategy,
        cutoff_date="2024-09-01",
        top_k=10,
    )
    all_terms = [term for pred in predictions for term in pred.key_terms]
    assert "leaked_future_term" not in all_terms

    train, future, _future_end_month, _future_end_date = split_train_future_by_cutoff(
        papers=papers,
        cutoff_date="2024-09-01",
        horizon_months=1,
    )
    assert {paper.paper_id for paper in train} == {"p-aug", "p-sep-01"}
    assert {paper.paper_id for paper in future} == {"p-sep-22"}
