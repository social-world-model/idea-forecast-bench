from pathlib import Path

import pytest

from src.backtest import BacktestConfig, backtest, evaluate, generate, load_papers_from_markdown
from src.strategy import create_strategy


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "arxiv_csml" / "raw_markdown"


def test_load_papers_from_markdown() -> None:
    papers = load_papers_from_markdown(DATA_DIR)
    assert len(papers) >= 20
    assert papers[0].month == "2024-01"
    assert papers[-1].month == "2025-12"
    assert papers[0].paper_id
    assert papers[0].summary


def test_strategy_generate_filters_generic_terms() -> None:
    papers = load_papers_from_markdown(DATA_DIR)
    strategy = create_strategy("keyword_trend", recent_months=3, min_keyword_freq=2)
    preds = generate(
        papers=papers,
        strategy=strategy,
        cutoff_month="2025-06",
        top_k=5,
    )
    assert len(preds) == 5
    lead_terms = [pred.key_terms[0] for pred in preds if pred.key_terms]
    assert "cs.ml" not in lead_terms


def test_evaluate_at_cutoff_metric_bounds() -> None:
    papers = load_papers_from_markdown(DATA_DIR)
    strategy = create_strategy("keyword_trend", recent_months=3, min_keyword_freq=2)
    result = evaluate(
        papers=papers,
        strategy=strategy,
        cutoff_month="2025-06",
        top_k=5,
        horizon_months=3,
    )
    assert 0.0 <= result.hit_at_k <= 1.0
    assert 0.0 <= result.recall_at_k <= 1.0
    assert 0.0 <= result.precision_at_k <= 1.0
    assert 0.0 <= result.mrr <= 1.0
    assert 0.0 <= result.novelty <= 1.0
    assert 0.0 <= result.diversity <= 1.0


def test_backtest_returns_windows_and_summary() -> None:
    papers = load_papers_from_markdown(DATA_DIR)
    strategy = create_strategy("keyword_trend", recent_months=3, min_keyword_freq=2)
    config = BacktestConfig(top_k=5, horizon_months=3, min_train_papers=4)
    report = backtest(papers=papers, strategy=strategy, config=config)

    summary = report["summary"]
    windows = report["windows"]

    assert summary["windows"] > 0
    assert len(windows) == summary["windows"]
    assert "avg_hit_at_k" in summary
    assert "avg_recall_at_k" in summary
    assert "avg_mrr" in summary


def test_create_strategy_invalid_name() -> None:
    with pytest.raises(ValueError):
        create_strategy("not_a_strategy")

