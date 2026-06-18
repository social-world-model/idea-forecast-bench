import dataclasses
from pathlib import Path
from typing import List

import pytest

import live_idea_bench.similarity as similarity_module
from live_idea_bench.backtest import BacktestConfig, backtest, evaluate, generate, load_papers_from_markdown
from live_idea_bench.config import load_similarity_config
from live_idea_bench.strategy import create_strategy


@pytest.fixture(autouse=True)
def _force_hybrid_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the backtest tests hermetic: the shipped default engine is
    ``embedding`` (Voyage, needs a key), but these tests run offline and only
    exercise the rolling/metric machinery, so pin the matcher to ``hybrid``."""

    def _hybrid_config(*args, **kwargs):
        cfg = load_similarity_config(*args, **kwargs)
        return dataclasses.replace(cfg, engine="hybrid")

    monkeypatch.setattr(similarity_module, "load_similarity_config", _hybrid_config)


def _setup_mock_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "arxiv_csml" / "raw_markdown"
    data_dir.mkdir(parents=True)
    # Create 24 months of data from 2024-01 to 2025-12
    for year in [2024, 2025]:
        for month in range(1, 13):
            month_str = f"{year}-{month:02d}"
            p = data_dir / month_str / f"paper_{month_str}.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            # Add stop terms and 5 repeating terms to ensure 5 predictions
            keywords = ["cs.ml", "deep learning"]
            for i in range(5):
                keywords.append(f"trend_term_{i}")

            content = (
                f"# Paper {month_str}\n\n"
                f"Paper ID: paper_{month_str}\n"
                f"Date: {month_str}\n"
                f"Keywords: {', '.join(keywords)}\n\n"
                "# Summary\n\n"
                f"This is a summary for {month_str}.\n"
            )
            p.write_text(content, encoding="utf-8")
    return data_dir


def test_load_papers_from_markdown(tmp_path: Path) -> None:
    data_dir = _setup_mock_data(tmp_path)
    papers = load_papers_from_markdown(data_dir)
    assert len(papers) >= 24
    assert papers[0].month == "2024-01"
    assert papers[-1].month == "2025-12"
    assert papers[0].paper_id
    assert papers[0].summary


def test_strategy_generate_filters_generic_terms(tmp_path: Path) -> None:
    data_dir = _setup_mock_data(tmp_path)
    papers = load_papers_from_markdown(data_dir)
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
    assert "deep learning" not in lead_terms


def test_evaluate_at_cutoff_metric_bounds(tmp_path: Path) -> None:
    data_dir = _setup_mock_data(tmp_path)
    papers = load_papers_from_markdown(data_dir)
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
    assert 0.0 <= result.coverage_at_k <= 1.0
    assert 0.0 <= result.precision_at_k <= 1.0
    assert 0.0 <= result.mrr <= 1.0
    assert 0.0 <= result.novelty <= 1.0
    assert 0.0 <= result.diversity <= 1.0


def test_backtest_returns_windows_and_summary(tmp_path: Path) -> None:
    data_dir = _setup_mock_data(tmp_path)
    papers = load_papers_from_markdown(data_dir)
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
    assert "cutoff_date" in windows[0]
    assert "future_end_date" in windows[0]
    # train_paper_ids must be serialized for the citation/coauthor validity checks.
    assert "train_paper_ids" in windows[0]
    assert isinstance(windows[0]["train_paper_ids"], list)


def test_create_strategy_invalid_name() -> None:
    with pytest.raises(ValueError):
        create_strategy("not_a_strategy")
