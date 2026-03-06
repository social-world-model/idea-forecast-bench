from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend import strategy_store
from backend.services import daily_pipeline


def _isolate_strategy_store(monkeypatch, tmp_path: Path) -> None:
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(strategy_store, "STRATEGIES_DIR", strategies_dir)
    monkeypatch.setattr(strategy_store, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(daily_pipeline.strategy_store, "PROJECT_ROOT", tmp_path)


def _write_markdown(path: Path, *, paper_id: str, title: str, date: str, keywords: list[str], summary: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keywords_yaml = "\n".join(f"  - \"{kw}\"" for kw in keywords) if keywords else "  - \"\""
    content = (
        "---\n"
        f"paper_id: \"{paper_id}\"\n"
        f"title: \"{title}\"\n"
        f"date: \"{date}\"\n"
        "keywords:\n"
        f"{keywords_yaml}\n"
        f"source_url: \"https://arxiv.org/abs/{paper_id}\"\n"
        "---\n\n"
        "# Abstract\n\n"
        f"{summary}\n"
    )
    path.write_text(content, encoding="utf-8")


def _fake_generation(monkeypatch) -> None:
    def _run_generation(strategy_id: str, cutoff_date: str | None = None) -> None:
        resolved_cutoff_date = cutoff_date or "2026-03-01"
        strategy_store.update_strategy(
            strategy_id,
            {
                "generation_status": "done",
                "generation": {
                    "cutoff_date": resolved_cutoff_date,
                    "cutoff_month": resolved_cutoff_date[:7],
                    "predictions": [
                        {
                            "rank": 1,
                            "title": "Daily Prediction",
                            "rationale": "auto",
                            "key_terms": ["agent"],
                            "confidence": 0.8,
                        }
                    ],
                },
            },
        )

    monkeypatch.setattr(strategy_store, "run_generation_sync", _run_generation)
    monkeypatch.setattr(daily_pipeline.strategy_store, "run_generation_sync", _run_generation)


def test_daily_pipeline_updates_daily_eval_and_leaderboard(monkeypatch, tmp_path) -> None:
    _isolate_strategy_store(monkeypatch, tmp_path)
    _fake_generation(monkeypatch)

    raw_dir = tmp_path / "raw_markdown"
    _write_markdown(
        raw_dir / "2026-02" / "p-old.md",
        paper_id="p-old",
        title="Old",
        date="2026-02-10",
        keywords=["agent"],
        summary="old paper",
    )
    _write_markdown(
        raw_dir / "2026-03" / "p-new.md",
        paper_id="p-new",
        title="New",
        date="2026-03-01",
        keywords=["agent"],
        summary="new paper",
    )

    s = strategy_store.create_strategy(
        {
            "strategy_name": "keyword_trend",
            "config": {
                "top_k": 5,
                "horizon_months": 3,
                "min_train_papers": 1,
                "start_month": "2026-01",
                "end_month": "2026-02",
                "data_dir": str(raw_dir),
            },
        }
    )
    strategy_store.update_strategy(
        s["id"],
        {
            "generation_status": "done",
            "generation": {
                "cutoff_month": "2026-02",
                "predictions": [
                    {
                        "rank": 1,
                        "title": "Old prediction",
                        "rationale": "old",
                        "key_terms": ["agent"],
                        "confidence": 0.9,
                    }
                ],
            },
        },
    )

    monkeypatch.setattr(
        daily_pipeline,
        "ingest_latest_arxiv_papers",
        lambda **kwargs: {
            "data_dir": str(raw_dir),
            "new_papers": [{"paper_id": "p-new"}],
            "fetched_count": 1,
            "ingested_count": 1,
        },
    )

    report = daily_pipeline.run_daily_pipeline(
        now=datetime(2026, 3, 3, 5, 0, 0, tzinfo=timezone.utc),
        data_dir=raw_dir,
    )
    assert report["strategies_processed"] == 1

    updated = strategy_store.get_strategy(s["id"])
    assert updated is not None
    assert updated["daily_evaluation"]["prediction_cutoff_date"] == "2026-02-01"
    assert updated["daily_evaluation"]["new_papers_count"] == 1
    assert updated["daily_evaluation"]["hit_at_k"] == 1.0
    assert updated["leaderboard_score"] > 0.0
    assert updated["last_generation_cutoff_month"] == "2026-03"
    assert updated["generation"]["cutoff_date"] == "2026-03-03"


def test_daily_pipeline_skips_eval_without_previous_generation(monkeypatch, tmp_path) -> None:
    _isolate_strategy_store(monkeypatch, tmp_path)
    _fake_generation(monkeypatch)

    raw_dir = tmp_path / "raw_markdown"
    _write_markdown(
        raw_dir / "2026-03" / "p-new.md",
        paper_id="p-new",
        title="New",
        date="2026-03-01",
        keywords=["agent"],
        summary="new paper",
    )
    s = strategy_store.create_strategy(
        {
            "strategy_name": "keyword_trend",
            "config": {
                "start_month": "2026-01",
                "end_month": "2026-03",
                "data_dir": str(raw_dir),
            },
        }
    )

    monkeypatch.setattr(
        daily_pipeline,
        "ingest_latest_arxiv_papers",
        lambda **kwargs: {"data_dir": str(raw_dir), "new_papers": []},
    )

    daily_pipeline.run_daily_pipeline(
        now=datetime(2026, 3, 3, 5, 0, 0, tzinfo=timezone.utc),
        data_dir=raw_dir,
    )

    updated = strategy_store.get_strategy(s["id"])
    assert updated is not None
    assert updated["daily_evaluation"] is None
    assert updated["generation_status"] == "done"


def test_daily_pipeline_writes_zero_hit_when_no_new_papers(monkeypatch, tmp_path) -> None:
    _isolate_strategy_store(monkeypatch, tmp_path)
    _fake_generation(monkeypatch)

    raw_dir = tmp_path / "raw_markdown"
    _write_markdown(
        raw_dir / "2026-02" / "p-old.md",
        paper_id="p-old",
        title="Old",
        date="2026-02-10",
        keywords=["agent"],
        summary="old paper",
    )
    s = strategy_store.create_strategy(
        {
            "strategy_name": "keyword_trend",
            "config": {
                "start_month": "2026-01",
                "end_month": "2026-02",
                "data_dir": str(raw_dir),
            },
        }
    )
    strategy_store.update_strategy(
        s["id"],
        {
            "generation_status": "done",
            "generation": {
                "cutoff_month": "2026-02",
                "predictions": [
                    {
                        "rank": 1,
                        "title": "Old prediction",
                        "rationale": "old",
                        "key_terms": ["agent"],
                        "confidence": 0.9,
                    }
                ],
            },
        },
    )

    monkeypatch.setattr(
        daily_pipeline,
        "ingest_latest_arxiv_papers",
        lambda **kwargs: {"data_dir": str(raw_dir), "new_papers": []},
    )

    daily_pipeline.run_daily_pipeline(
        now=datetime(2026, 3, 3, 5, 0, 0, tzinfo=timezone.utc),
        data_dir=raw_dir,
    )
    updated = strategy_store.get_strategy(s["id"])
    assert updated is not None
    assert updated["daily_evaluation"]["new_papers_count"] == 0
    assert updated["daily_evaluation"]["hit_at_k"] == 0.0
    assert updated["daily_evaluation"]["prediction_cutoff_date"] == "2026-02-01"


def test_daily_pipeline_lock_prevents_concurrent_runs(monkeypatch, tmp_path) -> None:
    _isolate_strategy_store(monkeypatch, tmp_path)
    lock_file = tmp_path / "data" / "daily_runs" / "pipeline.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(
        f"pid={os.getpid()} created_at={datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )

    with pytest.raises(daily_pipeline.PipelineAlreadyRunningError):
        daily_pipeline.run_daily_pipeline(
            now=datetime(2026, 3, 3, 5, 0, 0, tzinfo=timezone.utc),
            data_dir=tmp_path / "raw_markdown",
        )


def test_daily_pipeline_reclaims_stale_lock(monkeypatch, tmp_path) -> None:
    _isolate_strategy_store(monkeypatch, tmp_path)
    _fake_generation(monkeypatch)

    lock_file = tmp_path / "data" / "daily_runs" / "pipeline.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(
        "pid=999999 created_at=2020-01-01T00:00:00+00:00\n",
        encoding="utf-8",
    )

    raw_dir = tmp_path / "raw_markdown"
    raw_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        daily_pipeline,
        "ingest_latest_arxiv_papers",
        lambda **kwargs: {"data_dir": str(raw_dir), "new_papers": []},
    )

    report = daily_pipeline.run_daily_pipeline(
        now=datetime(2026, 3, 3, 5, 0, 0, tzinfo=timezone.utc),
        data_dir=raw_dir,
    )

    assert report["strategies_processed"] == 0
    assert not lock_file.exists()
