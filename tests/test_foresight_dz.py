"""Tests for D_z augmentation (hindsight JSONL -> D_z)."""

from __future__ import annotations

import json
from pathlib import Path

from forecaster.foresight.dz import augment_hindsight_rows, load_dz_rows
from live_idea_bench.models import PaperRecord


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_augment_drops_test_window_rows_and_maps_operator(tmp_path: Path):
    raw = [
        {
            "topic_id": "rag",
            "episode_id": "E1",
            "cutoff_date": "2024-06-30",
            "future_paper_id": "p_future",
            "future_paper_title": "Some Future Paper",
            "future_paper_published_date": "2024-08-15",
            "innovation": {"base_direction": "rag", "operator": "extend", "gap": "x"},
            "context_paper_count": 100,
        },
        {
            # Test-window cutoff → should be dropped.
            "topic_id": "rag",
            "episode_id": "E2",
            "cutoff_date": "2024-11-30",
            "future_paper_id": "p_test",
            "future_paper_title": "Test-window Future Paper",
            "future_paper_published_date": "2025-01-15",
            "innovation": {"base_direction": "rag", "operator": "compose", "gap": "y"},
            "context_paper_count": 100,
        },
        {
            # Free-text operator not in mapping → goes to "other".
            "topic_id": "agents",
            "episode_id": "E1",
            "cutoff_date": "2024-04-30",
            "future_paper_id": "p_other",
            "future_paper_title": "Analyze Paper",
            "future_paper_published_date": "2024-06-10",
            "innovation": {
                "base_direction": "agents",
                "operator": "analyze",
                "gap": "z",
            },
            "context_paper_count": 100,
        },
    ]
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "dz.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(in_path, raw)

    summary = augment_hindsight_rows(
        in_path,
        out_path,
        summary_path=summary_path,
    )
    rows = load_dz_rows(out_path)

    assert summary.total_rows == 3
    assert summary.dropped_test_window == 1
    assert summary.train_window_rows == 2
    assert len(rows) == 2

    by_id = {r["source_future_id"]: r for r in rows}
    assert by_id["p_future"]["operator_closed"] == "limitation_extension"
    assert by_id["p_other"]["operator_closed"] == "other"
    # When no corpus is supplied these keys are omitted from the JSONL row.
    assert by_id["p_future"].get("context_paper_ids") is None
    assert by_id["p_future"].get("memory_text") is None

    sum_data = json.loads(summary_path.read_text())
    assert sum_data["dropped_test_window"] == 1
    assert sum_data["other_ratio"] == 0.5  # 1 of the 2 kept rows is "other"


def test_augment_fills_memory_when_corpus_supplied(tmp_path: Path):
    raw = [
        {
            "topic_id": "rag",
            "episode_id": "E1",
            "cutoff_date": "2024-06-30",
            "future_paper_id": "p_future",
            "future_paper_title": "Future Paper",
            "future_paper_published_date": "2024-08-15",
            "innovation": {"base_direction": "rag", "operator": "extend", "gap": "x"},
            "context_paper_count": 2,
        }
    ]
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "dz.jsonl"
    _write_jsonl(in_path, raw)

    papers_by_id = {
        "ctx_1": PaperRecord(
            paper_id="ctx_1",
            title="Context one",
            month="2024-04",
            summary="A retrieval-augmented generation method.",
            keywords=["rag"],
            source_path="",
            published_date="2024-04-15",
            metadata={"topic_id": "rag"},
        ),
        "ctx_2": PaperRecord(
            paper_id="ctx_2",
            title="Context two",
            month="2024-05",
            summary="Agent planning over tool graphs.",
            keywords=["agents"],
            source_path="",
            published_date="2024-05-20",
            metadata={"topic_id": "agents"},
        ),
        "p_future": PaperRecord(  # in the future window — must NOT enter context
            paper_id="p_future",
            title="Future Paper",
            month="2024-08",
            summary="The future thing.",
            keywords=["rag"],
            source_path="",
            published_date="2024-08-15",
            metadata={"topic_id": "rag"},
        ),
    }
    augment_hindsight_rows(in_path, out_path, papers_by_id=papers_by_id)
    rows = load_dz_rows(out_path)
    row = rows[0]
    assert sorted(row["context_paper_ids"]) == ["ctx_1", "ctx_2"]
    assert "p_future" not in row["context_paper_ids"]
    assert "rag" in row["memory_text"]
    assert "agents" in row["memory_text"]
