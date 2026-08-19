"""Tests for FutureIndex / HistoryIndex + cutoff orchestrator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forecaster.foresight.indices import (
    FutureIndex,
    HashingEmbedder,
    build_cutoff_indices,
    build_future_index,
    build_history_index,
)
from live_idea_bench.models import PaperRecord


def _paper(pid: str, date: str, topic: str = "x") -> PaperRecord:
    return PaperRecord(
        paper_id=pid,
        title=f"Paper {pid}",
        month=date[:7],
        summary=f"Abstract about {topic}. " + " ".join([topic] * 20),
        keywords=[topic],
        source_path="",
        published_date=date,
    )


def test_future_index_round_trip(tmp_path: Path):
    emb = HashingEmbedder(dim=64, seed=1)
    papers = [_paper("a", "2024-05-01", "rag"), _paper("b", "2024-06-01", "agents")]
    idx = build_future_index(papers, emb, cutoff_date="2024-04-30")
    assert idx.size == 2
    assert idx.embeddings.shape == (2, 64)

    saved = idx.save(tmp_path / "future.npz")
    assert saved.exists()
    assert saved.with_suffix(".meta.json").exists()

    loaded = FutureIndex.load(tmp_path / "future.npz")
    assert loaded.paper_ids == idx.paper_ids
    assert loaded.kind == "future"
    np.testing.assert_allclose(loaded.embeddings, idx.embeddings, rtol=1e-5)


def test_search_returns_topk_in_descending_order():
    emb = HashingEmbedder(dim=64, seed=2)
    papers = [
        _paper("p_rag1", "2024-05-01", "rag"),
        _paper("p_rag2", "2024-05-15", "rag"),
        _paper("p_agents", "2024-06-01", "agents"),
    ]
    idx = build_future_index(papers, emb, cutoff_date="2024-04-30")
    q = emb.encode(["rag rag rag rag rag rag"])[0]
    hits = idx.search(q, top_k=2)
    assert len(hits) == 2
    ids = [pid for pid, _ in hits]
    assert ids[0].startswith("p_rag")
    # scores descending
    assert hits[0][1] >= hits[1][1]


def test_future_index_refuses_test_window_papers():
    emb = HashingEmbedder(dim=32, seed=3)
    papers = [
        _paper("ok", "2024-09-15", "rag"),
        _paper("bad", "2024-10-15", "rag"),  # crosses the hard limit
    ]
    with pytest.raises(AssertionError):
        build_future_index(papers, emb, cutoff_date="2024-07-01")


def test_history_index_does_not_assert_test_window():
    """History (X_<=t) is allowed to contain late papers (still <=t)."""
    emb = HashingEmbedder(dim=32, seed=4)
    papers = [_paper("p", "2024-05-01", "rag")]
    idx = build_history_index(papers, emb, cutoff_date="2024-06-01")
    assert idx.size == 1
    assert idx.kind == "history"


def test_build_cutoff_indices_handles_two_cutoffs():
    emb = HashingEmbedder(dim=32, seed=5)
    papers = [
        _paper("hist_1", "2023-02-01", "rag"),
        _paper("hist_2", "2023-03-15", "rag"),
        _paper("fut_1", "2023-05-01", "rag"),  # in future of cutoff 2023-03
        _paper("fut_2", "2023-06-15", "agents"),
        _paper("late", "2024-09-30", "agents"),  # in future of cutoff 2024-06
    ]
    bundles = build_cutoff_indices(
        papers=papers,
        cutoff_dates=["2023-03-31", "2024-06-30"],
        horizon_months=3,
        embedder=emb,
    )
    assert set(bundles.keys()) == {"2023-03-31", "2024-06-30"}
    early = bundles["2023-03-31"]
    assert early.history.size == 2
    assert "fut_1" in early.future.paper_ids
    late = bundles["2024-06-30"]
    assert "late" in late.future.paper_ids


def test_empty_future_returns_empty_index():
    emb = HashingEmbedder(dim=16, seed=6)
    idx = build_future_index([], emb, cutoff_date="2024-01-01")
    assert idx.size == 0
    assert idx.search(np.zeros(16), top_k=5) == []
