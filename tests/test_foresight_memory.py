"""Tests for build_memory(papers_before_t) -> str."""
from __future__ import annotations

from forecaster.foresight.memory import build_memory
from live_idea_bench.models import PaperRecord


def _make(paper_id: str, *, month: str, topic: str, summary: str = "") -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        month=month,
        summary=summary or f"Summary about {topic} work as of {month}.",
        keywords=[topic],
        source_path="",
        published_date=f"{month}-15",
        metadata={"topic_id": topic},
    )


def test_empty_papers_returns_empty_string():
    assert build_memory([], cutoff_t="2024-06-30") == ""


def test_groups_by_topic_and_lists_counts():
    papers = [
        _make("p1", month="2024-04", topic="rag"),
        _make("p2", month="2024-05", topic="rag"),
        _make("p3", month="2024-05", topic="agents"),
    ]
    out = build_memory(papers, cutoff_t="2024-06-30")
    assert "rag" in out
    assert "agents" in out
    # rag bucket should appear before agents (higher count, both recent)
    assert out.index("rag") < out.index("agents")
    assert "papers=3" in out  # header carries the total


def test_recency_window_orders_buckets():
    """Bucket with more recent papers should rank above bucket with stale ones."""
    papers = [
        _make("p_stale_1", month="2023-01", topic="legacy"),
        _make("p_stale_2", month="2023-02", topic="legacy"),
        _make("p_stale_3", month="2023-03", topic="legacy"),
        _make("p_fresh", month="2024-05", topic="fresh"),
    ]
    out = build_memory(papers, cutoff_t="2024-06-30", recency_window_months=6)
    # "fresh" has 1 recent paper; "legacy" has 0 recent papers (older than 6mo).
    assert out.index("fresh") < out.index("legacy")


def test_max_entries_caps_output():
    papers = [
        _make(f"p{i}", month="2024-05", topic=f"t{i}") for i in range(50)
    ]
    out = build_memory(papers, cutoff_t="2024-06-30", max_entries=5)
    bullet_count = sum(1 for ln in out.splitlines() if ln.startswith("- "))
    assert bullet_count == 5


def test_output_is_deterministic_under_reorder():
    papers = [
        _make("a", month="2024-04", topic="rag"),
        _make("b", month="2024-05", topic="agents"),
        _make("c", month="2024-05", topic="rag"),
    ]
    out1 = build_memory(papers, cutoff_t="2024-06-30")
    out2 = build_memory(list(reversed(papers)), cutoff_t="2024-06-30")
    assert out1 == out2
