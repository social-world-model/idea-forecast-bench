"""Tests for live_idea_bench/popularity.py — written BEFORE implementation (TDD RED phase)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from live_idea_bench.models import PaperRecord
from live_idea_bench.popularity import (
    compute_popularity_weight,
    enrich_papers_with_popularity,
    fetch_popularity_batch,
    load_popularity_cache,
    normalize_popularity_scores,
    save_popularity_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _paper(paper_id: str, title: str = "A Paper") -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        month="2024-01",
        summary="summary",
        keywords=["cs.AI"],
        source_path=f"/fake/{paper_id}.md",
        published_date="2024-01-01",
    )


def _s2_response(data: list[dict]) -> MagicMock:
    """Build a mock requests.Response for Semantic Scholar batch API."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# normalize_popularity_scores
# ---------------------------------------------------------------------------


def test_normalize_popularity_scores_basic() -> None:
    raw = {"a": 0, "b": 10, "c": 100}
    result = normalize_popularity_scores(raw)
    assert result["a"] == pytest.approx(0.0)
    assert result["b"] == pytest.approx(0.1)
    assert result["c"] == pytest.approx(1.0)


def test_normalize_popularity_scores_all_equal() -> None:
    raw = {"a": 5, "b": 5, "c": 5}
    result = normalize_popularity_scores(raw)
    # When all equal, everything gets 1.0 (no variation to penalize)
    assert all(v == pytest.approx(1.0) for v in result.values())


def test_normalize_popularity_scores_empty() -> None:
    assert normalize_popularity_scores({}) == {}


def test_normalize_popularity_scores_single_paper() -> None:
    result = normalize_popularity_scores({"only": 42})
    # Single paper gets 1.0 (maximum by convention)
    assert result["only"] == pytest.approx(1.0)


def test_normalize_popularity_scores_zero_citations() -> None:
    raw = {"a": 0, "b": 0, "c": 0}
    result = normalize_popularity_scores(raw)
    assert all(v == pytest.approx(1.0) for v in result.values())


# ---------------------------------------------------------------------------
# compute_popularity_weight
# ---------------------------------------------------------------------------


def test_compute_popularity_weight_floor() -> None:
    # score=0.0 → should return the floor, not 0
    weight = compute_popularity_weight(0.0, floor=0.1)
    assert weight == pytest.approx(0.1)


def test_compute_popularity_weight_max() -> None:
    weight = compute_popularity_weight(1.0, floor=0.1)
    assert weight == pytest.approx(1.0)


def test_compute_popularity_weight_midpoint() -> None:
    weight = compute_popularity_weight(0.5, floor=0.1)
    # floor + (1.0 - floor) * 0.5 = 0.1 + 0.45 = 0.55
    assert weight == pytest.approx(0.55)


def test_compute_popularity_weight_default_floor() -> None:
    # Default floor is 0.1
    weight = compute_popularity_weight(0.0)
    assert weight == pytest.approx(0.1)


def test_compute_popularity_weight_clamps_input() -> None:
    # Scores outside [0,1] should be clamped
    assert compute_popularity_weight(-0.5) == pytest.approx(0.1)
    assert compute_popularity_weight(1.5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# load_popularity_cache / save_popularity_cache
# ---------------------------------------------------------------------------


def test_save_and_load_popularity_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "pop_cache.json"
    data = {"2401.12345": {"citation_count": 42, "fetched_at": "2026-01-01T00:00:00Z"}}
    save_popularity_cache(cache_path, data)
    loaded = load_popularity_cache(cache_path)
    assert loaded["2401.12345"]["citation_count"] == 42


def test_load_popularity_cache_missing_file(tmp_path: Path) -> None:
    result = load_popularity_cache(tmp_path / "nonexistent.json")
    assert result == {}


def test_load_popularity_cache_corrupt_file(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("not valid json!!!", encoding="utf-8")
    result = load_popularity_cache(p)
    assert result == {}


# ---------------------------------------------------------------------------
# fetch_popularity_batch
# ---------------------------------------------------------------------------


def test_fetch_popularity_batch_from_api(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    paper_ids = ["2401.00001", "2401.00002"]

    s2_data = [
        {"paperId": "s2-1", "externalIds": {"ArXiv": "2401.00001"}, "citationCount": 10},
        {"paperId": "s2-2", "externalIds": {"ArXiv": "2401.00002"}, "citationCount": 25},
    ]

    with patch("live_idea_bench.popularity.requests.post", return_value=_s2_response(s2_data)):
        result = fetch_popularity_batch(paper_ids, cache_path=cache_path)

    assert result["2401.00001"] == 10
    assert result["2401.00002"] == 25


def test_fetch_popularity_batch_uses_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    # Pre-populate cache
    existing = {
        "2401.00001": {"citation_count": 99, "fetched_at": "2026-01-01T00:00:00Z"},
    }
    save_popularity_cache(cache_path, existing)

    paper_ids = ["2401.00001"]
    with patch("live_idea_bench.popularity.requests.post") as mock_post:
        result = fetch_popularity_batch(paper_ids, cache_path=cache_path)
        mock_post.assert_not_called()  # Should not hit API for cached paper

    assert result["2401.00001"] == 99


def test_fetch_popularity_batch_partial_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    # Only one paper cached
    existing = {
        "2401.00001": {"citation_count": 5, "fetched_at": "2026-01-01T00:00:00Z"},
    }
    save_popularity_cache(cache_path, existing)

    paper_ids = ["2401.00001", "2401.00002"]
    s2_data = [
        {"paperId": "s2-2", "externalIds": {"ArXiv": "2401.00002"}, "citationCount": 15},
    ]
    with patch("live_idea_bench.popularity.requests.post", return_value=_s2_response(s2_data)):
        result = fetch_popularity_batch(paper_ids, cache_path=cache_path)

    assert result["2401.00001"] == 5
    assert result["2401.00002"] == 15


def test_fetch_popularity_batch_api_failure_returns_zero(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    paper_ids = ["2401.99999"]

    with patch("live_idea_bench.popularity.requests.post", side_effect=Exception("network error")):
        result = fetch_popularity_batch(paper_ids, cache_path=cache_path)

    # Graceful fallback: paper gets 0 citations
    assert result.get("2401.99999", 0) == 0


def test_fetch_popularity_batch_saves_to_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    paper_ids = ["2401.00001"]
    s2_data = [
        {"paperId": "s2-1", "externalIds": {"ArXiv": "2401.00001"}, "citationCount": 7},
    ]
    with patch("live_idea_bench.popularity.requests.post", return_value=_s2_response(s2_data)):
        fetch_popularity_batch(paper_ids, cache_path=cache_path)

    # Verify cache was updated
    loaded = load_popularity_cache(cache_path)
    assert loaded["2401.00001"]["citation_count"] == 7


def test_fetch_popularity_batch_no_cache_path() -> None:
    """When cache_path is None, should still work (no caching)."""
    paper_ids = ["2401.00001"]
    s2_data = [
        {"paperId": "s2-1", "externalIds": {"ArXiv": "2401.00001"}, "citationCount": 3},
    ]
    with patch("live_idea_bench.popularity.requests.post", return_value=_s2_response(s2_data)):
        result = fetch_popularity_batch(paper_ids, cache_path=None)
    assert result["2401.00001"] == 3


# ---------------------------------------------------------------------------
# enrich_papers_with_popularity
# ---------------------------------------------------------------------------


def test_enrich_papers_with_popularity_returns_weights(tmp_path: Path) -> None:
    papers = [
        _paper("2401.00001"),
        _paper("2401.00002"),
    ]
    # Pre-populate cache with citation counts
    cache_path = tmp_path / "cache.json"
    save_popularity_cache(cache_path, {
        "2401.00001": {"citation_count": 0, "fetched_at": "2026-01-01T00:00:00Z"},
        "2401.00002": {"citation_count": 100, "fetched_at": "2026-01-01T00:00:00Z"},
    })

    with patch("live_idea_bench.popularity.requests.post"):
        weights = enrich_papers_with_popularity(papers, cache_path=cache_path)

    # Weights should be in [floor, 1.0]
    assert 0.0 < weights["2401.00001"] <= 1.0
    assert 0.0 < weights["2401.00002"] <= 1.0
    # Higher citation count → higher weight
    assert weights["2401.00002"] > weights["2401.00001"]


def test_enrich_papers_with_popularity_empty_list(tmp_path: Path) -> None:
    result = enrich_papers_with_popularity([], cache_path=None)
    assert result == {}


def test_enrich_papers_with_popularity_all_zero_citations(tmp_path: Path) -> None:
    papers = [_paper("2401.00001"), _paper("2401.00002")]
    cache_path = tmp_path / "cache.json"
    save_popularity_cache(cache_path, {
        "2401.00001": {"citation_count": 0, "fetched_at": "2026-01-01T00:00:00Z"},
        "2401.00002": {"citation_count": 0, "fetched_at": "2026-01-01T00:00:00Z"},
    })
    with patch("live_idea_bench.popularity.requests.post"):
        weights = enrich_papers_with_popularity(papers, cache_path=cache_path)
    # When all zero, everyone gets max weight (all equal, normalized to 1.0)
    assert all(w == pytest.approx(1.0) for w in weights.values())
