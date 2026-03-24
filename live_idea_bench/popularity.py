"""Popularity scoring for papers using the Semantic Scholar API.

Fetches citation counts and normalizes them into [0, 1] weights.
All functions gracefully degrade — API failures return zero citations.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from live_idea_bench.models import PaperRecord

logger = logging.getLogger(__name__)

_S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
_S2_FIELDS = "citationCount,externalIds"
_S2_BATCH_SIZE = 500
_S2_RATE_LIMIT_DELAY = 0.5  # seconds between batches
_DEFAULT_FLOOR = 0.1


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def load_popularity_cache(cache_path: Path) -> dict[str, Any]:
    """Load cached citation data from JSON. Returns empty dict on any error."""
    try:
        text = cache_path.read_text(encoding="utf-8")
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError):
        logger.warning("Popularity cache at %s is corrupt or unreadable; ignoring.", cache_path)
        return {}


def save_popularity_cache(cache_path: Path, data: dict[str, Any]) -> None:
    """Persist citation data dict to JSON at cache_path."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Normalization + weight computation
# ---------------------------------------------------------------------------


def normalize_popularity_scores(raw_counts: dict[str, int]) -> dict[str, float]:
    """Min-max normalize citation counts to [0, 1] within the batch.

    Special cases:
    - Empty dict → empty dict
    - Single paper → 1.0
    - All papers same count → all get 1.0
    """
    if not raw_counts:
        return {}

    min_c = min(raw_counts.values())
    max_c = max(raw_counts.values())

    if max_c == min_c:
        # No variation — everyone gets 1.0 (equally popular / no data)
        return {pid: 1.0 for pid in raw_counts}

    span = max_c - min_c
    return {pid: (count - min_c) / span for pid, count in raw_counts.items()}


def compute_popularity_weight(score: float, *, floor: float = _DEFAULT_FLOOR) -> float:
    """Convert a normalized [0, 1] score to a weight in [floor, 1.0].

    A floor > 0 prevents completely zeroing out obscure papers.
    Inputs outside [0, 1] are clamped.
    """
    clamped = max(0.0, min(1.0, score))
    return floor + (1.0 - floor) * clamped


# ---------------------------------------------------------------------------
# Semantic Scholar API
# ---------------------------------------------------------------------------


def _build_arxiv_ids(paper_ids: list[str]) -> list[str]:
    """Convert arXiv paper IDs to the ARXIV:{id} format S2 expects."""
    result = []
    for pid in paper_ids:
        if pid.startswith("ARXIV:"):
            result.append(pid)
        else:
            result.append(f"ARXIV:{pid}")
    return result


def _fetch_from_s2(arxiv_ids: list[str]) -> dict[str, int]:
    """Fetch citation counts from Semantic Scholar for a list of arXiv IDs.

    Returns a dict mapping raw arXiv ID → citation count.
    Returns empty dict on any error (graceful degradation).
    """
    try:
        response = requests.post(
            _S2_BATCH_URL,
            params={"fields": _S2_FIELDS},
            json={"ids": arxiv_ids},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.warning("Semantic Scholar API request failed: %s", exc)
        return {}

    result: dict[str, int] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        external_ids = entry.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv")
        if not arxiv_id:
            continue
        citation_count = entry.get("citationCount")
        if isinstance(citation_count, int):
            result[arxiv_id] = citation_count
    return result


def fetch_popularity_batch(
    paper_ids: list[str],
    *,
    cache_path: Path | None = None,
) -> dict[str, int]:
    """Fetch citation counts for a list of paper IDs with caching.

    1. Load existing cache (if cache_path given)
    2. Find which paper_ids are missing from cache
    3. Batch-fetch missing ones from Semantic Scholar
    4. Merge results and update cache

    Returns dict mapping paper_id → citation count (0 for failures).
    """
    cache: dict[str, Any] = {}
    if cache_path is not None:
        cache = load_popularity_cache(cache_path)

    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    result: dict[str, int] = {}
    missing: list[str] = []

    for pid in paper_ids:
        if pid in cache:
            entry = cache[pid]
            result[pid] = int(entry.get("citation_count", 0)) if isinstance(entry, dict) else 0
        else:
            missing.append(pid)

    if missing:
        # Process in batches of _S2_BATCH_SIZE
        new_entries: dict[str, Any] = {}
        for batch_start in range(0, len(missing), _S2_BATCH_SIZE):
            batch = missing[batch_start : batch_start + _S2_BATCH_SIZE]
            arxiv_ids = _build_arxiv_ids(batch)
            fetched = _fetch_from_s2(arxiv_ids)

            for pid in batch:
                count = fetched.get(pid, 0)
                result[pid] = count
                new_entries[pid] = {"citation_count": count, "fetched_at": now_iso}

            if batch_start + _S2_BATCH_SIZE < len(missing):
                time.sleep(_S2_RATE_LIMIT_DELAY)

        if cache_path is not None:
            save_popularity_cache(cache_path, {**cache, **new_entries})

    return result


# ---------------------------------------------------------------------------
# High-level: enrich papers with popularity weights
# ---------------------------------------------------------------------------


def enrich_papers_with_popularity(
    papers: list[PaperRecord],
    *,
    cache_path: Path | None = None,
    floor: float = _DEFAULT_FLOOR,
) -> dict[str, float]:
    """Return a paper_id → popularity weight mapping for the given papers.

    Weights are in [floor, 1.0]. Papers with unknown popularity get floor weight.
    Returns empty dict for empty input.
    """
    if not papers:
        return {}

    paper_ids = [p.paper_id for p in papers]
    raw_counts = fetch_popularity_batch(paper_ids, cache_path=cache_path)
    normalized = normalize_popularity_scores(raw_counts)
    return {pid: compute_popularity_weight(score, floor=floor) for pid, score in normalized.items()}
