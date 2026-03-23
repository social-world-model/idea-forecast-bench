"""Tests for MemoryStore (TDD: RED phase)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from forecaster.models import Innovation, MemoryEntry, MemoryInventory
from forecaster.prior.memory import MemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_innovation(base: str = "diffusion models", op: str = "extend", gap: str = "No gap.") -> Innovation:
    return Innovation(base_direction=base, operator=op, gap=gap)


def _make_store_with_entries(n: int, current_month: str = "2024-01") -> MemoryStore:
    store = MemoryStore.empty(current_month)
    for i in range(n):
        inn = _make_innovation(base=f"direction {i}", gap=f"gap {i}")
        store = store.append(inn, source_paper_id=f"paper-{i}", month=current_month)
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_creates_empty_store():
    store = MemoryStore.empty("2024-01")
    assert store.size == 0


def test_append_adds_entry():
    store = MemoryStore.empty("2024-01")
    inn = _make_innovation()
    new_store = store.append(inn, source_paper_id="paper-1", month="2024-01")
    assert new_store.size == 1


def test_append_returns_new_instance():
    """Original store must be unchanged (immutability)."""
    store = MemoryStore.empty("2024-01")
    inn = _make_innovation()
    new_store = store.append(inn, source_paper_id="paper-1", month="2024-01")
    assert store.size == 0
    assert new_store is not store


def test_append_increments_frequency_for_duplicate():
    """Same innovation (same base_direction+operator+gap) increments frequency."""
    store = MemoryStore.empty("2024-01")
    inn = _make_innovation()
    store = store.append(inn, source_paper_id="paper-1", month="2024-01")
    store = store.append(inn, source_paper_id="paper-2", month="2024-02")
    # Should still have 1 entry (deduplicated), but frequency=2
    assert store.size == 1
    assert store.inventory.entries[0].frequency == 2


def test_query_returns_top_n():
    store = _make_store_with_entries(10, "2024-01")
    results = store.query(5)
    assert len(results) == 5


def test_query_returns_all_when_fewer_than_n():
    store = _make_store_with_entries(3, "2024-01")
    results = store.query(10)
    assert len(results) == 3


def test_query_ranking_higher_recency_ranked_higher():
    """Entry with higher recency_score should appear first."""
    store = MemoryStore.empty("2024-01")
    inn_low = _make_innovation(base="low recency", gap="low")
    inn_high = _make_innovation(base="high recency", gap="high")
    # Add low-recency entry with older timestamp
    store = store.append(inn_low, source_paper_id="old", month="2023-01")
    # Manually create a store where we control recency scores via inventory
    from forecaster.models import MemoryEntry, MemoryInventory
    entry_low = MemoryEntry(innovation=inn_low, source_paper_id="old", timestamp_month="2023-01", recency_score=0.1)
    entry_high = MemoryEntry(innovation=inn_high, source_paper_id="new", timestamp_month="2024-01", recency_score=0.9)
    inv = MemoryInventory(entries=(entry_low, entry_high), last_updated_month="2024-01")
    store = MemoryStore(inv)
    results = store.query(2, recency_weight=1.0)
    assert results[0].innovation.base_direction == "high recency"


def test_decay_recency_reduces_scores():
    """After decay, recency_score should be <= original score."""
    store = _make_store_with_entries(3, "2024-01")
    original_scores = [e.recency_score for e in store.inventory.entries]
    decayed = store.decay_recency("2024-06")
    new_scores = [e.recency_score for e in decayed.inventory.entries]
    assert all(new <= orig for new, orig in zip(new_scores, original_scores))
    assert any(new < orig for new, orig in zip(new_scores, original_scores))


def test_decay_recency_returns_new_store():
    """decay_recency must not mutate the original store."""
    store = _make_store_with_entries(3, "2024-01")
    original_scores = [e.recency_score for e in store.inventory.entries]
    decayed = store.decay_recency("2024-06")
    assert decayed is not store
    # Original unchanged
    assert [e.recency_score for e in store.inventory.entries] == original_scores


def test_update_utility_applies_ema():
    """update_utility should apply EMA: new = alpha*delta + (1-alpha)*old."""
    store = MemoryStore.empty("2024-01")
    inn = _make_innovation()
    store = store.append(inn, source_paper_id="paper-1", month="2024-01")
    # Initial utility_score is 0.0
    updated = store.update_utility("paper-1", utility_delta=1.0, ema_alpha=0.3)
    entry = updated.inventory.entries[0]
    expected = 0.3 * 1.0 + 0.7 * 0.0
    assert abs(entry.utility_score - expected) < 1e-9


def test_update_utility_returns_new_store():
    """update_utility must not mutate the original store."""
    store = MemoryStore.empty("2024-01")
    inn = _make_innovation()
    store = store.append(inn, source_paper_id="paper-1", month="2024-01")
    original_utility = store.inventory.entries[0].utility_score
    updated = store.update_utility("paper-1", utility_delta=1.0)
    assert updated is not store
    assert store.inventory.entries[0].utility_score == original_utility


def test_persist_and_load_roundtrip():
    """Save to a temp file and load back; result should be equal."""
    store = _make_store_with_entries(5, "2024-03")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory.json"
        store.persist(path)
        loaded = MemoryStore.load(path)
    assert loaded.size == store.size
    assert loaded.inventory.last_updated_month == store.inventory.last_updated_month
    for orig, loaded_entry in zip(store.inventory.entries, loaded.inventory.entries):
        assert orig.innovation == loaded_entry.innovation
        assert orig.frequency == loaded_entry.frequency


def test_load_nonexistent_file_returns_empty():
    """Loading from a non-existent file should return an empty store."""
    store = MemoryStore.load("/tmp/totally_nonexistent_file_abc123.json")
    assert store.size == 0


def test_format_for_prompt_returns_string():
    """format_for_prompt should return non-empty string containing entry info."""
    store = _make_store_with_entries(3, "2024-01")
    result = store.format_for_prompt(top_n=3)
    assert isinstance(result, str)
    assert len(result) > 0
    # Should contain numbered entries
    assert "1." in result
    assert "base_direction" in result
