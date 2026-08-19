"""Tests for MemoryStore (TDD: RED phase)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from forecaster.models import HindsightSample, Innovation, MemoryEntry, MemoryInventory
from forecaster.prior.memory import (
    MemoryStore,
    build_memory_store_from_hindsight_samples,
    hindsight_sample_available_by_cutoff,
)

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


def test_query_utility_can_change_ranking():
    """Delayed utility should influence later retrieval order."""
    high_utility = MemoryEntry(
        innovation=_make_innovation(base="high utility", gap="a"),
        source_paper_id="high-utility",
        timestamp_month="2024-01",
        frequency=1,
        recency_score=0.4,
        utility_score=5.0,
    )
    high_recency = MemoryEntry(
        innovation=_make_innovation(base="high recency", gap="b"),
        source_paper_id="high-recency",
        timestamp_month="2024-01",
        frequency=1,
        recency_score=0.6,
        utility_score=0.0,
    )
    store = MemoryStore(
        MemoryInventory(entries=(high_utility, high_recency), last_updated_month="2024-01")
    )

    results = store.query(2)

    assert results[0].source_paper_id == "high-utility"


def test_decay_recency_reduces_scores():
    """After decay, recency_score should be <= original score."""
    store = _make_store_with_entries(3, "2024-01")
    original_scores = [e.recency_score for e in store.inventory.entries]
    decayed = store.decay_recency("2024-06")
    new_scores = [e.recency_score for e in decayed.inventory.entries]
    assert all(new <= orig for new, orig in zip(new_scores, original_scores, strict=False))
    assert any(new < orig for new, orig in zip(new_scores, original_scores, strict=False))


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
    for orig, loaded_entry in zip(store.inventory.entries, loaded.inventory.entries, strict=False):
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
    assert "frequency" in result
    assert "recency" in result
    assert "utility" in result


def test_hindsight_sample_available_by_cutoff_respects_exact_published_date():
    """A paper published later in the same month is not yet available at month-start cutoff."""
    sample = HindsightSample(
        context_paper_ids=("ctx",),
        cutoff_month="2024-01",
        future_paper_id="future-1",
        future_paper_published_date="2024-02-15",
        innovation=_make_innovation(),
    )
    assert not hindsight_sample_available_by_cutoff(sample, "2024-02")


def test_build_memory_store_from_hindsight_samples_filters_future_sources():
    """The cutoff snapshot should only include source papers already published by the cutoff."""
    visible = HindsightSample(
        context_paper_ids=("ctx-visible",),
        cutoff_month="2024-01",
        future_paper_id="visible-paper",
        future_paper_published_date="2024-02-01",
        innovation=_make_innovation(base="visible direction"),
    )
    hidden = HindsightSample(
        context_paper_ids=("ctx-hidden",),
        cutoff_month="2024-01",
        future_paper_id="hidden-paper",
        future_paper_published_date="2024-03-15",
        innovation=_make_innovation(base="hidden direction"),
    )

    store = build_memory_store_from_hindsight_samples([visible, hidden], "2024-02")

    assert [entry.source_paper_id for entry in store.inventory.entries] == ["visible-paper"]


# ---------------------------------------------------------------------------
# Phase 5: Chronology guard tests
# ---------------------------------------------------------------------------

def test_load_with_cutoff_warns_when_memory_newer(tmp_path, caplog):  # type: ignore[no-untyped-def]
    """MemoryStore.load with cutoff_month warns when memory is newer than cutoff."""
    import logging

    store = MemoryStore.empty("2025-06")  # future month
    path = tmp_path / "mem.json"
    store.persist(path)

    with caplog.at_level(logging.WARNING, logger="forecaster.prior.memory"):
        MemoryStore.load(path, cutoff_month="2024-01")

    assert any("newer than" in r.message for r in caplog.records)


def test_load_with_cutoff_no_warning_when_memory_is_current(tmp_path, caplog):  # type: ignore[no-untyped-def]
    """MemoryStore.load does not warn when memory is not newer than cutoff."""
    import logging

    store = MemoryStore.empty("2024-01")
    path = tmp_path / "mem.json"
    store.persist(path)

    with caplog.at_level(logging.WARNING, logger="forecaster.prior.memory"):
        MemoryStore.load(path, cutoff_month="2024-06")

    assert not any("newer than" in r.message for r in caplog.records)


def test_load_without_cutoff_never_warns(tmp_path, caplog):  # type: ignore[no-untyped-def]
    """MemoryStore.load without cutoff_month does not check chronology."""
    import logging

    store = MemoryStore.empty("2099-12")  # far-future month
    path = tmp_path / "mem.json"
    store.persist(path)

    with caplog.at_level(logging.WARNING, logger="forecaster.prior.memory"):
        MemoryStore.load(path)  # no cutoff_month

    assert not any("newer than" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Phase 4: Delayed utility update integration tests
# ---------------------------------------------------------------------------

def test_apply_delayed_utility_update_increases_utility_on_match():
    """_apply_delayed_utility_update raises utility for actual future support."""
    from forecaster.models import Innovation, ScoredProposal
    from forecaster.orchestrator import _apply_delayed_utility_update

    innovation = Innovation(base_direction="neural network", operator="extend", gap="efficiency")
    store = MemoryStore.empty("2024-01")
    store = store.append(innovation, source_paper_id="src-paper-1", month="2024-01")

    proposal = ScoredProposal(
        innovation=innovation,
        proposal_text="Title\nBody",
        prior_score=0.5,
        realization_score=0.5,
        joint_score=0.5,
        evidence_paper_ids=("historical-evidence",),
        rank=1,
    )

    updated = _apply_delayed_utility_update(
        store,
        [proposal],
        [
            {
                "proposal_rank": 1,
                "matched_future_paper_ids": ["future-paper-1"],
                "future_support_confirmed": True,
                "future_match_score": 0.91,
            }
        ],
    )

    original_entry = store.inventory.entries[0]
    updated_entry = updated.inventory.entries[0]
    # utility_delta=1.0 matched → new_utility = 0.3 * 1.0 + 0.7 * 0.0 = 0.3
    assert updated_entry.utility_score > original_entry.utility_score


def test_apply_delayed_utility_update_decreases_utility_on_no_match():
    """_apply_delayed_utility_update applies small negative delta for non-matching proposals."""
    from forecaster.models import Innovation, ScoredProposal
    from forecaster.orchestrator import _apply_delayed_utility_update

    innovation = Innovation(base_direction="neural network", operator="extend", gap="efficiency")
    store = MemoryStore.empty("2024-01")
    store = store.append(innovation, source_paper_id="src-paper-1", month="2024-01")
    # Set a positive utility first
    store = store.update_utility("src-paper-1", 1.0)

    proposal = ScoredProposal(
        innovation=innovation,
        proposal_text="Title\nBody",
        prior_score=0.5,
        realization_score=0.5,
        joint_score=0.5,
        evidence_paper_ids=("unrelated-paper",),  # no overlap with future set
        rank=1,
    )

    updated = _apply_delayed_utility_update(
        store,
        [proposal],
        [
            {
                "proposal_rank": 1,
                "matched_future_paper_ids": [],
                "future_support_confirmed": False,
            }
        ],
    )

    original_entry = store.inventory.entries[0]
    updated_entry = updated.inventory.entries[0]
    # utility_delta=-0.1 no match → utility decreases slightly
    assert updated_entry.utility_score < original_entry.utility_score


def test_apply_delayed_utility_update_preserves_immutability():
    """_apply_delayed_utility_update returns new store, does not mutate original."""
    from forecaster.models import Innovation, ScoredProposal
    from forecaster.orchestrator import _apply_delayed_utility_update

    innovation = Innovation(base_direction="neural", operator="extend", gap="test")
    store = MemoryStore.empty("2024-01")
    store = store.append(innovation, source_paper_id="src-1", month="2024-01")

    proposal = ScoredProposal(
        innovation=innovation,
        proposal_text="T\nB",
        prior_score=0.5,
        realization_score=0.5,
        joint_score=0.5,
        evidence_paper_ids=("future-1",),
        rank=1,
    )

    original_utility = store.inventory.entries[0].utility_score
    updated = _apply_delayed_utility_update(
        store,
        [proposal],
        [{"proposal_rank": 1, "matched_future_paper_ids": ["future-1"]}],
    )

    # Original is unchanged
    assert store.inventory.entries[0].utility_score == original_utility
    # Updated is different
    assert updated.inventory.entries[0].utility_score != original_utility


def test_apply_delayed_utility_update_records_provenance():
    """Delayed feedback should persist proposal/evidence provenance in memory metadata."""
    from forecaster.models import ScoredProposal
    from forecaster.orchestrator import _apply_delayed_utility_update

    innovation = Innovation(base_direction="retrieval agent", operator="compose", gap="ground long-horizon planning")
    store = MemoryStore.empty("2024-01").append(innovation, source_paper_id="src-1", month="2024-01")
    proposal = ScoredProposal(
        innovation=innovation,
        proposal_text="Grounded Retrieval Agent\nUse retrieval evidence to plan better.",
        prior_score=-0.4,
        realization_score=-0.2,
        joint_score=-0.28,
        evidence_paper_ids=("future-1", "ctx-1"),
        rank=1,
    )

    updated = _apply_delayed_utility_update(
        store,
        [proposal],
        [{"proposal_rank": 1, "matched_future_paper_ids": ["future-1"]}],
        cutoff_month="2024-02",
    )

    metadata = updated.inventory.entries[0].metadata
    assert metadata["last_delayed_feedback"]["cutoff_month"] == "2024-02"
    assert metadata["last_delayed_feedback"]["matched_future_paper_ids"] == ["future-1"]
    assert metadata["delayed_feedback_history"][-1]["proposal_rank"] == 1
    assert metadata["last_delayed_feedback"]["proposal_text"] == proposal.proposal_text
