"""Tests for forecaster domain models."""

from __future__ import annotations

import pytest

from forecaster.models import (
    ALLOWED_INNOVATION_OPERATORS,
    HindsightSample,
    Innovation,
    JointCandidate,
    MemoryEntry,
    MemoryInventory,
    ScoredProposal,
    innovation_from_dict,
    innovation_schema_contract,
    innovation_to_dict,
    memory_entry_from_dict,
    memory_entry_to_dict,
    memory_inventory_from_dict,
    memory_inventory_to_dict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_innovation() -> Innovation:
    return Innovation(
        base_direction="transformer-based NLP",
        operator="extend",
        gap="long-context reasoning",
    )


@pytest.fixture
def sample_entry(sample_innovation: Innovation) -> MemoryEntry:
    return MemoryEntry(
        innovation=sample_innovation,
        source_paper_id="arxiv:2401.00001",
        timestamp_month="2024-01",
    )


@pytest.fixture
def sample_inventory(sample_entry: MemoryEntry) -> MemoryInventory:
    return MemoryInventory(
        entries=(sample_entry,),
        last_updated_month="2024-01",
    )


# ---------------------------------------------------------------------------
# Immutability (frozen dataclass) tests
# ---------------------------------------------------------------------------


class TestInnovationImmutability:
    def test_frozen(self, sample_innovation: Innovation) -> None:
        with pytest.raises((AttributeError, TypeError)):
            sample_innovation.operator = "transfer"  # type: ignore[misc]

    def test_fields(self, sample_innovation: Innovation) -> None:
        assert sample_innovation.base_direction == "transformer-based NLP"
        assert sample_innovation.operator == "extend"
        assert sample_innovation.gap == "long-context reasoning"


class TestMemoryEntryImmutability:
    def test_frozen(self, sample_entry: MemoryEntry) -> None:
        with pytest.raises((AttributeError, TypeError)):
            sample_entry.frequency = 5  # type: ignore[misc]

    def test_defaults(self, sample_entry: MemoryEntry) -> None:
        assert sample_entry.frequency == 1
        assert sample_entry.recency_score == 1.0
        assert sample_entry.utility_score == 0.0
        assert sample_entry.metadata == {}

    def test_metadata_is_dict(self, sample_entry: MemoryEntry) -> None:
        assert isinstance(sample_entry.metadata, dict)


class TestMemoryInventoryImmutability:
    def test_frozen(self, sample_inventory: MemoryInventory) -> None:
        with pytest.raises((AttributeError, TypeError)):
            sample_inventory.version = 99  # type: ignore[misc]

    def test_entries_is_tuple(self, sample_inventory: MemoryInventory) -> None:
        assert isinstance(sample_inventory.entries, tuple)

    def test_default_version(self, sample_inventory: MemoryInventory) -> None:
        assert sample_inventory.version == 1


class TestHindsightSampleImmutability:
    def test_frozen(self, sample_innovation: Innovation) -> None:
        sample = HindsightSample(
            context_paper_ids=("arxiv:2401.00001",),
            cutoff_month="2024-01",
            future_paper_id="arxiv:2402.00001",
            future_paper_published_date="2024-02-01",
            innovation=sample_innovation,
        )
        with pytest.raises((AttributeError, TypeError)):
            sample.cutoff_month = "2024-02"  # type: ignore[misc]

    def test_context_ids_is_tuple(self, sample_innovation: Innovation) -> None:
        sample = HindsightSample(
            context_paper_ids=("a", "b"),
            cutoff_month="2024-01",
            future_paper_id="arxiv:2402.00001",
            future_paper_published_date="2024-02-01",
            innovation=sample_innovation,
        )
        assert isinstance(sample.context_paper_ids, tuple)

    def test_future_paper_month(self, sample_innovation: Innovation) -> None:
        sample = HindsightSample(
            context_paper_ids=("a",),
            cutoff_month="2024-01",
            future_paper_id="arxiv:2402.00001",
            future_paper_published_date="2024-02-15",
            innovation=sample_innovation,
        )
        assert sample.future_paper_month == "2024-02"


class TestJointCandidateImmutability:
    def test_frozen(self, sample_innovation: Innovation) -> None:
        candidate = JointCandidate(
            innovation=sample_innovation,
            prior_score=0.7,
            evidence_paper_ids=("arxiv:2401.00001",),
            proposal_text="Test proposal",
            realization_score=0.8,
        )
        with pytest.raises((AttributeError, TypeError)):
            candidate.prior_score = 0.5  # type: ignore[misc]

    def test_evidence_ids_is_tuple(self, sample_innovation: Innovation) -> None:
        candidate = JointCandidate(
            innovation=sample_innovation,
            prior_score=0.7,
            evidence_paper_ids=("x",),
            proposal_text="Test",
            realization_score=0.6,
        )
        assert isinstance(candidate.evidence_paper_ids, tuple)


class TestScoredProposalImmutability:
    def test_frozen(self, sample_innovation: Innovation) -> None:
        proposal = ScoredProposal(
            innovation=sample_innovation,
            proposal_text="Test",
            prior_score=0.4,
            realization_score=0.6,
            joint_score=0.52,
            evidence_paper_ids=("arxiv:2401.00001",),
        )
        with pytest.raises((AttributeError, TypeError)):
            proposal.joint_score = 0.99  # type: ignore[misc]

    def test_defaults(self, sample_innovation: Innovation) -> None:
        proposal = ScoredProposal(
            innovation=sample_innovation,
            proposal_text="Test",
            prior_score=0.4,
            realization_score=0.6,
            joint_score=0.52,
            evidence_paper_ids=(),
        )
        assert proposal.rank == 0
        assert proposal.metadata == {}


# ---------------------------------------------------------------------------
# Serialization round-trip tests
# ---------------------------------------------------------------------------


class TestInnovationSerialization:
    def test_schema_contract(self) -> None:
        contract = innovation_schema_contract()
        assert contract["schema_version"] == 1
        assert tuple(contract["allowed_operators"]) == ALLOWED_INNOVATION_OPERATORS

    def test_to_dict(self, sample_innovation: Innovation) -> None:
        d = innovation_to_dict(sample_innovation)
        assert d["base_direction"] == "transformer-based NLP"
        assert d["operator"] == "extend"
        assert d["gap"] == "long-context reasoning"

    def test_from_dict(self) -> None:
        d = {
            "base_direction": "vision models",
            "operator": "compose",
            "gap": "cross-modal alignment",
        }
        innovation = innovation_from_dict(d)
        assert isinstance(innovation, Innovation)
        assert innovation.base_direction == "vision models"
        assert innovation.operator == "compose"
        assert innovation.gap == "cross-modal alignment"

    def test_round_trip(self, sample_innovation: Innovation) -> None:
        assert (
            innovation_from_dict(innovation_to_dict(sample_innovation))
            == sample_innovation
        )


class TestMemoryEntrySerialization:
    def test_to_dict(self, sample_entry: MemoryEntry) -> None:
        d = memory_entry_to_dict(sample_entry)
        assert d["source_paper_id"] == "arxiv:2401.00001"
        assert d["timestamp_month"] == "2024-01"
        assert isinstance(d["innovation"], dict)

    def test_round_trip(self, sample_entry: MemoryEntry) -> None:
        assert (
            memory_entry_from_dict(memory_entry_to_dict(sample_entry)) == sample_entry
        )

    def test_round_trip_with_metadata(self, sample_innovation: Innovation) -> None:
        entry = MemoryEntry(
            innovation=sample_innovation,
            source_paper_id="arxiv:2401.99999",
            timestamp_month="2024-06",
            frequency=3,
            recency_score=0.8,
            utility_score=0.5,
            metadata={"domain": "NLP"},
        )
        assert memory_entry_from_dict(memory_entry_to_dict(entry)) == entry


class TestMemoryInventorySerialization:
    def test_to_dict(self, sample_inventory: MemoryInventory) -> None:
        d = memory_inventory_to_dict(sample_inventory)
        assert d["last_updated_month"] == "2024-01"
        assert isinstance(d["entries"], list)
        assert len(d["entries"]) == 1

    def test_round_trip(self, sample_inventory: MemoryInventory) -> None:
        assert (
            memory_inventory_from_dict(memory_inventory_to_dict(sample_inventory))
            == sample_inventory
        )

    def test_round_trip_empty(self) -> None:
        inventory = MemoryInventory(entries=(), last_updated_month="2024-03", version=2)
        assert (
            memory_inventory_from_dict(memory_inventory_to_dict(inventory)) == inventory
        )
