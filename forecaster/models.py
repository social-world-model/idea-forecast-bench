"""Domain models for the forecaster package.

All models are frozen (immutable) dataclasses representing the factorized
latent variable model: p(Y|X) ≈ Π_j Σ_z p(y_j|z_j,X) p(z_j|X).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

INNOVATION_SCHEMA_VERSION = 1
ALLOWED_INNOVATION_OPERATORS: tuple[str, ...] = (
    "extend",
    "transfer",
    "compose",
    "benchmark",
    "analyze",
    "simplify",
    "scale",
    "adapt",
)


@dataclass(frozen=True)
class Innovation:
    """Latent innovation variable z = {base_direction, operator, gap}."""

    base_direction: str  # The foundational research direction being built upon
    operator: str  # The methodological operator (e.g. "extend", "transfer", "compose")
    gap: str  # The specific research gap being addressed


@dataclass(frozen=True)
class MemoryEntry:
    """A single entry in the memory inventory with metadata."""

    innovation: Innovation
    source_paper_id: str
    timestamp_month: str  # Format: "YYYY-MM"
    frequency: int = 1
    recency_score: float = 1.0
    utility_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryInventory:
    """Serializable container of memory entries."""

    entries: tuple[MemoryEntry, ...]  # tuple (immutable) not list
    last_updated_month: str
    version: int = 1


@dataclass(frozen=True)
class HindsightSample:
    """One (context, z) training example from hindsight extraction."""

    context_paper_ids: tuple[str, ...]
    cutoff_month: str
    future_paper_id: str
    future_paper_published_date: str
    innovation: Innovation

    @property
    def future_paper_month(self) -> str:
        """Month bucket for the future paper that produced this hindsight label."""
        return self.future_paper_published_date[:7]


@dataclass(frozen=True)
class JointCandidate:
    """Intermediate: innovation + scores before final ranking."""

    innovation: Innovation
    prior_score: float
    evidence_paper_ids: tuple[str, ...]
    proposal_text: str
    realization_score: float
    popularity_bonus: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredProposal:
    """Final output: proposal with joint score from inference."""

    innovation: Innovation
    proposal_text: str
    prior_score: float
    realization_score: float
    joint_score: float
    evidence_paper_ids: tuple[str, ...]
    rank: int = 0
    popularity_bonus: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def innovation_to_dict(innovation: Innovation) -> dict[str, str]:
    """Serialize an Innovation to a plain dict."""
    return {
        "base_direction": innovation.base_direction,
        "operator": innovation.operator,
        "gap": innovation.gap,
    }


def innovation_schema_contract() -> dict[str, Any]:
    """Return the frozen Innovation runtime contract."""
    return {
        "schema_version": INNOVATION_SCHEMA_VERSION,
        "fields": ("base_direction", "operator", "gap"),
        "allowed_operators": list(ALLOWED_INNOVATION_OPERATORS),
    }


def innovation_from_dict(d: dict[str, Any]) -> Innovation:
    """Deserialize an Innovation from a plain dict."""
    return Innovation(
        base_direction=str(d["base_direction"]),
        operator=str(d["operator"]),
        gap=str(d["gap"]),
    )


def memory_entry_to_dict(entry: MemoryEntry) -> dict[str, Any]:
    """Serialize a MemoryEntry to a plain dict."""
    return {
        "innovation": innovation_to_dict(entry.innovation),
        "source_paper_id": entry.source_paper_id,
        "timestamp_month": entry.timestamp_month,
        "frequency": entry.frequency,
        "recency_score": entry.recency_score,
        "utility_score": entry.utility_score,
        "metadata": dict(entry.metadata),
    }


def memory_entry_from_dict(d: dict[str, Any]) -> MemoryEntry:
    """Deserialize a MemoryEntry from a plain dict."""
    return MemoryEntry(
        innovation=innovation_from_dict(d["innovation"]),
        source_paper_id=str(d["source_paper_id"]),
        timestamp_month=str(d["timestamp_month"]),
        frequency=int(d.get("frequency", 1)),
        recency_score=float(d.get("recency_score", 1.0)),
        utility_score=float(d.get("utility_score", 0.0)),
        metadata=dict(d.get("metadata", {})),
    )


def memory_inventory_to_dict(inventory: MemoryInventory) -> dict[str, Any]:
    """Serialize a MemoryInventory to a plain dict."""
    return {
        "entries": [memory_entry_to_dict(e) for e in inventory.entries],
        "last_updated_month": inventory.last_updated_month,
        "version": inventory.version,
    }


def memory_inventory_from_dict(d: dict[str, Any]) -> MemoryInventory:
    """Deserialize a MemoryInventory from a plain dict."""
    entries = tuple(memory_entry_from_dict(e) for e in d.get("entries", []))
    return MemoryInventory(
        entries=entries,
        last_updated_month=str(d["last_updated_month"]),
        version=int(d.get("version", 1)),
    )
