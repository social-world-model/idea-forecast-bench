from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal

Slot = Literal["object", "mechanism", "problem"]
SLOTS: tuple[Slot, ...] = ("object", "mechanism", "problem")

RECORD_OK = "ok"
RECORD_FAILED = "failed"


@dataclass(frozen=True)
class Term:
    """One extracted concept mention: the specific term and its broader parent.
    Both are already normalized surface forms."""

    text: str
    parent: str


@dataclass(frozen=True)
class ConceptRecord:
    """Extraction result for one paper. Immutable; the store appends new
    records rather than editing old ones."""

    paper_id: str
    published_date: str
    status: str
    objects: tuple[Term, ...] = ()
    mechanisms: tuple[Term, ...] = ()
    problems: tuple[Term, ...] = ()
    model: str = ""
    fingerprint: str = ""
    extracted_at: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == RECORD_OK

    def terms(self) -> Iterator[tuple[Slot, Term]]:
        for term in self.objects:
            yield "object", term
        for term in self.mechanisms:
            yield "mechanism", term
        for term in self.problems:
            yield "problem", term


def concept_key(slot: str, text: str) -> str:
    return f"{slot}:{text}"


def split_concept_key(key: str) -> tuple[Slot, str]:
    slot, _, text = key.partition(":")
    for known in SLOTS:
        if slot == known:
            return known, text
    raise ValueError(f"malformed concept key: {key!r}")


@dataclass(frozen=True)
class Concept:
    """A fine cluster of terms at one cutoff: one concept in one slot."""

    id: str
    slot: Slot
    label: str
    parent: str
    variants: tuple[str, ...]
    paper_ids: tuple[str, ...]
    count: int
    doc_frac: float
    first_seen: str
    recent_count: int
    background: bool
    emerging: bool
    slot_share: float


@dataclass(frozen=True)
class Vocabulary:
    """The vocabulary of one topic at one cutoff, built from training papers
    only. ``member_of`` maps every raw ``slot:text`` key seen in training to
    its concept id, so future papers can be assigned by exact text first."""

    topic_id: str
    cutoff_month: str
    cutoff_date: str
    n_train: int
    n_with_records: int
    concepts: Mapping[str, Concept]
    member_of: Mapping[str, str]
    slot_conflicts: tuple[str, ...] = ()
    config_sha: str = ""
    extras: Mapping[str, float] = field(default_factory=dict)

    def combinable(self) -> list[Concept]:
        """Concepts a sampler may draw: not background, and either regular
        (count >= min_count, enforced at build time) or emerging."""
        return [c for c in self.concepts.values() if not c.background]

    def background(self) -> list[Concept]:
        return sorted(
            (c for c in self.concepts.values() if c.background),
            key=lambda c: -c.count,
        )

    def emerging(self) -> list[Concept]:
        return sorted(
            (c for c in self.concepts.values() if c.emerging and not c.background),
            key=lambda c: (-c.recent_count, c.first_seen, c.id),
        )
