from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

ElementType = Literal["theme", "domain", "method", "frame"]
ELEMENT_TYPES: tuple[ElementType, ...] = ("theme", "domain", "method", "frame")

#: Innovation operations a paper applies to prior work (the "move").
MOVES: tuple[str, ...] = (
    "extend",
    "transfer",
    "combine",
    "benchmark",
    "analyze",
    "simplify",
    "scale",
    "adapt",
)
UNKNOWN_MOVE = "unknown"

#: Short definitions shown to the realisation prompt.
MOVE_DEFINITIONS: Mapping[str, str] = {
    "extend": "push an existing method or problem further along its own axis",
    "transfer": "carry a method from one domain or modality to another",
    "combine": "fuse two previously separate methods or problems",
    "benchmark": "build a dataset, benchmark or evaluation protocol",
    "analyze": "study, explain or measure an existing phenomenon",
    "simplify": "remove components or show a simpler recipe suffices",
    "scale": "scale data, model or compute along an existing recipe",
    "adapt": "specialise a general method to a specific setting",
}

RECORD_OK = "ok"
RECORD_FAILED = "failed"

PairKey = tuple[str, str]


def pair_key(a: str, b: str) -> PairKey:
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True)
class ExtractionRecord:
    """Elements extracted from ONE paper. Contains no cross-paper information,
    so a global cache of these records is safe to consult for any cutoff."""

    paper_id: str
    published_date: str
    status: str
    themes: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    frames: tuple[str, ...] = ()
    template: str = ""
    move: str = UNKNOWN_MOVE
    model: str = ""
    fingerprint: str = ""
    extracted_at: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == RECORD_OK

    def typed_elements(self) -> tuple[tuple[ElementType, str], ...]:
        out: list[tuple[ElementType, str]] = []
        out.extend(("theme", t) for t in self.themes)
        out.extend(("domain", d) for d in self.domains)
        out.extend(("method", m) for m in self.methods)
        out.extend(("frame", f) for f in self.frames)
        return tuple(out)


@dataclass(frozen=True)
class Element:
    """A canonical element within one window's community state."""

    id: str
    type: ElementType
    label: str
    variants: tuple[str, ...]
    paper_ids: tuple[str, ...]
    first_seen: str
    count: int
    heat: float


@dataclass(frozen=True)
class CommunityState:
    """S_t = (C_t, w_t, P_t): elements, their heat, and pair preferences.

    All mappings are built from papers dated on or before ``cutoff_date``."""

    cutoff_date: str
    n_train: int
    n_with_records: int
    elements: Mapping[str, Element]
    pair_count: Mapping[PairKey, int]
    pair_heat: Mapping[PairKey, float]
    pair_recent_rate: Mapping[PairKey, float]
    pair_older_rate: Mapping[PairKey, float]
    move_dist: Mapping[str, float]
    max_heat: float
    hot_threshold: float

    @property
    def coverage(self) -> float:
        return self.n_with_records / self.n_train if self.n_train else 0.0


@dataclass(frozen=True)
class Evidence:
    paper_id: str
    title: str
    month: str
    snippet: str
    matched_elements: tuple[str, ...]


@dataclass(frozen=True)
class Combo:
    elements: tuple[Element, ...]
    move: str
    sampler: str
    score: float
    components: Mapping[str, float] = field(default_factory=dict)
    evidence: tuple[Evidence, ...] = ()

    @property
    def element_ids(self) -> tuple[str, ...]:
        return tuple(e.id for e in self.elements)


@dataclass(frozen=True)
class ParsedIdea:
    combo_index: int
    title: str
    rationale: str
    approach: str
    confidence: float
    key_terms: tuple[str, ...]
