"""Closed operator inventory + free-text → closed mapping.

The hindsight extractor (forecaster/hindsight/extractor.py) emits one of
ALLOWED_INNOVATION_OPERATORS (8 verbs). For the Foresight plan we collapse
this to a closed 4-set plus an `other` bucket so the rubric/reward gates
can branch on a small enum.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


CLOSED_OPERATORS: tuple[str, ...] = (
    "limitation_extension",
    "cross_domain_transfer",
    "benchmark_proposal",
    "method_composition",
)
UNMAPPABLE_BUCKET: str = "other"


@dataclass(frozen=True)
class OperatorSpec:
    id: str
    one_line: str
    example: str


@dataclass(frozen=True)
class OperatorInventory:
    operators: tuple[OperatorSpec, ...]
    free_text_mapping: dict[str, str]
    unmappable_bucket: str

    @property
    def closed_ids(self) -> tuple[str, ...]:
        return tuple(op.id for op in self.operators)

    def map_one(self, free_text: str) -> str:
        return map_free_text_operator(free_text, self)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "operators.yaml"


def load_operator_inventory(path: str | Path | None = None) -> OperatorInventory:
    """Load and validate operators.yaml."""
    p = Path(path) if path is not None else _default_config_path()
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"operators.yaml must decode to a mapping, got {type(payload)}")

    raw_ops = payload.get("operators") or []
    if not isinstance(raw_ops, list) or not raw_ops:
        raise ValueError("operators.yaml: 'operators' must be a non-empty list")
    ops = tuple(
        OperatorSpec(
            id=str(o["id"]),
            one_line=str(o.get("one_line", "")).strip(),
            example=str(o.get("example", "")).strip(),
        )
        for o in raw_ops
    )

    closed_ids = {op.id for op in ops}
    if closed_ids != set(CLOSED_OPERATORS):
        raise ValueError(
            f"operators.yaml ids {sorted(closed_ids)} != expected closed inventory "
            f"{sorted(CLOSED_OPERATORS)}"
        )

    raw_map = payload.get("free_text_mapping") or {}
    if not isinstance(raw_map, dict):
        raise ValueError("operators.yaml: 'free_text_mapping' must be a mapping")
    free_text_mapping = {str(k).strip().lower(): str(v).strip() for k, v in raw_map.items()}
    unmappable = str(payload.get("unmappable_bucket", UNMAPPABLE_BUCKET)).strip()

    allowed_targets = closed_ids | {unmappable}
    for src, dst in free_text_mapping.items():
        if dst not in allowed_targets:
            raise ValueError(
                f"operators.yaml: mapping {src!r} -> {dst!r} points outside "
                f"closed inventory + {{{unmappable}}}"
            )

    return OperatorInventory(
        operators=ops,
        free_text_mapping=free_text_mapping,
        unmappable_bucket=unmappable,
    )


def map_free_text_operator(
    free_text: str,
    inventory: OperatorInventory,
) -> str:
    """Map a free-text operator to a closed id or the unmappable bucket.

    Comparison is case-insensitive and trims whitespace. If the input is
    already a closed id, it is returned unchanged.
    """
    if not isinstance(free_text, str):
        return inventory.unmappable_bucket
    key = free_text.strip().lower()
    if not key:
        return inventory.unmappable_bucket
    if key in {op.id for op in inventory.operators}:
        return key
    return inventory.free_text_mapping.get(key, inventory.unmappable_bucket)


def operator_distribution(
    free_text_operators: Iterable[str],
    inventory: OperatorInventory,
) -> dict[str, int]:
    """Return per-closed-id counts (including the unmappable bucket)."""
    counts: dict[str, int] = {op.id: 0 for op in inventory.operators}
    counts[inventory.unmappable_bucket] = 0
    for raw in free_text_operators:
        counts[map_free_text_operator(raw, inventory)] = (
            counts.get(map_free_text_operator(raw, inventory), 0) + 1
        )
    return counts
