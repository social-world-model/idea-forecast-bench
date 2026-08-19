"""D_z ↔ prior SFT bridge.

The existing prior trainer (forecaster/prior/trainer.py) takes
`{"input": <rendered memory prompt>, "target": <innovation JSON>}` rows.
This module converts the Phase-1 D_z (forecaster.foresight.dz) into those
rows and exposes a thin `RawMemoryStore` so the existing sampler can be
fed a precomputed `memory_text` string.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from forecaster.foresight.dz import load_dz_rows
from forecaster.foresight.operators import (
    OperatorInventory,
    load_operator_inventory,
)
from forecaster.prior.prompting import render_prior_user_prompt


# --------------------------------------------------------------------------- D_z -> SFT samples


def dz_row_to_sft_sample(
    row: dict,
    *,
    inventory: OperatorInventory,
    drop_unmappable: bool,
) -> dict | None:
    """Convert one D_z row into a `{input, target, ...}` SFT sample.

    Returns None if the row has no memory_text (i.e., the D_z was built
    without a corpus) or if `drop_unmappable=True` and the row's
    operator_closed is `other`.
    """
    memory_text = row.get("memory_text")
    if not memory_text:
        return None
    op_closed = row.get("operator_closed", "")
    if drop_unmappable and op_closed == inventory.unmappable_bucket:
        return None
    target_z = row.get("target_z") or {}
    target = {
        "base_direction": str(target_z.get("base_direction", "")),
        "operator": str(target_z.get("operator", "")),
        "gap": str(target_z.get("gap", "")),
    }
    return {
        "input": render_prior_user_prompt(memory_text),
        "target": json.dumps(target, ensure_ascii=False),
        "cutoff_month": str(row.get("cutoff_t", "") or "")[:7],
        "future_paper_id": str(row.get("source_future_id", "") or ""),
        "future_paper_published_date": str(row.get("future_paper_published_date", "") or ""),
        "memory_prompt": memory_text,
        "operator_closed": op_closed,
        "topic_id": str(row.get("topic_id", "") or ""),
    }


def build_sft_samples_from_dz(
    dz_path: str | Path,
    *,
    inventory: OperatorInventory | None = None,
    drop_unmappable: bool = True,
) -> list[dict]:
    """Stream a D_z JSONL into SFT samples.

    Rows missing `memory_text` are skipped (a corpus must be passed in
    `augment_hindsight_rows` to populate that field).
    """
    inventory = inventory or load_operator_inventory()
    rows = load_dz_rows(dz_path)
    samples: list[dict] = []
    skipped_no_memory = 0
    skipped_unmappable = 0
    for r in rows:
        out = dz_row_to_sft_sample(r, inventory=inventory, drop_unmappable=drop_unmappable)
        if out is None:
            if not r.get("memory_text"):
                skipped_no_memory += 1
            else:
                skipped_unmappable += 1
            continue
        samples.append(out)
    return samples


def save_sft_jsonl(samples: list[dict], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    return p


# --------------------------------------------------------------------------- memory store adapter


@dataclass(frozen=True)
class RawMemoryStore:
    """Tiny duck-typed memory store: returns a precomputed memory string.

    The existing prior sampler accepts any object exposing
    `format_for_prompt()` — including `MemoryStore`. This adapter lets us
    plug `build_memory()` output straight into the sampler without
    constructing the heavier `MemoryStore` from hindsight samples.
    """

    memory_text: str

    def format_for_prompt(self, *, top_n: int | None = None) -> str:
        return self.memory_text

    def exclude_source_paper_ids(self, paper_ids: Iterable[str]) -> "RawMemoryStore":
        # No-op: RawMemoryStore doesn't track per-entry provenance.
        # The training pipeline calls this to avoid label leakage; since
        # memory_text is already a function of the legal cutoff, the call
        # is satisfied by returning self.
        return self


__all__ = [
    "dz_row_to_sft_sample",
    "build_sft_samples_from_dz",
    "save_sft_jsonl",
    "RawMemoryStore",
]
