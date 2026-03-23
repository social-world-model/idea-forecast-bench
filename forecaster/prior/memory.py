"""Memory store for the innovation prior.

All mutation operations return new MemoryStore instances (immutable pattern).
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from forecaster.models import (
    Innovation,
    MemoryEntry,
    MemoryInventory,
    memory_entry_to_dict,
    memory_entry_from_dict,
    memory_inventory_to_dict,
    memory_inventory_from_dict,
    innovation_to_dict,
)

_RECENCY_DECAY_PER_MONTH: float = 0.9


def _months_between(earlier: str, later: str) -> int:
    """Months between two 'YYYY-MM' strings."""
    y1, m1 = map(int, earlier.split("-"))
    y2, m2 = map(int, later.split("-"))
    return (y2 - y1) * 12 + (m2 - m1)


def _innovation_key(innovation: Innovation) -> tuple[str, str, str]:
    return (innovation.base_direction, innovation.operator, innovation.gap)


class MemoryStore:
    """Immutable wrapper around MemoryInventory with query and update operations.

    All mutation methods return new MemoryStore instances.
    """

    def __init__(self, inventory: MemoryInventory) -> None:
        self._inventory = inventory

    @property
    def inventory(self) -> MemoryInventory:
        return self._inventory

    @property
    def size(self) -> int:
        return len(self._inventory.entries)

    @classmethod
    def empty(cls, current_month: str) -> MemoryStore:
        """Create an empty MemoryStore."""
        inventory = MemoryInventory(entries=(), last_updated_month=current_month)
        return cls(inventory)

    @classmethod
    def load(cls, path: str | Path) -> MemoryStore:
        """Load from JSON file. Returns empty store if file doesn't exist."""
        path = Path(path)
        if not path.exists():
            return cls.empty("1970-01")
        raw = json.loads(path.read_text(encoding="utf-8"))
        inventory = memory_inventory_from_dict(raw)
        return cls(inventory)

    def persist(self, path: str | Path) -> None:
        """Save to JSON file atomically (write to tmp, rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(memory_inventory_to_dict(self._inventory), indent=2, ensure_ascii=False)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def append(
        self,
        innovation: Innovation,
        source_paper_id: str,
        month: str,
    ) -> MemoryStore:
        """Add a new entry or increment frequency for duplicate. Returns new MemoryStore."""
        key = _innovation_key(innovation)
        new_entries: list[MemoryEntry] = []
        found = False
        for entry in self._inventory.entries:
            if _innovation_key(entry.innovation) == key:
                updated = MemoryEntry(
                    innovation=entry.innovation,
                    source_paper_id=entry.source_paper_id,
                    timestamp_month=entry.timestamp_month,
                    frequency=entry.frequency + 1,
                    recency_score=entry.recency_score,
                    utility_score=entry.utility_score,
                    metadata=entry.metadata,
                )
                new_entries.append(updated)
                found = True
            else:
                new_entries.append(entry)
        if not found:
            new_entries.append(
                MemoryEntry(
                    innovation=innovation,
                    source_paper_id=source_paper_id,
                    timestamp_month=month,
                )
            )
        new_inventory = MemoryInventory(
            entries=tuple(new_entries),
            last_updated_month=month,
            version=self._inventory.version,
        )
        return MemoryStore(new_inventory)

    def query(
        self,
        n: int,
        *,
        recency_weight: float = 0.5,
    ) -> list[MemoryEntry]:
        """Return top-n entries ranked by weighted score.

        Score = recency_weight * recency_score + (1 - recency_weight) * normalized_frequency
        """
        entries = list(self._inventory.entries)
        if not entries:
            return []
        max_freq = max(e.frequency for e in entries) or 1
        def score(entry: MemoryEntry) -> float:
            norm_freq = entry.frequency / max_freq
            return recency_weight * entry.recency_score + (1.0 - recency_weight) * norm_freq
        ranked = sorted(entries, key=score, reverse=True)
        return ranked[:n]

    def decay_recency(self, current_month: str) -> MemoryStore:
        """Apply exponential recency decay: recency_score *= decay^months_elapsed."""
        new_entries: list[MemoryEntry] = []
        for entry in self._inventory.entries:
            months = max(0, _months_between(entry.timestamp_month, current_month))
            new_score = entry.recency_score * (_RECENCY_DECAY_PER_MONTH ** months)
            updated = MemoryEntry(
                innovation=entry.innovation,
                source_paper_id=entry.source_paper_id,
                timestamp_month=entry.timestamp_month,
                frequency=entry.frequency,
                recency_score=new_score,
                utility_score=entry.utility_score,
                metadata=entry.metadata,
            )
            new_entries.append(updated)
        new_inventory = MemoryInventory(
            entries=tuple(new_entries),
            last_updated_month=current_month,
            version=self._inventory.version,
        )
        return MemoryStore(new_inventory)

    def update_utility(
        self,
        source_paper_id: str,
        utility_delta: float,
        *,
        ema_alpha: float = 0.3,
    ) -> MemoryStore:
        """Update utility score for entry with given source_paper_id using EMA."""
        new_entries: list[MemoryEntry] = []
        for entry in self._inventory.entries:
            if entry.source_paper_id == source_paper_id:
                new_utility = ema_alpha * utility_delta + (1.0 - ema_alpha) * entry.utility_score
                updated = MemoryEntry(
                    innovation=entry.innovation,
                    source_paper_id=entry.source_paper_id,
                    timestamp_month=entry.timestamp_month,
                    frequency=entry.frequency,
                    recency_score=entry.recency_score,
                    utility_score=new_utility,
                    metadata=entry.metadata,
                )
                new_entries.append(updated)
            else:
                new_entries.append(entry)
        new_inventory = MemoryInventory(
            entries=tuple(new_entries),
            last_updated_month=self._inventory.last_updated_month,
            version=self._inventory.version,
        )
        return MemoryStore(new_inventory)

    def format_for_prompt(self, top_n: int = 20) -> str:
        """Format top-n entries as a numbered list for LLM prompts."""
        entries = self.query(top_n)
        if not entries:
            return "(no entries in memory)"
        lines: list[str] = []
        for i, entry in enumerate(entries, start=1):
            inn = entry.innovation
            line = (
                f'{i}. base_direction="{inn.base_direction}", '
                f'operator="{inn.operator}", '
                f'gap="{inn.gap}"'
            )
            lines.append(line)
        return "\n".join(lines)
