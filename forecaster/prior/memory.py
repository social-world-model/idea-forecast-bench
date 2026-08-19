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

from live_idea_bench.papers import date_to_ordinal, month_start_date

from forecaster.models import (
    HindsightSample,
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
_DEFAULT_QUERY_RECENCY_WEIGHT: float = 0.45
_DEFAULT_QUERY_FREQUENCY_WEIGHT: float = 0.35
_DEFAULT_QUERY_UTILITY_WEIGHT: float = 0.20


def _months_between(earlier: str, later: str) -> int:
    """Months between two 'YYYY-MM' strings."""
    y1, m1 = map(int, earlier.split("-"))
    y2, m2 = map(int, later.split("-"))
    return (y2 - y1) * 12 + (m2 - m1)


def _innovation_key(innovation: Innovation) -> tuple[str, str, str]:
    return (innovation.base_direction, innovation.operator, innovation.gap)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalized_utility(value: float) -> float:
    return _clamp01(0.5 + (0.5 * math.tanh(value)))


def hindsight_sample_available_by_cutoff(
    sample: HindsightSample,
    cutoff_month: str,
) -> bool:
    """Return whether the hindsight source paper is available by this cutoff."""
    published_date = str(sample.future_paper_published_date or "").strip()
    if not published_date:
        return sample.future_paper_month <= cutoff_month
    return date_to_ordinal(published_date) <= date_to_ordinal(month_start_date(cutoff_month))


def build_memory_store_from_hindsight_samples(
    hindsight_samples: list[HindsightSample],
    cutoff_month: str,
    *,
    exclude_future_paper_ids: set[str] | None = None,
) -> MemoryStore:
    """Materialize the legal memory snapshot for a given cutoff."""
    excluded = exclude_future_paper_ids or set()
    eligible_samples = sorted(
        (
            sample
            for sample in hindsight_samples
            if sample.future_paper_id not in excluded
            and hindsight_sample_available_by_cutoff(sample, cutoff_month)
        ),
        key=lambda sample: (
            sample.future_paper_published_date,
            sample.future_paper_id,
            sample.cutoff_month,
        ),
    )
    store = MemoryStore.empty(cutoff_month)
    for sample in eligible_samples:
        store = store.append(
            sample.innovation,
            source_paper_id=sample.future_paper_id,
            month=sample.future_paper_month,
            metadata={
                "source_cutoff_month": sample.cutoff_month,
                "source_published_date": sample.future_paper_published_date,
            },
        )
    return store


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
    def load(cls, path: str | Path, *, cutoff_month: str | None = None) -> MemoryStore:
        """Load from JSON file. Returns empty store if file doesn't exist.

        Args:
            path: Path to the JSON file.
            cutoff_month: Optional inference cutoff month ('YYYY-MM'). When provided,
                logs a warning if the memory's last_updated_month is newer than the
                cutoff, which would indicate a temporal leakage risk.
        """
        path = Path(path)
        if not path.exists():
            return cls.empty("1970-01")
        raw = json.loads(path.read_text(encoding="utf-8"))
        inventory = memory_inventory_from_dict(raw)
        if cutoff_month and inventory.last_updated_month > cutoff_month:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Memory last_updated_month=%s is newer than inference cutoff_month=%s. "
                "Ensure the memory was built only from papers up to the cutoff to avoid leakage.",
                inventory.last_updated_month,
                cutoff_month,
            )
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
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryStore:
        """Add a new entry or increment frequency for duplicate. Returns new MemoryStore."""
        key = _innovation_key(innovation)
        new_entries: list[MemoryEntry] = []
        found = False
        for entry in self._inventory.entries:
            if _innovation_key(entry.innovation) == key:
                merged_metadata = dict(entry.metadata)
                if metadata:
                    for meta_key, meta_value in metadata.items():
                        merged_metadata.setdefault(meta_key, meta_value)
                updated = MemoryEntry(
                    innovation=entry.innovation,
                    source_paper_id=entry.source_paper_id,
                    timestamp_month=entry.timestamp_month,
                    frequency=entry.frequency + 1,
                    recency_score=entry.recency_score,
                    utility_score=entry.utility_score,
                    metadata=merged_metadata,
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
                    metadata=dict(metadata or {}),
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
        recency_weight: float = _DEFAULT_QUERY_RECENCY_WEIGHT,
        frequency_weight: float | None = None,
        utility_weight: float | None = None,
    ) -> list[MemoryEntry]:
        """Return top-n entries ranked by weighted score.

        Score is a frozen blend of recency, frequency, and delayed utility.
        """
        entries = list(self._inventory.entries)
        if not entries:
            return []
        max_freq = max(e.frequency for e in entries) or 1
        if frequency_weight is None and utility_weight is None:
            remaining = max(0.0, 1.0 - recency_weight)
            extra_total = _DEFAULT_QUERY_FREQUENCY_WEIGHT + _DEFAULT_QUERY_UTILITY_WEIGHT
            frequency_weight = remaining * (_DEFAULT_QUERY_FREQUENCY_WEIGHT / extra_total)
            utility_weight = remaining * (_DEFAULT_QUERY_UTILITY_WEIGHT / extra_total)
        else:
            frequency_weight = _DEFAULT_QUERY_FREQUENCY_WEIGHT if frequency_weight is None else frequency_weight
            utility_weight = _DEFAULT_QUERY_UTILITY_WEIGHT if utility_weight is None else utility_weight
        total_weight = recency_weight + frequency_weight + utility_weight
        if total_weight <= 0:
            total_weight = 1.0
        def score(entry: MemoryEntry) -> float:
            norm_freq = entry.frequency / max_freq
            return (
                (recency_weight * _clamp01(entry.recency_score))
                + (frequency_weight * _clamp01(norm_freq))
                + (utility_weight * _normalized_utility(entry.utility_score))
            ) / total_weight
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
        metadata: dict[str, Any] | None = None,
    ) -> MemoryStore:
        """Update utility score for entry with given source_paper_id using EMA."""
        new_entries: list[MemoryEntry] = []
        for entry in self._inventory.entries:
            if entry.source_paper_id == source_paper_id:
                new_utility = ema_alpha * utility_delta + (1.0 - ema_alpha) * entry.utility_score
                merged_metadata = dict(entry.metadata)
                if metadata:
                    merged_metadata.update(metadata)
                updated = MemoryEntry(
                    innovation=entry.innovation,
                    source_paper_id=entry.source_paper_id,
                    timestamp_month=entry.timestamp_month,
                    frequency=entry.frequency,
                    recency_score=entry.recency_score,
                    utility_score=new_utility,
                    metadata=merged_metadata,
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

    def exclude_source_paper_ids(self, source_paper_ids: set[str]) -> MemoryStore:
        """Return a snapshot with selected source paper ids removed."""
        excluded = {paper_id for paper_id in source_paper_ids if str(paper_id).strip()}
        if not excluded:
            return self
        new_inventory = MemoryInventory(
            entries=tuple(
                entry for entry in self._inventory.entries if entry.source_paper_id not in excluded
            ),
            last_updated_month=self._inventory.last_updated_month,
            version=self._inventory.version,
        )
        return MemoryStore(new_inventory)

    def apply_utility_overrides(
        self,
        overrides: dict[str, tuple[float, dict[str, Any] | None]],
    ) -> MemoryStore:
        """Return a snapshot with explicit utility scores/metadata merged in."""
        if not overrides:
            return self
        new_entries: list[MemoryEntry] = []
        for entry in self._inventory.entries:
            override = overrides.get(entry.source_paper_id)
            if override is None:
                new_entries.append(entry)
                continue
            utility_score, metadata = override
            merged_metadata = dict(entry.metadata)
            if metadata:
                merged_metadata.update(metadata)
            new_entries.append(
                MemoryEntry(
                    innovation=entry.innovation,
                    source_paper_id=entry.source_paper_id,
                    timestamp_month=entry.timestamp_month,
                    frequency=entry.frequency,
                    recency_score=entry.recency_score,
                    utility_score=float(utility_score),
                    metadata=merged_metadata,
                )
            )
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
            payload = {
                "base_direction": inn.base_direction,
                "operator": inn.operator,
                "gap": inn.gap,
                "timestamp_month": entry.timestamp_month,
                "frequency": entry.frequency,
                "recency": round(float(entry.recency_score), 4),
                "utility": round(float(entry.utility_score), 4),
            }
            line = (
                f"{i}. "
                f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
            )
            lines.append(line)
        return "\n".join(lines)
