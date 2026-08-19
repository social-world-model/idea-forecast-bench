"""D_z augmentation: turn the raw hindsight JSONL into the Phase-1 D_z.

Input rows (existing `data/topic_hindsight/hindsight_samples.jsonl`):
  {topic_id, episode_id, cutoff_date, future_paper_id, future_paper_title,
   future_paper_published_date, innovation:{base_direction, operator, gap},
   context_paper_count}

Output rows (D_z, written by `augment_hindsight_rows`):
  {cutoff_t, topic_id, episode_id,
   target_z: {base_direction, operator, gap},
   operator_closed,                    # one of CLOSED_OPERATORS ∪ {"other"}
   source_future_id,                   # = future_paper_id
   future_paper_published_date,        # for downstream window asserts
   context_paper_ids: [...] | null,    # populated only if a corpus is supplied
   memory_text: str | null}            # populated only if a corpus is supplied
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forecaster.foresight.cutoffs import (
    FUTURE_WINDOW_HARD_LIMIT,
    _to_date,
)
from forecaster.foresight.memory import build_memory
from forecaster.foresight.operators import (
    OperatorInventory,
    load_operator_inventory,
    map_free_text_operator,
)
from live_idea_bench.backtest import split_train_future_by_cutoff
from live_idea_bench.models import PaperRecord

logger = logging.getLogger(__name__)


@dataclass
class DZRow:
    cutoff_t: str
    topic_id: str
    episode_id: str
    target_z: dict[str, str]  # {base_direction, operator, gap}
    operator_closed: str
    source_future_id: str
    future_paper_published_date: str
    context_paper_ids: list[str] | None = None
    memory_text: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cutoff_t": self.cutoff_t,
            "topic_id": self.topic_id,
            "episode_id": self.episode_id,
            "target_z": dict(self.target_z),
            "operator_closed": self.operator_closed,
            "source_future_id": self.source_future_id,
            "future_paper_published_date": self.future_paper_published_date,
        }
        if self.context_paper_ids is not None:
            payload["context_paper_ids"] = list(self.context_paper_ids)
        if self.memory_text is not None:
            payload["memory_text"] = self.memory_text
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


@dataclass
class AugmentationSummary:
    total_rows: int = 0
    train_window_rows: int = 0
    dropped_test_window: int = 0
    dropped_missing_cutoff: int = 0
    operator_closed_counts: dict[str, int] = field(default_factory=dict)
    other_ratio: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "train_window_rows": self.train_window_rows,
            "dropped_test_window": self.dropped_test_window,
            "dropped_missing_cutoff": self.dropped_missing_cutoff,
            "operator_closed_counts": dict(self.operator_closed_counts),
            "other_ratio": round(self.other_ratio, 4),
        }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "skipping malformed line %d in %s: %s", line_no, path, exc
                )


def augment_hindsight_rows(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    inventory: OperatorInventory | None = None,
    papers_by_id: dict[str, PaperRecord] | None = None,
    horizon_months: int = 3,
    enforce_train_window: bool = True,
    summary_path: str | Path | None = None,
) -> AugmentationSummary:
    """Stream-augment a hindsight JSONL into D_z.

    Args:
        input_jsonl: path to the raw hindsight samples (rows shaped like
            the existing `data/topic_hindsight/hindsight_samples.jsonl`).
        output_jsonl: D_z target path.
        inventory: closed-operator inventory; loads default if None.
        papers_by_id: optional corpus indexed by paper_id. If supplied,
            each row gets `context_paper_ids` (papers <= cutoff_t) and
            `memory_text = build_memory(...)`. If None, those fields stay
            null and can be backfilled later when the corpus is available.
        horizon_months: passed through to split_train_future_by_cutoff
            when papers_by_id is provided.
        enforce_train_window: when True (the default) drop rows whose
            cutoff_t lands on/after FUTURE_WINDOW_HARD_LIMIT. Surfaced in
            `dropped_test_window`.
        summary_path: if provided, write the AugmentationSummary as JSON.
    """
    inventory = inventory or load_operator_inventory()
    in_path = Path(input_jsonl)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hard_limit = _to_date(FUTURE_WINDOW_HARD_LIMIT)

    summary = AugmentationSummary()
    counts: dict[str, int] = dict.fromkeys(inventory.closed_ids, 0)
    counts[inventory.unmappable_bucket] = 0

    all_papers: list[PaperRecord] | None = None
    if papers_by_id is not None:
        all_papers = list(papers_by_id.values())

    with out_path.open("w", encoding="utf-8") as fout:
        for raw in _iter_jsonl(in_path):
            summary.total_rows += 1
            cutoff_t = str(raw.get("cutoff_date") or "").strip()
            if not cutoff_t:
                summary.dropped_missing_cutoff += 1
                continue
            try:
                cutoff_d = _to_date(cutoff_t)
            except ValueError:
                summary.dropped_missing_cutoff += 1
                continue
            if enforce_train_window and cutoff_d >= hard_limit:
                summary.dropped_test_window += 1
                continue

            innovation = raw.get("innovation") or {}
            target_z = {
                "base_direction": str(innovation.get("base_direction") or "").strip(),
                "operator": str(innovation.get("operator") or "").strip(),
                "gap": str(innovation.get("gap") or "").strip(),
            }
            op_closed = map_free_text_operator(target_z["operator"], inventory)
            counts[op_closed] = counts.get(op_closed, 0) + 1

            context_paper_ids: list[str] | None = None
            memory_text: str | None = None
            if all_papers is not None:
                train_papers, _future_papers, _future_end_month, _future_end_date = (
                    split_train_future_by_cutoff(
                        papers=all_papers,
                        cutoff_month=cutoff_t[:7],
                        cutoff_date=cutoff_t,
                        horizon_months=horizon_months,
                    )
                )
                context_paper_ids = [p.paper_id for p in train_papers]
                memory_text = build_memory(train_papers, cutoff_t=cutoff_t)

            row = DZRow(
                cutoff_t=cutoff_t,
                topic_id=str(raw.get("topic_id") or ""),
                episode_id=str(raw.get("episode_id") or ""),
                target_z=target_z,
                operator_closed=op_closed,
                source_future_id=str(raw.get("future_paper_id") or ""),
                future_paper_published_date=str(
                    raw.get("future_paper_published_date") or ""
                ),
                context_paper_ids=context_paper_ids,
                memory_text=memory_text,
                extra={
                    "future_paper_title": str(raw.get("future_paper_title") or ""),
                },
            )
            fout.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
            summary.train_window_rows += 1

    total_closed = sum(counts.values())
    summary.operator_closed_counts = counts
    other_count = counts.get(inventory.unmappable_bucket, 0)
    summary.other_ratio = (other_count / total_closed) if total_closed else 0.0

    if summary_path is not None:
        Path(summary_path).write_text(
            json.dumps(summary.to_json(), indent=2), encoding="utf-8"
        )

    logger.info(
        "augment_hindsight_rows: total=%d kept=%d dropped(test_window)=%d "
        "dropped(missing_cutoff)=%d other_ratio=%.3f",
        summary.total_rows,
        summary.train_window_rows,
        summary.dropped_test_window,
        summary.dropped_missing_cutoff,
        summary.other_ratio,
    )
    return summary


def load_dz_rows(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Read an augmented D_z back into a list of dicts (no schema coerce)."""
    return list(_iter_jsonl(Path(jsonl_path)))
