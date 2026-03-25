"""Build SFT training dataset for the innovation prior."""
from __future__ import annotations

import json
from pathlib import Path

from forecaster.models import HindsightSample, innovation_to_dict, memory_inventory_to_dict
from forecaster.prior.memory import build_memory_store_from_hindsight_samples


def build_sft_samples(
    hindsight_samples: list[HindsightSample],
    *,
    max_memory_entries: int = 20,
) -> list[dict[str, str]]:
    """Build SFT training samples from hindsight dataset.

    Each sample has {"input": memory_prompt_string, "target": innovation_json_string}
    plus provenance metadata. For each hindsight sample, the input prompt is built
    from the exact memory snapshot that would have been legal at that cutoff.

    Args:
        hindsight_samples: Temporally ordered list of HindsightSample objects.
        max_memory_entries: Max entries to include in memory prompt.

    Returns:
        List of training dicts ready for SFT and audit logging.
    """
    if not hindsight_samples:
        return []
    sorted_samples = sorted(
        hindsight_samples,
        key=lambda sample: (
            sample.cutoff_month,
            sample.future_paper_published_date,
            sample.future_paper_id,
        ),
    )
    results: list[dict[str, str]] = []
    for sample in sorted_samples:
        memory = build_memory_store_from_hindsight_samples(
            sorted_samples,
            sample.cutoff_month,
            exclude_future_paper_ids={sample.future_paper_id},
        )
        memory_summary = memory.format_for_prompt(top_n=max_memory_entries)
        target = json.dumps(innovation_to_dict(sample.innovation), ensure_ascii=False)
        results.append(
            {
                "input": memory_summary,
                "target": target,
                "cutoff_month": sample.cutoff_month,
                "future_paper_id": sample.future_paper_id,
                "future_paper_published_date": sample.future_paper_published_date,
                "memory_inventory": json.dumps(
                    memory_inventory_to_dict(memory.inventory),
                    ensure_ascii=False,
                ),
            }
        )
    return results


def save_sft_dataset(samples: list[dict[str, str]], path: str | Path) -> None:
    """Save samples to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(s, ensure_ascii=False) for s in samples]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_sft_dataset(path: str | Path) -> list[dict[str, str]]:
    """Load samples from JSONL file."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]
