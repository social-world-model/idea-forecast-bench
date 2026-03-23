"""Build SFT training dataset for the innovation prior."""
from __future__ import annotations

import json
from pathlib import Path

from forecaster.models import HindsightSample, innovation_to_dict
from forecaster.prior.memory import MemoryStore


def build_sft_samples(
    hindsight_samples: list[HindsightSample],
    *,
    max_memory_entries: int = 20,
) -> list[dict[str, str]]:
    """Build SFT training samples from hindsight dataset.

    Each sample has {"input": memory_prompt_string, "target": innovation_json_string}.
    Replays the memory chronologically: for each hindsight sample, builds the memory
    state at that cutoff (using only earlier samples), formats it as a prompt.

    Args:
        hindsight_samples: Temporally ordered list of HindsightSample objects.
        max_memory_entries: Max entries to include in memory prompt.

    Returns:
        List of {"input": str, "target": str} dicts ready for SFT.
    """
    if not hindsight_samples:
        return []
    results: list[dict[str, str]] = []
    memory = MemoryStore.empty(hindsight_samples[0].cutoff_month)
    for sample in hindsight_samples:
        memory_summary = memory.format_for_prompt(top_n=max_memory_entries)
        target = json.dumps(innovation_to_dict(sample.innovation), ensure_ascii=False)
        results.append({"input": memory_summary, "target": target})
        memory = memory.append(
            sample.innovation,
            source_paper_id=sample.future_paper_id,
            month=sample.cutoff_month,
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
