"""Tests for SFT dataset builder (TDD: RED phase)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from forecaster.models import Innovation, HindsightSample, MemoryEntry, MemoryInventory
from forecaster.prior.memory import MemoryStore
from forecaster.prior.sft_dataset import build_sft_samples, save_sft_dataset, load_sft_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_hindsight_sample(i: int, month: str = "2024-01") -> HindsightSample:
    return HindsightSample(
        context_paper_ids=(f"ctx-{i}-a", f"ctx-{i}-b"),
        cutoff_month=month,
        future_paper_id=f"future-{i}",
        future_paper_published_date=f"{month}-15",
        innovation=Innovation(
            base_direction=f"direction {i}",
            operator="extend",
            gap=f"This is gap number {i}.",
        ),
    )


def _make_samples(n: int) -> list[HindsightSample]:
    months = ["2023-01", "2023-03", "2023-06", "2023-09", "2024-01"]
    return [_make_hindsight_sample(i, months[i % len(months)]) for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_sft_samples_returns_dicts():
    """Each sample must have 'input' and 'target' string keys."""
    hindsight = _make_samples(5)
    samples = build_sft_samples(hindsight)
    assert len(samples) > 0
    for s in samples:
        assert "input" in s
        assert "target" in s
        assert isinstance(s["input"], str)
        assert isinstance(s["target"], str)


def test_build_sft_samples_chronological_memory():
    """For sample N, the memory context should be built from samples 0..N-1.

    We verify indirectly: first sample has empty memory (no prior context entries),
    and later samples should have growing memory (more content in 'input').
    """
    hindsight = _make_samples(5)
    samples = build_sft_samples(hindsight)
    # First sample has empty memory → input mentions empty or no numbered entries
    first_input = samples[0]["input"]
    last_input = samples[-1]["input"]
    # The last input should be longer because the memory has more entries
    assert len(last_input) >= len(first_input)


def test_build_sft_samples_empty_input_returns_empty_list():
    """Empty hindsight list should yield empty samples list."""
    samples = build_sft_samples([])
    assert samples == []


def test_build_sft_samples_target_is_valid_json():
    """Target must be valid JSON with the three required fields."""
    hindsight = _make_samples(3)
    samples = build_sft_samples(hindsight)
    for s in samples:
        obj = json.loads(s["target"])
        assert "base_direction" in obj
        assert "operator" in obj
        assert "gap" in obj


def test_build_sft_samples_uses_shared_prompt_contract():
    """Training rows should already contain the shared user prompt wrapper."""
    samples = build_sft_samples(_make_samples(1))
    assert "Current research memory" in samples[0]["input"]
    assert "predict the most likely next innovation" in samples[0]["input"]
    assert "memory_prompt" in samples[0]


def test_build_sft_samples_supports_replayed_memory_snapshots() -> None:
    hindsight = [_make_hindsight_sample(1, "2024-01")]
    replay_memory = MemoryStore(
        MemoryInventory(
            entries=(
                MemoryEntry(
                    innovation=Innovation(
                        base_direction="replayed direction",
                        operator="compose",
                        gap="utility-shaped gap",
                    ),
                    source_paper_id="replay-paper",
                    timestamp_month="2023-12",
                    frequency=3,
                    recency_score=0.8,
                    utility_score=0.9,
                ),
            ),
            last_updated_month="2024-01",
        )
    )

    samples = build_sft_samples(
        hindsight,
        memory_snapshots_by_cutoff={"2024-01": replay_memory},
    )

    assert "replayed direction" in samples[0]["memory_prompt"]
    assert '"utility":0.9' in samples[0]["memory_prompt"]


def test_build_sft_samples_excludes_future_unavailable_at_cutoff():
    """Memory for cutoff t must exclude hindsight labels from papers published after t."""
    early_future = HindsightSample(
        context_paper_ids=("ctx-a",),
        cutoff_month="2024-01",
        future_paper_id="future-late",
        future_paper_published_date="2024-03-15",
        innovation=Innovation(
            base_direction="late direction",
            operator="extend",
            gap="late gap",
        ),
    )
    target = HindsightSample(
        context_paper_ids=("ctx-b",),
        cutoff_month="2024-02",
        future_paper_id="future-target",
        future_paper_published_date="2024-04-15",
        innovation=Innovation(
            base_direction="target direction",
            operator="extend",
            gap="target gap",
        ),
    )

    samples = build_sft_samples([early_future, target])

    target_sample = next(sample for sample in samples if sample["future_paper_id"] == "future-target")
    assert "late direction" not in target_sample["input"]


def test_save_and_load_sft_dataset_roundtrip():
    """Save to JSONL, load back — contents should be identical."""
    hindsight = _make_samples(4)
    samples = build_sft_samples(hindsight)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sft.jsonl"
        save_sft_dataset(samples, path)
        loaded = load_sft_dataset(path)
    assert loaded == samples
