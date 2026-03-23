"""Tests for SFT dataset builder (TDD: RED phase)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from forecaster.models import Innovation, HindsightSample
from forecaster.prior.sft_dataset import build_sft_samples, save_sft_dataset, load_sft_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_hindsight_sample(i: int, month: str = "2024-01") -> HindsightSample:
    return HindsightSample(
        context_paper_ids=(f"ctx-{i}-a", f"ctx-{i}-b"),
        cutoff_month=month,
        future_paper_id=f"future-{i}",
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


def test_save_and_load_sft_dataset_roundtrip():
    """Save to JSONL, load back — contents should be identical."""
    hindsight = _make_samples(4)
    samples = build_sft_samples(hindsight)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sft.jsonl"
        save_sft_dataset(samples, path)
        loaded = load_sft_dataset(path)
    assert loaded == samples
