"""Tests for the D_z ↔ prior SFT bridge + sample_z wrapper."""

from __future__ import annotations

import json
from pathlib import Path

from forecaster.foresight.dz import augment_hindsight_rows
from forecaster.foresight.prior_api import operator_distribution, sample_z
from forecaster.foresight.prior_io import (
    RawMemoryStore,
    build_sft_samples_from_dz,
    save_sft_jsonl,
)
from forecaster.models import Innovation


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _make_dz_with_memory(tmp_path: Path) -> Path:
    raw_in = tmp_path / "hindsight.jsonl"
    dz_out = tmp_path / "dz.jsonl"
    raw_rows = [
        {
            "topic_id": "rag",
            "episode_id": "E1",
            "cutoff_date": "2024-06-30",
            "future_paper_id": "p_pos_1",
            "future_paper_title": "p_pos_1 title",
            "future_paper_published_date": "2024-08-15",
            "innovation": {"base_direction": "rag", "operator": "extend", "gap": "x"},
            "context_paper_count": 1,
        },
        {
            "topic_id": "rag",
            "episode_id": "E2",
            "cutoff_date": "2024-05-31",
            "future_paper_id": "p_other_1",
            "future_paper_title": "p_other_1 title",
            "future_paper_published_date": "2024-07-15",
            "innovation": {"base_direction": "rag", "operator": "analyze", "gap": "y"},
            "context_paper_count": 1,
        },
    ]
    _write_jsonl(raw_in, raw_rows)
    # Need a corpus to populate memory_text; reuse a tiny PaperRecord map.
    from live_idea_bench.models import PaperRecord

    papers = {
        "ctx_1": PaperRecord(
            paper_id="ctx_1",
            title="ctx_1",
            month="2024-04",
            summary="A retrieval-augmented baseline.",
            keywords=["rag"],
            source_path="",
            published_date="2024-04-15",
            metadata={"topic_id": "rag"},
        )
    }
    augment_hindsight_rows(raw_in, dz_out, papers_by_id=papers)
    return dz_out


def test_drop_unmappable_filters_other(tmp_path: Path):
    dz_path = _make_dz_with_memory(tmp_path)
    samples = build_sft_samples_from_dz(dz_path, drop_unmappable=True)
    assert len(samples) == 1
    assert "rag" in samples[0]["input"]
    target = json.loads(samples[0]["target"])
    assert target == {"base_direction": "rag", "operator": "extend", "gap": "x"}
    assert samples[0]["operator_closed"] == "limitation_extension"


def test_keep_unmappable_when_flag_off(tmp_path: Path):
    dz_path = _make_dz_with_memory(tmp_path)
    samples = build_sft_samples_from_dz(dz_path, drop_unmappable=False)
    assert len(samples) == 2
    closed_ops = sorted(s["operator_closed"] for s in samples)
    assert closed_ops == ["limitation_extension", "other"]


def test_save_sft_jsonl_round_trip(tmp_path: Path):
    dz_path = _make_dz_with_memory(tmp_path)
    samples = build_sft_samples_from_dz(dz_path, drop_unmappable=False)
    out_path = save_sft_jsonl(samples, tmp_path / "sft.jsonl")
    loaded = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert loaded == samples


def test_rows_missing_memory_text_are_skipped(tmp_path: Path):
    raw_in = tmp_path / "hindsight.jsonl"
    dz_out = tmp_path / "dz.jsonl"
    _write_jsonl(
        raw_in,
        [
            {
                "topic_id": "rag",
                "episode_id": "E1",
                "cutoff_date": "2024-06-30",
                "future_paper_id": "p_pos_1",
                "future_paper_title": "p_pos_1",
                "future_paper_published_date": "2024-08-15",
                "innovation": {
                    "base_direction": "rag",
                    "operator": "extend",
                    "gap": "x",
                },
                "context_paper_count": 1,
            }
        ],
    )
    augment_hindsight_rows(
        raw_in, dz_out, papers_by_id=None
    )  # no corpus -> no memory_text
    samples = build_sft_samples_from_dz(dz_out, drop_unmappable=False)
    assert samples == []


def test_sample_z_uses_injected_sampler():
    """sample_z should pipe the memory string into an injected sampler."""

    captured: dict = {}

    def fake_sampler(
        store: RawMemoryStore, n: int, temperature: float
    ) -> list[Innovation]:
        captured["memory"] = store.format_for_prompt()
        captured["n"] = n
        captured["t"] = temperature
        return [
            Innovation(base_direction="rag", operator="extend", gap="a"),
            Innovation(base_direction="rag", operator="compose", gap="b"),
            Innovation(base_direction="rag", operator="transfer", gap="c"),
        ]

    out = sample_z("MEMORY HERE", n=3, temperature=0.7, sampler=fake_sampler)
    assert captured == {"memory": "MEMORY HERE", "n": 3, "t": 0.7}
    assert len(out) == 3
    dist = operator_distribution(out)
    assert set(dist.keys()) == {"extend", "compose", "transfer"}


def test_sample_z_n_zero_returns_empty():
    out = sample_z("m", n=0, sampler=lambda *_a: [])
    assert out == []
