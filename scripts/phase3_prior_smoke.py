#!/usr/bin/env python
"""Phase-3 smoke: confirm the D_z → prior-SFT bridge + sample_z stub-wire.

What this does:
  1. Synthesizes a tiny corpus + a 6-row hindsight JSONL.
  2. Runs augment_hindsight_rows to produce a D_z with memory_text + closed
     operator labels (Decision 2's `other` rows included).
  3. Converts the D_z to SFT samples (`{input, target, ...}` shape) the
     existing forecaster/prior/trainer.py:train_prior consumes verbatim.
  4. Verifies the operator distribution over the sample is non-degenerate.
  5. Calls sample_z() with a stub sampler to prove the API shape works.

It does NOT call train_prior() — that needs a real GPU + ~minutes per row.
Run the dedicated SFT script (scripts/run_prior_sft.sh or the runner in
examples/run_prior_sft.py) when you want to actually train.
"""
from __future__ import annotations

import json
import logging
import tempfile
from collections import Counter
from pathlib import Path

from live_idea_bench.models import PaperRecord
from forecaster.foresight.dz import augment_hindsight_rows
from forecaster.foresight.prior_api import operator_distribution, sample_z
from forecaster.foresight.prior_io import build_sft_samples_from_dz, save_sft_jsonl
from forecaster.models import Innovation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("phase3_smoke")
REPO_ROOT = Path(__file__).resolve().parents[1]


SYNTHETIC_RAW_ROWS = [
    {
        "topic_id": "rag", "episode_id": "E1",
        "cutoff_date": "2024-03-31",
        "future_paper_id": "p_a", "future_paper_title": "RAG meets time series",
        "future_paper_published_date": "2024-05-20",
        "innovation": {"base_direction": "rag", "operator": "extend",
                       "gap": "extends RAG to non-stationary forecasting"},
        "context_paper_count": 3,
    },
    {
        "topic_id": "rag", "episode_id": "E2",
        "cutoff_date": "2024-04-30",
        "future_paper_id": "p_b", "future_paper_title": "RAG-as-Plan composition",
        "future_paper_published_date": "2024-06-10",
        "innovation": {"base_direction": "rag", "operator": "compose",
                       "gap": "composes retrieval with a planner"},
        "context_paper_count": 3,
    },
    {
        "topic_id": "agents", "episode_id": "E1",
        "cutoff_date": "2024-04-30",
        "future_paper_id": "p_c", "future_paper_title": "Tool agents on long horizons",
        "future_paper_published_date": "2024-06-15",
        "innovation": {"base_direction": "tool agents", "operator": "benchmark",
                       "gap": "proposes a long-horizon evaluation"},
        "context_paper_count": 3,
    },
    {
        "topic_id": "agents", "episode_id": "E2",
        "cutoff_date": "2024-05-31",
        "future_paper_id": "p_d", "future_paper_title": "Multi-modal agents",
        "future_paper_published_date": "2024-07-20",
        "innovation": {"base_direction": "tool agents", "operator": "transfer",
                       "gap": "ports image-grounded reasoning to agent tooling"},
        "context_paper_count": 3,
    },
    {
        "topic_id": "agents", "episode_id": "E3",
        "cutoff_date": "2024-06-30",
        "future_paper_id": "p_e", "future_paper_title": "Agent failure analysis",
        "future_paper_published_date": "2024-08-15",
        "innovation": {"base_direction": "tool agents", "operator": "analyze",
                       "gap": "categorizes agent failure modes"},
        "context_paper_count": 3,
    },
    {
        "topic_id": "rag", "episode_id": "E3",
        "cutoff_date": "2024-06-30",
        "future_paper_id": "p_f", "future_paper_title": "RAG with scale-aware retrievers",
        "future_paper_published_date": "2024-09-01",
        "innovation": {"base_direction": "rag", "operator": "scale",
                       "gap": "scales retrievers to 1B tokens"},
        "context_paper_count": 3,
    },
]


def _build_synthetic_corpus() -> dict[str, PaperRecord]:
    contexts = [
        ("ctx_rag_1", "2024-01-15", "rag", "Retrieval-augmented generation baseline."),
        ("ctx_rag_2", "2024-02-20", "rag", "Dense passage retrieval improvements."),
        ("ctx_rag_3", "2024-03-05", "rag", "Hybrid sparse-dense retrievers."),
        ("ctx_agents_1", "2024-01-30", "agents", "Tool-using LLM agents."),
        ("ctx_agents_2", "2024-02-25", "agents", "ReAct-style planning."),
        ("ctx_agents_3", "2024-03-10", "agents", "Agent memory and reflection."),
    ]
    out: dict[str, PaperRecord] = {}
    for pid, d, topic, summ in contexts:
        out[pid] = PaperRecord(
            paper_id=pid,
            title=f"Paper {pid}",
            month=d[:7],
            summary=summ,
            keywords=[topic],
            source_path="",
            published_date=d,
            metadata={"topic_id": topic},
        )
    # Also include the future papers in the corpus so we can prove they get
    # excluded from M_t by the split-by-cutoff guard.
    for r in SYNTHETIC_RAW_ROWS:
        pid = r["future_paper_id"]
        out[pid] = PaperRecord(
            paper_id=pid,
            title=r["future_paper_title"],
            month=r["future_paper_published_date"][:7],
            summary="(future paper synthetic abstract)",
            keywords=[r["topic_id"]],
            source_path="",
            published_date=r["future_paper_published_date"],
            metadata={"topic_id": r["topic_id"]},
        )
    return out


def _fake_sampler(memory_store, n, temperature):
    """Deterministic stub that returns 6 innovations spanning 3 operators."""
    bank = [
        Innovation("rag", "extend", "extends rag to long-context"),
        Innovation("rag", "compose", "composes rag with planning"),
        Innovation("rag", "transfer", "ports rag to vision-language"),
        Innovation("agents", "extend", "extends tool agents to multi-turn"),
        Innovation("agents", "benchmark", "proposes failure-mode benchmark"),
        Innovation("agents", "compose", "composes planner + critic"),
    ]
    return bank[:n]


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        hindsight_path = td_path / "hindsight.jsonl"
        dz_path = td_path / "dz.jsonl"
        sft_path = REPO_ROOT / "output/phase3_smoke/sft_dataset.jsonl"
        sft_path.parent.mkdir(parents=True, exist_ok=True)

        hindsight_path.write_text(
            "\n".join(json.dumps(r) for r in SYNTHETIC_RAW_ROWS), encoding="utf-8"
        )

        corpus = _build_synthetic_corpus()
        summary = augment_hindsight_rows(
            hindsight_path, dz_path,
            papers_by_id=corpus, horizon_months=3,
        )
        logger.info("augment summary: %s", summary.to_json())

        samples = build_sft_samples_from_dz(dz_path, drop_unmappable=True)
        save_sft_jsonl(samples, sft_path)
        logger.info("wrote %d SFT samples to %s", len(samples), sft_path)

        # Acceptance: operator distribution of the SFT targets is non-degenerate.
        target_ops = Counter()
        closed_ops = Counter()
        for s in samples:
            tgt = json.loads(s["target"])
            target_ops[tgt["operator"]] += 1
            closed_ops[s["operator_closed"]] += 1
        logger.info("target operator counts (free-text): %s", dict(target_ops))
        logger.info("target operator counts (closed)   : %s", dict(closed_ops))

        # sample_z stub check.
        z_list = sample_z(
            memory_text=samples[0]["memory_prompt"],
            n=6,
            temperature=0.9,
            sampler=_fake_sampler,
        )
        dist = operator_distribution(z_list)
        logger.info("sample_z operator distribution: %s", dist)
        assert len(z_list) == 6
        assert len(dist) >= 2, "sample_z stub returned a degenerate operator distribution"

        # Confirm the SFT JSONL is well-formed.
        for line in sft_path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            assert "input" in obj and "target" in obj
            assert "Current research memory" in obj["input"]
            tgt = json.loads(obj["target"])
            assert set(tgt.keys()) == {"base_direction", "operator", "gap"}
        logger.info("phase 3 smoke OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
