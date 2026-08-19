#!/usr/bin/env python
"""Phase-4 end-to-end smoke: save indices + rubric to disk, load through
`build_foresight_context`, drive TRL-style `reward_fn` with a tiny batch.

Demonstrates:
  * Real serialized artifact layout (indices/{future,history}_<cutoff>.npz + rubrics/{topic}.json).
  * `make_reward_fn(config, ...)` loads them and returns a callable matching
    the TRL contract `reward_fn(completions, **kwargs) -> list[float]`.
  * Four representative completions exercise the four sanity cases.
"""
from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from forecaster.foresight.indices import (
    HashingEmbedder,
    build_cutoff_indices,
)
from forecaster.foresight.rubric import Rubric, save_rubric, stamp_metadata
from forecaster.foresight.trainer_wiring import (
    make_reward_fn,
)
from live_idea_bench.models import PaperRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("phase4_smoke")


@dataclass
class StubConfig:
    """Stand-in for OnlineRLTrainConfig — only the fields make_reward_fn reads."""
    reward_mode: str = "foresight"
    foresight_artifact_dir: str = ""
    foresight_embedder: str = "hashing:128"
    foresight_judge_mode: str = "stub"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "foresight_artifact"
        indices_dir = root / "indices"
        rubrics_dir = root / "rubrics"
        indices_dir.mkdir(parents=True)
        rubrics_dir.mkdir(parents=True)

        # ---- build a tiny corpus + indices ----
        papers = [
            PaperRecord(
                paper_id="hist1", title="Dense passage retrieval baseline",
                month="2024-04", summary="Dense passage retrieval for RAG.",
                keywords=["rag"], source_path="", published_date="2024-04-15",
            ),
            PaperRecord(
                paper_id="hist2", title="Hybrid sparse-dense retrievers",
                month="2024-05", summary="Hybrid sparse-dense retrievers.",
                keywords=["rag"], source_path="", published_date="2024-05-20",
            ),
            PaperRecord(
                paper_id="future1", title="RAG meets time series",
                month="2024-08",
                summary="Retrieval-augmented forecasting; novel extension.",
                keywords=["rag"], source_path="", published_date="2024-08-15",
            ),
            PaperRecord(
                paper_id="future2", title="Composed retrievers for agents",
                month="2024-08",
                summary="Composition of retrievers and planners; new pipeline.",
                keywords=["rag"], source_path="", published_date="2024-08-20",
            ),
        ]
        embedder = HashingEmbedder(dim=128, seed=11)
        bundles = build_cutoff_indices(
            papers=papers,
            cutoff_dates=["2024-06-30"],
            horizon_months=3,
            embedder=embedder,
            save_dir=indices_dir,
        )
        bundle = bundles["2024-06-30"]
        logger.info("history.size=%d future.size=%d",
                    bundle.history.size, bundle.future.size)

        # ---- write a synthetic rubric ----
        save_rubric(
            Rubric(
                topic_id="rag", cutoff_t="2024-06-30",
                criteria=(
                    "Must explicitly extend retrieval or compose retrievers with another component.",
                    "Must identify a concrete gap or limitation in pre-cutoff RAG work.",
                ),
                must_not=("Restates long-standing baselines without a novel operator.",),
                operator_focus=("limitation_extension", "method_composition"),
                version=1,
                metadata=stamp_metadata(model="smoke"),
            ),
            rubrics_dir / "rag.json",
        )

        # ---- wire reward_fn via make_reward_fn ----
        cfg = StubConfig(foresight_artifact_dir=str(root))
        reward_fn = make_reward_fn(cfg, trainer_name="grpo")
        logger.info("reward_fn loaded: %s", reward_fn.__name__)

        # ---- four representative completions ----
        completions = [
            # 1. Real emerged-style idea
            ("We extend retrieval with a novel time-series adaptation, "
             "building on hist1's dense retriever to introduce a long-context "
             "retrieval-extension layer."),
            # 2. Legacy-style idea (operator wrong: 'transfer' rollout while z=extend)
            ("This work proposes a brand new benchmark for retrieval-augmented agents."),
            # 3. Cites a non-existent paper
            ("Building on arxiv:9999.99999, we propose a new retrieval extension."),
            # 4. Long-standing framing (should be operator-fine but judge-low)
            ("Long-standing established line of retrieval-augmented work, "
             "building on hist1 with no new operator or gap."),
        ]
        extra_infos = []
        for _ in range(4):
            extra_infos.append(json.dumps({
                "cutoff_date": "2024-06-30",
                "topic_id": "rag",
                "innovation": {"base_direction": "rag", "operator": "extend", "gap": "x"},
                "prompt_mode": "z_conditioned_realization",
            }))

        rewards = reward_fn(completions, extra_info=extra_infos)
        logger.info("rewards: %s", rewards)
        for completion, reward in zip(completions, rewards, strict=False):
            logger.info(
                "%.3f | %s",
                reward,
                (completion[:80] + "…") if len(completion) > 80 else completion,
            )

        # Spot checks: positive > others.
        assert rewards[0] > 0.0, "positive case should produce a non-zero reward"
        # 2: operator gate trips → 0 (z=extend, rollout=benchmark)
        # 3: grounding trips for the bogus arxiv id (since it's not in known paper_ids)
        # 4: judge can fire but stub returns 0.5 — so reward depends on gates.
        assert rewards[1] == 0.0, "wrong-operator should be zeroed by operator gate"
        # NB the stub judge returns a constant 0.5, so we can't differentiate
        # cases 1 vs 4 by judge score alone; only the gates do the work here.
        logger.info("phase 4 smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
