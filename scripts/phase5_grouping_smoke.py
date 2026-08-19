#!/usr/bin/env python
"""Phase-5 smoke: exercise make_reward_fn under correct + broken grouping.

  1. Build a tiny artifact dir (indices + rubric).
  2. Drive reward_fn with two well-formed groups (num_generations=4 each).
     Expect: invariant passes, dedup penalty fires on a triplicate.
  3. Drive reward_fn with a mixed group (different operators per row).
     Expect: GroupingInvariantError raised.
"""
from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from live_idea_bench.models import PaperRecord
from forecaster.foresight.indices import HashingEmbedder, build_cutoff_indices
from forecaster.foresight.rubric import Rubric, save_rubric, stamp_metadata
from forecaster.foresight.trainer_wiring import make_reward_fn
from forecaster.foresight.grouping import GroupingInvariantError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("phase5_smoke")


@dataclass
class StubConfig:
    reward_mode: str = "foresight"
    foresight_artifact_dir: str = ""
    foresight_embedder: str = "hashing:128"
    foresight_judge_mode: str = "stub"
    num_generations: int = 4
    grouping_assert: bool = True
    dedup_penalty: float = 0.3
    dedup_jaccard_threshold: float = 0.5


def _extra(cutoff: str, base: str, op: str, gap: str) -> str:
    return json.dumps({
        "cutoff_date": cutoff,
        "topic_id": "rag",
        "innovation": {"base_direction": base, "operator": op, "gap": gap},
        "prompt_mode": "z_conditioned_realization",
    })


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "artifact"
        (root / "indices").mkdir(parents=True)
        (root / "rubrics").mkdir(parents=True)

        papers = [
            PaperRecord("hist1", "Dense retrieval baseline", "2024-04", "RAG dense retriever.",
                        ["rag"], "", "2024-04-15"),
            PaperRecord("fut1", "RAG-extension", "2024-08", "Retrieval extension and new gap.",
                        ["rag"], "", "2024-08-15"),
            PaperRecord("fut2", "RAG composition", "2024-08", "Composes retrievers and planners.",
                        ["rag"], "", "2024-08-20"),
        ]
        embedder = HashingEmbedder(dim=128, seed=7)
        build_cutoff_indices(
            papers=papers,
            cutoff_dates=["2024-06-30"],
            horizon_months=3,
            embedder=embedder,
            save_dir=root / "indices",
        )
        save_rubric(
            Rubric(
                topic_id="rag", cutoff_t="2024-06-30",
                criteria=("must extend retrieval with a new gap",),
                operator_focus=("limitation_extension",),
                version=1, metadata=stamp_metadata(model="smoke"),
            ),
            root / "rubrics" / "rag.json",
        )

        cfg = StubConfig(foresight_artifact_dir=str(root))
        reward_fn = make_reward_fn(cfg, trainer_name="grpo")

        # ---- happy path: two well-formed groups (4 each) ----
        extras_ok = [
            *[_extra("2024-06-30", "rag", "extend", "x")] * 4,
            *[_extra("2024-06-30", "rag", "compose", "y")] * 4,
        ]
        completions = [
            # group 1: three near-duplicates + one unique
            "We extend retrieval with a novel time-series adaptation, new gap addressed.",
            "We extend retrieval with a novel time-series adaptation, new gap addressed.",
            "We extend retrieval with a novel time-series adaptation, new gap addressed.",
            "Independent realization that introduces a new layer for retrieval extension.",
            # group 2
            "We compose retrievers and planners to introduce a new integrated pipeline.",
            "We integrate planner and retriever in a new way.",
            "We combine retrievers with planning agents.",
            "Composition of retriever and planner modules; new pipeline.",
        ]
        rewards = reward_fn(completions, extra_info=extras_ok)
        logger.info("rewards under correct grouping: %s", rewards)
        # The first three should suffer the dedup penalty relative to a fresh sample.
        assert rewards[3] >= max(rewards[0:3])

        # ---- broken path: mixed operators inside a single group ----
        extras_bad = [
            _extra("2024-06-30", "rag", "extend", "x"),
            _extra("2024-06-30", "rag", "extend", "x"),
            _extra("2024-06-30", "rag", "compose", "x"),  # drift
            _extra("2024-06-30", "rag", "extend", "x"),
        ]
        completions_bad = completions[:4]
        try:
            reward_fn(completions_bad, extra_info=extras_bad)
        except GroupingInvariantError as exc:
            logger.info("grouping invariant correctly raised: %s", exc)
        else:
            raise AssertionError("expected GroupingInvariantError on mixed-operator group")

        logger.info("phase 5 smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
