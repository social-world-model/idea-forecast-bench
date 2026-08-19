#!/usr/bin/env python
"""Phase-4 LIVE smoke: full reward stack with the OpenAI judge.

Uses one of the rubrics that *passed* the Phase-2 AUC gate
(default: anomaly_detection) so we exercise the actual judge that
training will use, not a stub. Indices use the local HashingEmbedder
(no network) so the only LLM dependency in this run is the judge.

Steps:
  1. Build tiny indices for a fake cutoff using a 3-paper corpus.
  2. Load the live rubric from `--rubrics-dir` (must already exist).
  3. Build a ForesightContext with `judge_mode=live` (OpenAI).
  4. Drive `make_reward_fn` with four representative completions and
     print rewards per case.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from forecaster.foresight.indices import (
    HashingEmbedder,
    build_cutoff_indices,
)
from forecaster.foresight.rubric import load_rubric, save_rubric
from forecaster.foresight.trainer_wiring import make_reward_fn
from live_idea_bench.models import PaperRecord

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("phase4_live")

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class StubConfig:
    reward_mode: str = "foresight"
    foresight_artifact_dir: str = ""
    foresight_embedder: str = "hashing:128"
    foresight_judge_mode: str = "live"
    num_generations: int = 4
    grouping_assert: bool = True
    dedup_penalty: float = 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--topic",
        default="anomaly_detection",
        help="topic whose Phase-2 rubric you want to drive",
    )
    ap.add_argument("--rubrics-dir", default=str(REPO_ROOT / "rubrics_live"))
    args = ap.parse_args()

    src_rubric_path = Path(args.rubrics_dir) / f"{args.topic}.json"
    if not src_rubric_path.exists():
        raise SystemExit(
            f"rubric for topic={args.topic!r} not found at {src_rubric_path}. "
            "Run examples/forecaster/phase2_rubric_validation.py --mode live first."
        )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "artifact"
        (root / "indices").mkdir(parents=True)
        (root / "rubrics").mkdir(parents=True)

        # Tiny topic-aligned corpus.
        papers = [
            PaperRecord(
                paper_id="hist_anom_1",
                title="Classic anomaly detection baseline",
                month="2024-04",
                summary="Density-based anomaly detection baseline for time series.",
                keywords=[args.topic],
                source_path="",
                published_date="2024-04-15",
            ),
            PaperRecord(
                paper_id="hist_anom_2",
                title="One-class SVM revisited",
                month="2024-05",
                summary="One-class SVM with feature reweighting.",
                keywords=[args.topic],
                source_path="",
                published_date="2024-05-20",
            ),
            PaperRecord(
                paper_id="fut_anom_1",
                title="Diffusion-based anomaly detection for industrial sensors",
                month="2024-08",
                summary="Proposes a diffusion model that scores anomalies via reverse-process likelihood; "
                "extends prior density-based detectors with a novel score-based likelihood ratio.",
                keywords=[args.topic],
                source_path="",
                published_date="2024-08-15",
            ),
            PaperRecord(
                paper_id="fut_anom_2",
                title="Self-supervised anomaly detection via contrastive masking",
                month="2024-08",
                summary="Composes contrastive learning with masked autoencoding to produce novel pseudo-labels for unsupervised anomaly scoring.",
                keywords=[args.topic],
                source_path="",
                published_date="2024-08-20",
            ),
        ]
        embedder = HashingEmbedder(dim=128, seed=11)
        build_cutoff_indices(
            papers=papers,
            cutoff_dates=["2024-06-30"],
            horizon_months=3,
            embedder=embedder,
            save_dir=root / "indices",
        )
        save_rubric(
            load_rubric(src_rubric_path), root / "rubrics" / f"{args.topic}.json"
        )

        cfg = StubConfig(foresight_artifact_dir=str(root))
        reward_fn = make_reward_fn(cfg, trainer_name="grpo")
        logger.info("reward_fn loaded: %s (judge=live)", reward_fn.__name__)

        completions = [
            # 1. Solid future-ish idea, real grounding, correct operator
            (
                "We extend density-based anomaly detection with a diffusion-model-based "
                "score that addresses prior overfitting to clean signals. Building on "
                "hist_anom_1, we introduce a novel reverse-process likelihood ratio for "
                "industrial sensor streams."
            ),
            # 2. Wrong operator (z=extend; rollout is purely a benchmark proposal)
            (
                "This work proposes a new benchmark suite for anomaly detection on industrial "
                "sensors and reports baselines."
            ),
            # 3. Cites a non-existent paper
            (
                "Building on arxiv:9999.99999, we propose a new anomaly detection extension."
            ),
            # 4. Legacy framing — should be low judge but operator-fine
            (
                "Long-standing line of established anomaly detection work; we briefly extend "
                "hist_anom_1 with minor tuning, no novel operator or gap."
            ),
        ]
        extras = []
        for _ in range(4):
            extras.append(
                json.dumps(
                    {
                        "cutoff_date": "2024-06-30",
                        "topic_id": args.topic,
                        "innovation": {
                            "base_direction": "anomaly detection",
                            "operator": "extend",
                            "gap": "address overfitting in industrial sensors",
                        },
                        "prompt_mode": "z_conditioned_realization",
                    }
                )
            )

        rewards = reward_fn(completions, extra_info=extras)
        for c, r in zip(completions, rewards, strict=False):
            preview = c[:90] + ("…" if len(c) > 90 else "")
            logger.info("%.3f | %s", r, preview)
        print(json.dumps({"rewards": rewards}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
