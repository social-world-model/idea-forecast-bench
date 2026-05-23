#!/usr/bin/env python
"""Phase-8 ablation runner.

Two modes:
  --mode smoke   deterministic stub metrics. Confirms the result-table
                 plumbing without touching the live benchmark.
  --mode live    invoke live_idea_bench.backtest.evaluate_at_cutoff (or
                 a custom evaluator) per ablation config. Requires the
                 trained policy + judge + index artifacts.

Writes reports/results.md (main table + ablation tables) and
reports/results.json.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from forecaster.foresight.ablations import (
    AblationConfig,
    AblationResult,
    baseline_set,
)
from forecaster.foresight.metrics import (
    impact_stratified_breakdown,
    mmd_rbf,
    wasserstein_1d,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("phase8")
REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- evaluators


def smoke_evaluator(cfg: AblationConfig) -> AblationResult:
    """Deterministic stub: derive 'metrics' from the ablation cell hash.

    Used to exercise reporting end-to-end without running the bench.
    Numbers are *not* meaningful; they are deterministic per cell so
    we can assert the table renders consistently.
    """
    seed = hash(cfg.name) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    # Baseline-style spread; "ours" is the reference cell.
    base = {
        "hit_at_k": float(np.clip(rng.normal(0.50, 0.05), 0.0, 1.0)),
        "mrr": float(np.clip(rng.normal(0.40, 0.05), 0.0, 1.0)),
        "novelty": float(np.clip(rng.normal(0.55, 0.05), 0.0, 1.0)),
        "diversity": float(np.clip(rng.normal(0.45, 0.05), 0.0, 1.0)),
        "mmd_to_truth": float(np.clip(rng.normal(0.20, 0.04), 0.0, None)),
        "wasserstein_to_truth": float(np.clip(rng.normal(0.30, 0.05), 0.0, None)),
    }
    # Encode the plan's expected directional effects (just for the stub).
    if cfg.reward_variant == "embedding_threshold":
        base["hit_at_k"] *= 0.92
        base["novelty"] *= 0.95
    elif cfg.reward_variant == "raw_judge":
        base["hit_at_k"] *= 0.95
        base["mmd_to_truth"] *= 1.20      # raw-judge tends to drift more
    if cfg.decomposition_variant == "single_shot_k":
        base["diversity"] *= 0.85
    if cfg.rubric_variant == "co_evolve":
        base["hit_at_k"] *= 1.02
        base["wasserstein_to_truth"] *= 0.95
    if cfg.gate_variant != "both":
        base["hit_at_k"] *= 0.93           # losing gates -> reward gameable
    return AblationResult(config=cfg, metrics=base, notes="smoke stub")


def live_evaluator(cfg: AblationConfig) -> AblationResult:  # pragma: no cover
    """Invoke the real benchmark per ablation cell.

    The implementation is intentionally light: it just defers to the
    project's `evaluate_at_cutoff` and reports the standard metrics.
    Plug in your trained policy path + cutoff schedule before running.
    """
    raise NotImplementedError(
        "live_evaluator: wire this to live_idea_bench.backtest.evaluate_at_cutoff "
        "with your trained policy + test cutoffs. See scripts/run_eval_trained.sh "
        "for the standard invocation pattern."
    )


# --------------------------------------------------------------------------- reporting


def _format_table(results: list[AblationResult]) -> str:
    metric_keys = sorted({k for r in results for k in r.metrics.keys()})
    header = ["config"] + metric_keys
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for r in results:
        cells = [f"`{r.config.name}`"] + [
            f"{r.metrics.get(k, 0.0):.4f}" for k in metric_keys
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_reports(
    results: list[AblationResult],
    *,
    report_md: Path,
    report_json: Path,
) -> None:
    report_md.parent.mkdir(parents=True, exist_ok=True)
    body: list[str] = ["# Phase 8 — Foresight ablation results\n"]

    ours = [r for r in results if r.config.name == "ours"]
    others = [r for r in results if r.config.name != "ours"]
    body.append("## Main table (test cutoffs)\n")
    body.append(_format_table(ours))
    body.append("\n## Ablations (one switch each)\n")
    body.append(_format_table(others))

    report_md.write_text("\n".join(body) + "\n", encoding="utf-8")
    report_json.write_text(
        json.dumps([r.to_json() for r in results], indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "live"], default="smoke")
    ap.add_argument("--report-md", default=str(REPO_ROOT / "reports/results.md"))
    ap.add_argument("--report-json", default=str(REPO_ROOT / "reports/results.json"))
    args = ap.parse_args()

    evaluator: Callable[[AblationConfig], AblationResult] = (
        smoke_evaluator if args.mode == "smoke" else live_evaluator
    )
    configs = baseline_set()
    logger.info("running %d ablations in mode=%s", len(configs), args.mode)
    results: list[AblationResult] = []
    for cfg in configs:
        res = evaluator(cfg)
        results.append(res)
        logger.info(
            "%-20s  hit@k=%.3f mrr=%.3f novelty=%.3f div=%.3f mmd=%.3f w1=%.3f",
            cfg.name,
            res.metrics.get("hit_at_k", 0.0),
            res.metrics.get("mrr", 0.0),
            res.metrics.get("novelty", 0.0),
            res.metrics.get("diversity", 0.0),
            res.metrics.get("mmd_to_truth", 0.0),
            res.metrics.get("wasserstein_to_truth", 0.0),
        )

    write_reports(
        results,
        report_md=Path(args.report_md),
        report_json=Path(args.report_json),
    )
    print(f"wrote {args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
