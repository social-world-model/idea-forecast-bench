"""Run every baseline over one corpus and print a comparison table.

Reproducing the baseline table used to mean invoking `benchmark` once per
strategy, by hand, and remembering to keep every window/scoring flag identical
between runs -- if one run drifted, the numbers were no longer comparable.

Every strategy sees exactly the same windows and the same matcher; the shared
settings are printed once so a run is self-documenting.

Every baseline needs two things: an LLM provider (all five generate their
predictions with one) and VOYAGE_API_KEY (matching is embedding-only). Both are
checked before any run starts, so a missing key fails in one second rather than
five subprocesses later.

Usage:
    live-idea-bench baselines
    live-idea-bench baselines --only topic_trend,summary_prompting
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

#: Every baseline generates its predictions with an LLM -- topic_trend included:
#: it clusters the taxonomy itself, then asks a model to write the predictions
#: for each trending cluster. There is no LLM-free baseline.
ALL_BASELINES = (
    "topic_trend",
    "predictor_llm",
    "summary_prompting",
    "retrieval_prompting",
    "memory_prompting",
)

#: Any one of these satisfies the LLM requirement. OPENAI_BASE_URL is included
#: so a local OpenAI-compatible endpoint (vLLM, SGLang, a stub) counts.
LLM_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "TOGETHER_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_BASE_URL",
)

METRICS = (
    "avg_hit_at_k",
    "avg_precision_at_k",
    "avg_mrr",
    "avg_novelty",
    "avg_diversity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all baselines over one corpus under identical settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, default="data/csml/raw_markdown")
    parser.add_argument("--output-dir", type=str, default="output/baselines")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset of: " + ",".join(ALL_BASELINES),
    )
    parser.add_argument("--start-month", type=str, default="2024-01")
    parser.add_argument("--end-month", type=str, default="2025-06")
    parser.add_argument(
        "--min-cutoff-month",
        type=str,
        default=None,
        help="Earliest cutoff to EVALUATE (>= this). Lets --start-month load "
        "earlier papers as reading context while only scoring the test-period "
        "cutoffs, which is the only way to fix the window count independently "
        "of how much lead-in context the first cutoff gets.",
    )
    parser.add_argument(
        "--horizon-months",
        type=int,
        default=3,
        help="Months past the cutoff month. The cutoff month itself counts as "
        "future, so N=3 spans four calendar months.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-train-papers", type=int, default=5)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="Max future papers each prediction is matched against. The matcher "
        "issues one embedding call per (prediction, candidate) pair, so leaving "
        "this unset is O(top_k * |future|) calls per window -- ~1.5M calls for a "
        "208-window five-baseline sweep. Must be identical across baselines for "
        "the rows to stay comparable.",
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _selected(args: argparse.Namespace) -> list[str]:
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in ALL_BASELINES]
        if unknown:
            raise SystemExit(
                f"Unknown baseline(s): {', '.join(unknown)}. "
                f"Choose from: {', '.join(ALL_BASELINES)}"
            )
        return names
    return list(ALL_BASELINES)


def _run_one(strategy: str, args: argparse.Namespace, out_path: Path) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "live_idea_bench",
        "benchmark",
        "--input-dir",
        args.input_dir,
        "--strategy",
        strategy,
        "--start-month",
        args.start_month,
        "--end-month",
        args.end_month,
        "--horizon-months",
        str(args.horizon_months),
        "--top-k",
        str(args.top_k),
        "--min-train-papers",
        str(args.min_train_papers),
        "--workers",
        str(args.workers),
        "--output",
        str(out_path),
    ]
    if args.min_cutoff_month:
        cmd += ["--min-cutoff-month", args.min_cutoff_month]
    if args.candidate_limit:
        cmd += ["--candidate-limit", str(args.candidate_limit)]
    if args.model_name:
        cmd += ["--model-name", args.model_name]
    print(f"\n─── {strategy} " + "─" * (52 - len(strategy)))
    return subprocess.run(cmd, check=False).returncode == 0


def _read_metrics(path: Path) -> dict[str, float] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    block = payload.get("aggregate_summary")
    if not isinstance(block, dict):
        return None
    metrics = {m: float(block[m]) for m in METRICS if m in block}
    metrics["windows"] = float(payload.get("total_windows", 0))
    return metrics or None


def main() -> int:
    args = parse_args()
    baselines = _selected(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    if not any(os.environ.get(v) for v in LLM_ENV_VARS):
        missing.append(
            "  an LLM provider: set one of "
            + " / ".join(LLM_ENV_VARS[:-1])
            + ",\n    or OPENAI_BASE_URL for a local OpenAI-compatible endpoint"
        )
    if not os.environ.get("VOYAGE_API_KEY"):
        missing.append("  VOYAGE_API_KEY: matching is embedding-only")
    if missing:
        # Fail here rather than launching runs that cannot produce a number.
        # Every baseline needs both, so one missing key means an empty table.
        print(
            "Cannot run: every baseline needs both of these.\n" + "\n".join(missing),
            file=sys.stderr,
        )
        return 2

    print("Shared settings (identical for every baseline, so the rows compare):")
    print(f"  corpus            {args.input_dir}")
    print(f"  window            {args.start_month} .. {args.end_month}")
    min_cutoff = args.min_cutoff_month or "(none -- every eligible month scored)"
    cand_limit = args.candidate_limit or "(none -- exhaustive matching)"
    print(f"  min_cutoff_month  {min_cutoff}")
    print(f"  horizon_months    {args.horizon_months}")
    print(f"  candidate_limit   {cand_limit}")
    print(f"  top_k             {args.top_k}")
    print(f"  min_train_papers  {args.min_train_papers}")
    print(f"  baselines         {', '.join(baselines)}")

    results: dict[str, dict[str, float] | None] = {}
    for strategy in baselines:
        out_path = out_dir / f"{strategy}.json"
        # Remove first: otherwise a crashed run silently reports the numbers a
        # previous run left in the same file.
        out_path.unlink(missing_ok=True)
        _run_one(strategy, args, out_path)
        # Read regardless of exit code. `benchmark` exits 1 when it scores zero
        # windows, which is "ran but the corpus was too thin" -- a different
        # problem from a traceback, and the JSON is written either way.
        results[strategy] = _read_metrics(out_path)

    print(f"\n{'=' * 78}\nBaseline comparison\n{'=' * 78}")
    cols = [m.replace("avg_", "") for m in METRICS]
    width = max(len(c) for c in cols) + 2
    header = f"{'strategy':<22}{'windows':>9}" + "".join(f"{c:>{width}}" for c in cols)
    print(header)
    print("-" * len(header))
    for strategy in baselines:
        metrics = results[strategy]
        if metrics is None:
            print(f"{strategy:<22}{'FAILED':>9}   (run errored -- see output above)")
            continue
        windows = int(metrics.get("windows", 0))
        if windows == 0:
            # Never print a row of zeros here: it reads as "scored 0.0" when it
            # actually means the run produced nothing to score.
            print(f"{strategy:<22}{0:>9}   NOT SCORED -- no windows produced")
            continue
        row = "".join(f"{metrics.get(m, float('nan')):>{width}.4f}" for m in METRICS)
        print(f"{strategy:<22}{windows:>9}{row}")
    print(f"\nPer-run JSON in {out_dir}/")

    # Keep these apart: a run that errored and a run that scored nothing have
    # different causes, and collapsing them sent people to widen the corpus when
    # the real problem was an exception.
    errored = [s for s, m in results.items() if m is None]
    empty = [s for s, m in results.items() if m is not None and not int(m["windows"])]
    if errored:
        print(
            f"\n{len(errored)} baseline(s) exited non-zero: {', '.join(errored)}\n"
            "The traceback is in this run's output above -- read that first; the "
            "corpus settings are not the cause."
        )
    if empty:
        print(
            f"\n{len(empty)} baseline(s) ran but scored no windows: "
            f"{', '.join(empty)}\n"
            "The corpus is too thin for these settings: widen "
            "`fetch --lookback-days`, or lower --min-train-papers."
        )
    return 1 if (errored or empty) else 0


if __name__ == "__main__":
    raise SystemExit(main())
