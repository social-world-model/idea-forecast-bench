"""Run every baseline over one corpus and print a comparison table.

Reproducing the baseline table used to mean invoking `benchmark` once per
strategy, by hand, and remembering to keep every window/scoring flag identical
between runs -- if one run drifted, the numbers were no longer comparable.

    live-idea-bench baselines --input-dir data/csml/raw_markdown

Every strategy sees exactly the same windows and the same matcher; the shared
settings are printed once so a run is self-documenting.

Usage:
    live-idea-bench baselines                          # keyless: heuristic matcher
    live-idea-bench baselines --only keyword_trend,topic_trend
    live-idea-bench baselines --similarity-engine embedding   # needs VOYAGE_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

#: Baselines that need no model API key -- the default set, so a fresh clone
#: can produce a table offline.
OFFLINE_BASELINES = ("keyword_trend", "topic_trend")
#: Baselines that call an LLM, and therefore need a provider key.
LLM_BASELINES = (
    "predictor_llm",
    "summary_prompting",
    "retrieval_prompting",
    "memory_prompting",
)
ALL_BASELINES = OFFLINE_BASELINES + LLM_BASELINES

METRICS = (
    "avg_hit_at_k",
    "avg_recall_at_k",
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
        help="Comma-separated subset, e.g. 'keyword_trend,summary_prompting'. "
        f"Offline: {','.join(OFFLINE_BASELINES)}. LLM: {','.join(LLM_BASELINES)}.",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Also run the LLM baselines (needs a provider API key). Off by "
        "default so the command works with no credentials.",
    )
    parser.add_argument("--start-month", type=str, default="2024-01")
    parser.add_argument("--end-month", type=str, default="2025-06")
    parser.add_argument("--horizon-months", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-train-papers", type=int, default=5)
    parser.add_argument(
        "--similarity-engine",
        type=str,
        default="heuristic",
        help="heuristic (no key) | embedding (VOYAGE_API_KEY) | llm (judge key).",
    )
    parser.add_argument("--model-name", type=str, default=None)
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
    return list(ALL_BASELINES if args.include_llm else OFFLINE_BASELINES)


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
        "--similarity-engine",
        args.similarity_engine,
        "--workers",
        str(args.workers),
        "--output",
        str(out_path),
    ]
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

    needs_key = [b for b in baselines if b in LLM_BASELINES]
    if needs_key and not any(
        os.environ.get(v)
        for v in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")
    ):
        print(
            f"WARNING: {', '.join(needs_key)} call an LLM, but no provider key is "
            "set (OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY).\n"
            "         Those rows will come back unscored. Continuing in case a "
            "local endpoint is configured.\n"
        )

    print("Shared settings (identical for every baseline, so the rows compare):")
    print(f"  corpus            {args.input_dir}")
    print(f"  window            {args.start_month} .. {args.end_month}")
    print(f"  horizon_months    {args.horizon_months}")
    print(f"  top_k             {args.top_k}")
    print(f"  min_train_papers  {args.min_train_papers}")
    print(f"  similarity_engine {args.similarity_engine}")
    print(f"  baselines         {', '.join(baselines)}")

    results: dict[str, dict[str, float] | None] = {}
    for strategy in baselines:
        out_path = out_dir / f"{strategy}.json"
        results[strategy] = (
            _read_metrics(out_path) if _run_one(strategy, args, out_path) else None
        )

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

    unscored = [
        s for s, m in results.items() if m is None or int(m.get("windows", 0)) == 0
    ]
    if unscored:
        print(
            f"\n{len(unscored)} baseline(s) produced no scored windows: "
            f"{', '.join(unscored)}\n"
            "Either the corpus is too thin (widen `fetch --lookback-days`, or "
            "lower --min-train-papers) or a required API key is missing."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
