"""Fetch an arXiv corpus in the layout the benchmark expects.

Without this the benchmark has no input: `idea-forecast-bench benchmark` defaults
to `--input-dir data/csml/raw_markdown`, and nothing in the repo put anything
there. `idea_forecast_bench.ingest` could already do the work but was reachable
only from Python, so reproducing a run started with "write your own script".

Papers land as ``<out-dir>/<YYYY-MM>/<paper_id>.md``. The loader globs for
``*.md`` recursively, so this layout is directly consumable:

    idea-forecast-bench fetch --out-dir data/csml/raw_markdown
    idea-forecast-bench benchmark --input-dir data/csml/raw_markdown

Usage:
    idea-forecast-bench fetch                              # cs.LG, 12 months back
    idea-forecast-bench fetch --query "cat:cs.CL" --max-results 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from idea_forecast_bench.ingest import ingest_latest_arxiv_papers

DEFAULT_OUT_DIR = "data/csml/raw_markdown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download an arXiv corpus for the benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Where to write the corpus; pass this same path to `benchmark --input-dir`.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="cat:cs.LG",
        help="arXiv search query, e.g. 'cat:cs.CL' or 'cat:cs.LG OR cat:cs.AI'.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=2000,
        help="Upper bound on papers to fetch.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=540,
        help="How far back to reach. A backtest needs several months of history "
        "before its first cutoff, so keep this comfortably larger than the "
        "window you intend to score.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)

    print(f"Fetching up to {args.max_results} papers for {args.query!r} ...")
    result = ingest_latest_arxiv_papers(
        data_dir=out_dir,
        query=args.query,
        max_results=args.max_results,
        lookback_days=args.lookback_days,
    )

    paths = sorted(out_dir.rglob("*.md"))
    months = sorted({p.parent.name for p in paths})
    print(
        f"  fetched {result['fetched_count']}, "
        f"new {result['ingested_count']}, "
        f"already present {result['skipped_existing_count']}"
    )
    print(f"  corpus now: {len(paths)} papers across {len(months)} months")
    if months:
        print(f"  range: {months[0]} .. {months[-1]}")
        print(f"\nNext: idea-forecast-bench benchmark --input-dir {out_dir}")
    else:
        print("\nNo papers written. Widen --lookback-days or loosen --query.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
