#!/usr/bin/env python3
"""Split the topic taxonomy into shards balanced by paper count.

One ``benchmark`` process is one Python interpreter, and the per-window work
outside the API call (the lexical prefilter, corpus filtering) is GIL-bound, so
``--workers`` only parallelises the API wait. The way to scale a sweep is to run
several processes over disjoint ``--topics`` sets. Within a shard the cutoffs run
serially, so the shards must carry similar numbers of papers or every other
process ends up waiting on the heaviest one.

Prints one line per shard, each a comma-separated list of topic ids, ready for
``benchmark --topics``. ``scripts/benchmark/run_sharded_backtest.sh`` calls this.

Usage::

    python examples/benchmark/split_topics.py --input-dir data/csml/raw_markdown --shards 4
"""

from __future__ import annotations

import argparse
import json
import sys

from idea_forecast_bench.paper_cache import load_papers_and_topics


def balanced_split(counts: dict[str, int], shards: int) -> list[list[str]]:
    """Greedy longest-processing-time assignment: heaviest topic first, each
    to the currently lightest shard. Deterministic for a fixed corpus."""
    if shards < 1:
        raise ValueError("shards must be >= 1")
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    buckets: list[list[str]] = [[] for _ in range(shards)]
    loads = [0] * shards
    for topic_id, n in ordered:
        i = loads.index(min(loads))
        buckets[i] = [*buckets[i], topic_id]
        loads[i] += n
    return buckets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-dir", default="data/csml/raw_markdown")
    parser.add_argument("--shards", type=int, default=4)
    # Same defaults as run_domain_backtest.py: the split must see the same
    # papers the backtest will, or the balance is off.
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2025-06")
    parser.add_argument(
        "--json", action="store_true", help="Emit a JSON list of lists instead."
    )
    args = parser.parse_args()

    _papers, _topics, grouped = load_papers_and_topics(
        args.input_dir, args.start_month, args.end_month, verbose=False
    )
    counts = {topic_id: len(papers) for topic_id, papers in grouped.items()}
    if not counts:
        print("No topics with papers in that window.", file=sys.stderr)
        return 1

    buckets = balanced_split(counts, args.shards)
    if args.json:
        print(json.dumps(buckets))
    else:
        for bucket in buckets:
            print(",".join(bucket))
    for i, bucket in enumerate(buckets):
        load = sum(counts[t] for t in bucket)
        print(f"shard {i}: {len(bucket)} topics, {load} papers", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
