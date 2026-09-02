#!/usr/bin/env python3
"""Assemble the main results table from judged artifacts, at two match thresholds.

The two hit@k columns are NOT two judge runs. ``S>=2`` is the judge's own
``is_match`` (``MATCH_PM_THRESHOLD=5`` and ``MATCH_S_THRESHOLD=2`` in
``idea_forecast_bench/judge/config.py``). ``S>=3`` re-applies
``problem + method >= 5 AND specificity >= 3`` to the raw per-dimension scores
already stored in ``per_prediction`` -- it is a recount, not a re-judge, so it
costs nothing and cannot drift from the ``S>=2`` column. To move the threshold
again, change ``_strict_match`` below; to change what a dimension means, you
have to re-run the judge.

Why the strict column exists: over 90% of matches at ``S>=2`` sit exactly on
the threshold, and the rubric reads ``S=1`` as "generic enough to loosely fit",
so the loose column rewards breadth. Across all three backbones,
``topic_trend`` and ``predictor_llm`` swap rank between the two columns while
the backbone ordering holds under both. Report the two facts separately.

Usage::

    python -m idea_forecast_bench main-table \\
        --source "gpt-4.1=output/judged/gpt41.*.judged.json" \\
        --source "Qwen2.5-7B=output/judged/qwen7b.*.judged.json" \\
        --source "MDF-Qwen2.5-7B=output/judged/mdf.*.judged.json"

Each ``--source`` is ``<backbone label>=<glob>`` over ``judge-eval`` outputs.
Sources are printed in the order given.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path
from typing import Any

STRATEGY_ORDER = (
    "topic_trend",
    "predictor_llm",
    "summary_prompting",
    "retrieval_prompting",
    "memory_prompting",
    "forecaster",
)

# Full 52-topic sweep: 52 topics x 12 cutoffs. Override on the command line for
# a partial run, but never report a row that fails the check.
DEFAULT_EXPECTED_TOPICS = 52
DEFAULT_EXPECTED_WINDOWS = 624

STRICT_PM_THRESHOLD = 5
STRICT_S_THRESHOLD = 3


def _parse_source(spec: str) -> tuple[str, str]:
    label, sep, pattern = spec.partition("=")
    if not sep or not label.strip() or not pattern.strip():
        raise argparse.ArgumentTypeError(
            f"--source must be '<backbone label>=<glob>', got {spec!r}"
        )
    return label.strip(), pattern.strip()


def _strategy_of(path: str, artifact: dict[str, Any]) -> str:
    """Judged artifacts do not carry ``strategy`` at top level (the judge
    writes its own schema), so fall back to the filename. MDF's files are named
    ``mdf.*``, which contains none of the strategy ids -- name it explicitly
    rather than letting it drop out of the printed table while still being
    counted in the row total."""
    strategy = artifact.get("strategy")
    if strategy:
        return str(strategy)
    for known in STRATEGY_ORDER:
        if known in path:
            return known
    name = Path(path).name
    if name.startswith("mdf.") or "/mdf" in path:
        return "forecaster"
    return "?"


def _strict_match(prediction: dict[str, Any]) -> bool:
    problem = prediction.get("problem_score") or 0
    method = prediction.get("method_score") or 0
    specificity = prediction.get("specificity_score") or 0
    return problem + method >= STRICT_PM_THRESHOLD and (
        specificity >= STRICT_S_THRESHOLD
    )


def _new_row() -> dict[str, int]:
    return {"windows": 0, "hit2": 0, "hit3": 0, "short": 0, "prior_fallback": 0}


def _count_window(row: dict[str, int], window: dict[str, Any]) -> None:
    per_prediction = window.get("per_prediction") or []
    row["windows"] += 1
    if any(p.get("is_match") for p in per_prediction):
        row["hit2"] += 1
    if any(_strict_match(p) for p in per_prediction):
        row["hit3"] += 1
    if len(per_prediction) < 5:
        row["short"] += 1
    for prediction in per_prediction:
        events = (prediction.get("metadata") or {}).get("fallback_events") or []
        row["prior_fallback"] += sum(1 for e in events if e.get("phase") == "prior")


def collect(
    sources: list[tuple[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, int]], dict[tuple[str, str], set[Any]]]:
    rows: dict[tuple[str, str], dict[str, int]] = collections.defaultdict(_new_row)
    coverage: dict[tuple[str, str], set[Any]] = collections.defaultdict(set)
    for backbone, pattern in sources:
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"  !! {backbone}: no files match {pattern}", file=sys.stderr)
        for path in paths:
            try:
                with open(path, encoding="utf-8") as handle:
                    artifact = json.load(handle)
            except (OSError, ValueError) as exc:
                print(f"  !! skipping {path}: {exc}", file=sys.stderr)
                continue
            key = (backbone, _strategy_of(path, artifact))
            for topic_id, topic_result in artifact.get("topic_results", {}).items():
                backtest = topic_result.get("backtest")
                if not backtest:
                    continue
                for window in backtest.get("windows", []):
                    _count_window(rows[key], window)
                    coverage[key].add((topic_id, window.get("cutoff_month")))
    return rows, coverage


def _print_table(
    sources: list[tuple[str, str]],
    rows: dict[tuple[str, str], dict[str, int]],
) -> None:
    header = (
        f"{'backbone':<17}{'strategy':<21}{'windows':>8}"
        f"{'hit@k S>=2':>12}{'hit@k S>=3':>12}{'short':>7}{'prior_fb':>10}"
    )
    print(header)
    print("-" * len(header))
    for backbone, _ in sources:
        known = [s for s in STRATEGY_ORDER if (backbone, s) in rows]
        extra = [s for (b, s) in rows if b == backbone and s not in STRATEGY_ORDER]
        for strategy in known + extra:
            row = rows[(backbone, strategy)]
            if row["windows"] == 0:
                continue
            n = row["windows"]
            print(
                f"{backbone:<17}{strategy:<21}{n:>8}"
                f"{row['hit2'] / n:>12.4f}{row['hit3'] / n:>12.4f}"
                f"{row['short']:>7}{row['prior_fallback']:>10}"
            )


def _check_coverage(
    coverage: dict[tuple[str, str], set[Any]],
    expected_topics: int,
    expected_windows: int,
) -> bool:
    """Both checks are needed and neither implies the other: counting unique
    topics alone passes a row whose topic ran only three of its twelve cutoffs,
    and counting unique (topic, cutoff) pairs alone is filled in by duplicates
    -- two shard sets built from different partitions once summed to exactly
    624 windows while covering 40 of 52 topics."""
    complete = True
    for (backbone, strategy), pairs in sorted(coverage.items()):
        topics = {t for t, _ in pairs}
        if len(topics) != expected_topics or len(pairs) != expected_windows:
            complete = False
            print(
                f"  !! {backbone}/{strategy}: {len(topics)}/{expected_topics} "
                f"topics, {len(pairs)}/{expected_windows} windows -- "
                "incomplete, do not report"
            )
    return complete


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        action="append",
        type=_parse_source,
        required=True,
        metavar="LABEL=GLOB",
        help="Backbone label and a glob over judge-eval outputs. Repeatable.",
    )
    parser.add_argument("--expected-topics", type=int, default=DEFAULT_EXPECTED_TOPICS)
    parser.add_argument(
        "--expected-windows", type=int, default=DEFAULT_EXPECTED_WINDOWS
    )
    args = parser.parse_args()

    rows, coverage = collect(args.source)
    _print_table(args.source, rows)
    total_windows = sum(r["windows"] for r in rows.values())
    print(f"\n{len(rows)} rows / {total_windows} windows")
    complete = _check_coverage(coverage, args.expected_topics, args.expected_windows)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
