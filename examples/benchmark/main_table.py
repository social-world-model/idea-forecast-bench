#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path
from typing import Any

from idea_forecast_bench.judge.config import MATCH_PM_THRESHOLD

DESCRIPTION = """\
Assemble the main results table from judge-eval outputs, at two match thresholds.

S>=2 is the judge's own is_match. S>=3 re-applies problem + method >= 5 AND
specificity >= 3 to the stored per-dimension scores: a recount, not a re-judge.
Each --source is '<backbone label>=<glob>' over judge-eval outputs; sources are
printed in the order given, and a row that does not cover every expected
topic and window exactly once is flagged rather than reported.
"""

STRATEGY_ORDER = (
    "topic_trend",
    "predictor_llm",
    "summary_prompting",
    "retrieval_prompting",
    "memory_prompting",
    "forecaster",
    "combinatorial",
    "combinatorial_frequency",
    "combinatorial_independent",
    "combinatorial_random",
)

# Full 52-topic sweep: 52 topics x 12 cutoffs.
DEFAULT_EXPECTED_TOPICS = 52
DEFAULT_EXPECTED_WINDOWS = 624

STRICT_S_THRESHOLD = 3

Row = dict[str, int]
Key = tuple[str, str]


def _parse_source(spec: str) -> tuple[str, str]:
    label, sep, pattern = spec.partition("=")
    if not sep or not label.strip() or not pattern.strip():
        raise argparse.ArgumentTypeError(
            f"--source must be '<backbone label>=<glob>', got {spec!r}"
        )
    return label.strip(), pattern.strip()


def _strategy_of(path: str, artifact: dict[str, Any]) -> str:
    """The judge copies the generator's `strategy` into its artifact; older
    artifacts lack it, so fall back to the file name."""
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
    return problem + method >= MATCH_PM_THRESHOLD and (
        specificity >= STRICT_S_THRESHOLD
    )


def _empty_row() -> Row:
    return {"windows": 0, "hit2": 0, "hit3": 0, "short": 0, "prior_fallback": 0}


def _window_counts(window: dict[str, Any]) -> Row:
    per_prediction = window.get("per_prediction") or []
    prior_fallback = sum(
        1
        for p in per_prediction
        for e in (p.get("metadata") or {}).get("fallback_events") or []
        if e.get("phase") == "prior"
    )
    return {
        "windows": 1,
        "hit2": int(any(p.get("is_match") for p in per_prediction)),
        "hit3": int(any(_strict_match(p) for p in per_prediction)),
        "short": int(len(per_prediction) < 5),
        "prior_fallback": prior_fallback,
    }


def _add(a: Row, b: Row) -> Row:
    return {k: a[k] + b[k] for k in a}


def collect(
    sources: list[tuple[str, str]],
) -> tuple[dict[Key, Row], dict[Key, set[Any]]]:
    rows: dict[Key, Row] = collections.defaultdict(_empty_row)
    coverage: dict[Key, set[Any]] = collections.defaultdict(set)
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
                    rows[key] = _add(rows[key], _window_counts(window))
                    coverage[key].add((topic_id, window.get("cutoff_month")))
    return rows, coverage


def _print_table(sources: list[tuple[str, str]], rows: dict[Key, Row]) -> None:
    header = (
        f"{'backbone':<17}{'strategy':<27}{'windows':>8}"
        f"{'hit@k S>=2':>12}{'hit@k S>=3':>12}{'short':>7}{'prior_fb':>10}"
    )
    print(header)
    print("-" * len(header))
    for backbone, _ in sources:
        known = [s for s in STRATEGY_ORDER if (backbone, s) in rows]
        extra = [s for (b, s) in rows if b == backbone and s not in STRATEGY_ORDER]
        for strategy in known + extra:
            row = rows[(backbone, strategy)]
            n = row["windows"]
            if n == 0:
                continue
            print(
                f"{backbone:<17}{strategy:<27}{n:>8}"
                f"{row['hit2'] / n:>12.4f}{row['hit3'] / n:>12.4f}"
                f"{row['short']:>7}{row['prior_fallback']:>10}"
            )


def _check_coverage(
    rows: dict[Key, Row],
    coverage: dict[Key, set[Any]],
    expected_topics: int,
    expected_windows: int,
) -> bool:
    """Three checks, none implied by the others: unique topics (a topic that
    ran only some of its cutoffs), unique (topic, cutoff) pairs (a missing
    topic backfilled by duplicates of another), and windows counted equal to
    unique pairs (duplicate artifacts matched by one glob inflating the
    denominator while coverage still looks complete)."""
    complete = True
    for key, pairs in sorted(coverage.items()):
        backbone, strategy = key
        topics = {t for t, _ in pairs}
        counted = rows[key]["windows"]
        problems = []
        if len(topics) != expected_topics:
            problems.append(f"{len(topics)}/{expected_topics} topics")
        if len(pairs) != expected_windows:
            problems.append(f"{len(pairs)}/{expected_windows} unique windows")
        if counted != len(pairs):
            problems.append(f"{counted} windows counted for {len(pairs)} unique")
        if problems:
            complete = False
            print(
                f"  !! {backbone}/{strategy}: {', '.join(problems)} -- "
                "incomplete or duplicated, do not report"
            )
    return complete


def main() -> int:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter
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
    complete = _check_coverage(
        rows, coverage, args.expected_topics, args.expected_windows
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
