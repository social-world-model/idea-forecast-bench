#!/usr/bin/env python3
"""Data leakage check: compare hit@k across time-since-cutoff buckets.

If a model has seen future data, papers published shortly after the cutoff
should have systematically higher hit rates than papers published much later.

Usage:
    python examples/analysis_leakage.py \\
        --input memory_prompting_llmjudge.json predictor_llm_llmjudge.json \\
        --output leakage_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _month_diff(a: str, b: str) -> int:
    """Return (b - a) in months. Both strings are 'YYYY-MM'."""
    ay, am = int(a[:4]), int(a[5:7])
    by, bm = int(b[:4]), int(b[5:7])
    return (by - ay) * 12 + (bm - am)


def _bucket(gap_months: int) -> str:
    if gap_months <= 3:
        return "1-3mo"
    elif gap_months <= 6:
        return "4-6mo"
    else:
        return "7+mo"


def _analyze(data: dict, label: str) -> dict:
    bucket_hits: dict[str, list[float]] = defaultdict(list)

    for topic_id, tr in data.get("topic_results", {}).items():
        bt = tr.get("backtest")
        if not bt:
            continue
        for w in bt.get("windows", []):
            cutoff = w.get("cutoff_month", "")
            future_end = w.get("future_end_month", "")
            if not cutoff or not future_end:
                continue
            gap = _month_diff(cutoff, future_end)
            bucket = _bucket(gap)
            hit = w["evaluation"].get("hit_at_k", 0.0)
            bucket_hits[bucket].append(hit)

    result: dict = {"label": label, "buckets": {}}
    for b in ["1-3mo", "4-6mo", "7+mo"]:
        vals = bucket_hits.get(b, [])
        result["buckets"][b] = {
            "n_windows": len(vals),
            "hit_at_k_mean": round(sum(vals) / len(vals), 4) if vals else None,
        }

    # Simple leakage indicator: is hit rate for 1-3mo significantly higher?
    early = bucket_hits.get("1-3mo", [])
    late  = bucket_hits.get("7+mo",  [])
    if early and late:
        try:
            from scipy.stats import mannwhitneyu
            stat, pval = mannwhitneyu(early, late, alternative="greater")
            result["leakage_test"] = {
                "test": "Mann-Whitney U (early > late)",
                "statistic": round(float(stat), 4),
                "p_value": round(float(pval), 4),
                "significant_at_05": float(pval) < 0.05,
            }
        except ImportError:
            early_mean = sum(early) / len(early)
            late_mean  = sum(late)  / len(late)
            result["leakage_test"] = {
                "test": "mean comparison (scipy not available)",
                "early_mean": round(early_mean, 4),
                "late_mean":  round(late_mean, 4),
                "delta":      round(early_mean - late_mean, 4),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  nargs="+", required=True,
                        help="llmjudge output JSON files")
    parser.add_argument("--output", default="leakage_report.json")
    args = parser.parse_args()

    results = []
    for path in args.input:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        label = Path(path).stem
        r = _analyze(data, label)
        results.append(r)
        print(f"\n=== {label} ===")
        for b, v in r["buckets"].items():
            print(f"  {b}: n={v['n_windows']}  hit@k={v['hit_at_k_mean']}")
        if "leakage_test" in r:
            lt = r["leakage_test"]
            print(f"  Leakage test: {lt}")

    Path(args.output).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
