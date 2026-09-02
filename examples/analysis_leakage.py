#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# lead_time is a fraction of the horizon in [0, 1]. These thirds correspond,
# for a 3-month horizon, to papers published in roughly month 1 / 2 / 3.
_EARLY_MAX = 1.0 / 3.0
_MID_MAX = 2.0 / 3.0
_BUCKET_ORDER = ["early", "mid", "late"]


def _bucket(lead_time: float) -> str:
    if lead_time <= _EARLY_MAX:
        return "early"
    if lead_time <= _MID_MAX:
        return "mid"
    return "late"


def _analyze(data: dict, label: str) -> dict:
    # One observation per TOP-K prediction across all windows/topics: 1.0 if it
    # matched a paper in its lead-time bucket, 0.0 otherwise. Bucketing is by the
    # canonical per-match ``lead_time`` fraction, NOT by the (constant) horizon.
    bucket_hits: dict[str, list[float]] = defaultdict(list)
    saw_matches_key = False

    for _topic_id, tr in data.get("topic_results", {}).items():
        bt = tr.get("backtest")
        if not bt:
            continue
        for w in bt.get("windows", []):
            matches = w.get("matches")
            if matches is None:
                continue
            saw_matches_key = True
            for m in matches:
                lead_time = float(m.get("lead_time", 0.0))
                is_match = bool(m.get("is_match", False))
                bucket_hits[_bucket(lead_time)].append(1.0 if is_match else 0.0)

    if not saw_matches_key:
        raise SystemExit(
            f"[leakage] '{label}': no window carried a 'matches' list. This "
            "script reads the CANONICAL backtest schema (benchmark.py "
            "output with per-match lead_time), not llm_judge_eval output."
        )

    result: dict = {"label": label, "buckets": {}}
    for b in _BUCKET_ORDER:
        vals = bucket_hits.get(b, [])
        result["buckets"][b] = {
            "n_predictions": len(vals),
            "match_rate": round(sum(vals) / len(vals), 4) if vals else None,
        }

    # Leakage indicator: do early (near-cutoff) matches hit more than late ones?
    early = bucket_hits.get("early", [])
    late = bucket_hits.get("late", [])
    non_empty = [b for b in _BUCKET_ORDER if bucket_hits.get(b)]
    if len(non_empty) < 2:
        # Guard against the original silent no-op: a single populated bucket
        # cannot support a comparison, so say so loudly instead of pretending.
        result["leakage_test"] = {
            "test": "skipped",
            "reason": (
                f"only {len(non_empty)} lead-time bucket(s) populated "
                f"({non_empty}); need >=2 to compare. Check the horizon / data."
            ),
        }
        return result

    if early and late:
        try:
            from scipy.stats import mannwhitneyu

            stat, pval = mannwhitneyu(early, late, alternative="greater")
            result["leakage_test"] = {
                "test": "Mann-Whitney U (early match-rate > late)",
                "statistic": round(float(stat), 4),
                "p_value": round(float(pval), 4),
                "significant_at_05": float(pval) < 0.05,
            }
        except ImportError:
            early_mean = sum(early) / len(early)
            late_mean = sum(late) / len(late)
            result["leakage_test"] = {
                "test": "mean comparison (scipy not available)",
                "early_mean": round(early_mean, 4),
                "late_mean": round(late_mean, 4),
                "delta": round(early_mean - late_mean, 4),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="canonical benchmark.py output JSON files",
    )
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
            print(f"  {b}: n={v['n_predictions']}  match_rate={v['match_rate']}")
        if "leakage_test" in r:
            print(f"  Leakage test: {r['leakage_test']}")

    Path(args.output).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
