#!/usr/bin/env python3
"""Recompute hit/recall/precision/mrr metrics from a voyage JSON at a new threshold.

No API calls — just re-thresholds the stored best_score values.

Usage:
    python examples/rethreshold_voyage.py \
        --input  logs/baselines/keyword_trend_voyage.json \
        --output logs/baselines/keyword_trend_voyage_t60.json \
        --threshold 0.60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _recompute_window(window: dict, threshold: float, top_k: int) -> dict:
    per_pred = window["evaluation"]["per_prediction"]

    # Re-threshold
    for p in per_pred:
        p["is_match"] = p["best_score"] >= threshold

    hits = [p for p in per_pred if p["is_match"]]
    n = len(per_pred)

    hit_at_k = 1.0 if hits else 0.0
    precision = len(hits) / n if n else 0.0
    recall = min(1.0, len(hits) / top_k) if top_k else 0.0

    # MRR: rank of first hit
    mrr = 0.0
    for i, p in enumerate(per_pred, start=1):
        if p["is_match"]:
            mrr = 1.0 / i
            break

    return {
        **window,
        "evaluation": {
            "hit_at_k": hit_at_k,
            "recall_at_k": recall,
            "precision_at_k": precision,
            "mrr": mrr,
            "per_prediction": per_pred,
        },
    }


def _summarize(windows: list[dict]) -> dict:
    if not windows:
        return {"windows": 0, "avg_hit_at_k": 0.0, "avg_recall_at_k": 0.0,
                "avg_precision_at_k": 0.0, "avg_mrr": 0.0}
    n = len(windows)
    return {
        "windows": n,
        "avg_hit_at_k":       round(sum(w["evaluation"]["hit_at_k"]       for w in windows) / n, 4),
        "avg_recall_at_k":    round(sum(w["evaluation"]["recall_at_k"]    for w in windows) / n, 4),
        "avg_precision_at_k": round(sum(w["evaluation"]["precision_at_k"] for w in windows) / n, 4),
        "avg_mrr":            round(sum(w["evaluation"]["mrr"]             for w in windows) / n, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    top_k = data.get("config", {}).get("top_k", 5)
    new_topic_results = {}
    total_windows = 0

    for tid, tval in data["topic_results"].items():
        bt = tval.get("backtest")
        if not bt:
            new_topic_results[tid] = tval
            continue

        new_windows = [_recompute_window(w, args.threshold, top_k) for w in bt["windows"]]
        summary = _summarize(new_windows)
        total_windows += summary["windows"]

        new_topic_results[tid] = {
            **tval,
            "backtest": {**bt, "windows": new_windows, "summary": summary},
        }

    # Weighted aggregate
    metric_keys = ["avg_hit_at_k", "avg_recall_at_k", "avg_precision_at_k", "avg_mrr"]
    num = dict.fromkeys(metric_keys, 0.0)
    den = 0
    for tval in new_topic_results.values():
        bt = tval.get("backtest")
        if not bt:
            continue
        w = bt["summary"]["windows"]
        if w > 0:
            for k in metric_keys:
                num[k] += bt["summary"][k] * w
            den += w
    aggregate = {k: round(num[k] / den, 4) if den else 0.0 for k in metric_keys}

    output = {
        **data,
        "threshold": args.threshold,
        "total_windows": total_windows,
        "aggregate_summary": aggregate,
        "topic_results": new_topic_results,
    }

    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Saved to {args.output}")
    print(f"Threshold: {args.threshold}  |  {total_windows} windows")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
