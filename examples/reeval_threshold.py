"""Re-compute evaluation metrics from a saved reeval JSON at any threshold.

The reeval JSON (produced by reeval_from_json.py) stores per_prediction_scores
with best_score already computed.  This script re-derives hit@k, MRR, precision,
recall at arbitrary thresholds without any API calls.

Usage::

    python examples/reeval_threshold.py \\
        --input /tmp/reeval_t06.json \\
        --thresholds 0.4 0.45 0.5 0.55 0.6

Output: table printed to stdout + optional JSON saved with --output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _recompute_window(per_prediction_scores: list[dict], future_papers: int, threshold: float) -> dict:
    """Re-derive metrics from saved per_prediction_scores at a new threshold."""
    k = len(per_prediction_scores)
    matched_ranks: list[int] = []
    used_paper_ids: set[str] = set()

    for ps in per_prediction_scores:
        rank = ps["rank"]
        best_score = ps["best_score"]
        best_paper_id = ps.get("best_paper_id")
        is_match = (
            best_score >= threshold
            and best_paper_id is not None
            and best_paper_id not in used_paper_ids
        )
        if is_match:
            matched_ranks.append(rank)
            used_paper_ids.add(best_paper_id)

    hit_at_k = 1.0 if matched_ranks else 0.0
    mrr = 1.0 / matched_ranks[0] if matched_ranks else 0.0
    precision = len(matched_ranks) / k if k else 0.0
    recall = len(matched_ranks) / future_papers if future_papers else 0.0

    return {
        "hit_at_k": round(hit_at_k, 4),
        "mrr": round(mrr, 4),
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "matched": len(matched_ranks),
    }


def recompute(data: dict, threshold: float) -> dict:
    """Re-compute all topics and aggregate at given threshold."""
    topic_summaries: dict[str, dict] = {}
    total_windows = 0

    for topic_id, tr in data["topic_results"].items():
        bt = tr.get("backtest")
        if not bt or not bt.get("windows"):
            topic_summaries[topic_id] = {
                "topic_name": tr.get("topic_name", topic_id),
                "windows": 0,
            }
            continue

        windows_metrics = []
        for w in bt["windows"]:
            pps = w["evaluation"].get("per_prediction_scores", [])
            future_papers = w.get("future_papers", 1)
            if not pps:
                continue
            m = _recompute_window(pps, future_papers, threshold)
            # carry over novelty/diversity (threshold-independent)
            m["novelty"] = w["evaluation"].get("novelty", 0.0)
            m["diversity"] = w["evaluation"].get("diversity", 0.0)
            windows_metrics.append(m)

        if not windows_metrics:
            topic_summaries[topic_id] = {
                "topic_name": tr.get("topic_name", topic_id),
                "windows": 0,
            }
            continue

        def avg(key: str) -> float:
            vals = [m[key] for m in windows_metrics]
            return round(sum(vals) / len(vals), 4)

        n = len(windows_metrics)
        total_windows += n
        topic_summaries[topic_id] = {
            "topic_name": tr.get("topic_name", topic_id),
            "windows": n,
            "avg_hit_at_k": avg("hit_at_k"),
            "avg_mrr": avg("mrr"),
            "avg_precision_at_k": avg("precision_at_k"),
            "avg_recall_at_k": avg("recall_at_k"),
            "avg_novelty": avg("novelty"),
            "avg_diversity": avg("diversity"),
        }

    # Weighted aggregate
    metrics = ["avg_hit_at_k", "avg_mrr", "avg_precision_at_k", "avg_recall_at_k", "avg_novelty", "avg_diversity"]
    aggregate: dict[str, float] = {}
    for metric in metrics:
        num, den = 0.0, 0
        for s in topic_summaries.values():
            w = s.get("windows", 0)
            if w > 0 and metric in s:
                num += s[metric] * w
                den += w
        aggregate[metric] = round(num / den, 4) if den else 0.0

    return {
        "threshold": threshold,
        "total_windows": total_windows,
        "aggregate": aggregate,
        "topics": topic_summaries,
    }


def print_table(results: list[dict]) -> None:
    topics_all = [tid for tid in results[0]["topics"]]
    thresholds = [r["threshold"] for r in results]

    # ── per-topic hit@k table ──────────────────────────────────────────────
    col_w = 12
    topic_col = 22
    print("\n" + "=" * 70)
    print("hit@k by topic")
    print("=" * 70)
    header = f"{'Topic':<{topic_col}}" + "".join(f"  t={t:.2f}".rjust(col_w) for t in thresholds)
    print(header)
    print("-" * len(header))
    for tid in topics_all:
        name = results[0]["topics"][tid].get("topic_name", tid)[:topic_col - 1]
        row = f"{name:<{topic_col}}"
        for r in results:
            s = r["topics"][tid]
            val = f"{s['avg_hit_at_k']:.4f}" if s.get("windows", 0) > 0 else "  N/A"
            row += val.rjust(col_w)
        print(row)
    print("-" * len(header))
    agg_row = f"{'Aggregate':<{topic_col}}"
    for r in results:
        agg_row += f"{r['aggregate']['avg_hit_at_k']:.4f}".rjust(col_w)
    print(agg_row)

    # ── per-topic MRR table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MRR by topic")
    print("=" * 70)
    print(header)
    print("-" * len(header))
    for tid in topics_all:
        name = results[0]["topics"][tid].get("topic_name", tid)[:topic_col - 1]
        row = f"{name:<{topic_col}}"
        for r in results:
            s = r["topics"][tid]
            val = f"{s['avg_mrr']:.4f}" if s.get("windows", 0) > 0 else "  N/A"
            row += val.rjust(col_w)
        print(row)
    print("-" * len(header))
    agg_row = f"{'Aggregate':<{topic_col}}"
    for r in results:
        agg_row += f"{r['aggregate']['avg_mrr']:.4f}".rjust(col_w)
    print(agg_row)

    # ── aggregate summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Aggregate summary")
    print("=" * 70)
    metrics = ["avg_hit_at_k", "avg_mrr", "avg_precision_at_k", "avg_recall_at_k", "avg_novelty", "avg_diversity"]
    metric_col = 22
    print(f"{'Metric':<{metric_col}}" + "".join(f"  t={t:.2f}".rjust(col_w) for t in thresholds))
    print("-" * (metric_col + col_w * len(thresholds)))
    for m in metrics:
        row = f"{m:<{metric_col}}"
        for r in results:
            row += f"{r['aggregate'][m]:.4f}".rjust(col_w)
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="reeval JSON from reeval_from_json.py")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.4, 0.45, 0.5, 0.55, 0.6])
    parser.add_argument("--output", default="", help="optional JSON output path")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = [recompute(data, t) for t in args.thresholds]

    print_table(results)

    if args.output:
        Path(args.output).write_text(
            json.dumps({"source": args.input, "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
