"""Run a domain-separated backtest from the command line.

This exercises the same per-topic backtest pipeline used by the backend
(strategy_store.run_backtest_sync) but without requiring Flask or a
persisted strategy JSON file.

Usage::

    python examples/run_domain_backtest.py \
        --input-dir data/arxiv_csml/raw_markdown \
        --strategy keyword_trend \
        --start-month 2024-01 --end-month 2025-06 \
        --output /tmp/domain_backtest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from live_idea_bench.backtest import BacktestConfig, backtest  # noqa: E402
from live_idea_bench.config import load_topics  # noqa: E402
from live_idea_bench.papers import load_papers_from_markdown  # noqa: E402
from live_idea_bench.strategy import create_strategy  # noqa: E402
from live_idea_bench.topics import classify_papers_by_topic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run domain-separated backtest across configured topics.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/arxiv_csml/raw_markdown",
        help="Directory with markdown papers.",
    )
    parser.add_argument("--strategy", type=str, default="keyword_trend")
    parser.add_argument("--recent-months", type=int, default=3)
    parser.add_argument("--min-keyword-freq", type=int, default=1)
    parser.add_argument("--model-name", type=str, help="Model override for predictor_llm.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--horizon-months", type=int, default=3)
    parser.add_argument("--min-train-papers", type=int, default=2)
    parser.add_argument("--start-month", type=str, default="2024-01")
    parser.add_argument("--end-month", type=str, default="2025-06")
    parser.add_argument("--similarity-config", type=str, default="similarity.yaml")
    parser.add_argument(
        "--similarity-engine",
        type=str,
        default=None,
        help="Override the similarity engine at runtime (e.g. heuristic, embedding, llm). "
             "Useful when you want to skip API calls and re-evaluate later with reeval_from_json.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/domain_backtest.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    # Apply runtime engine override: write a minimal temp YAML so the caller
    # never needs to touch similarity.yaml just to change the engine.
    _tmp_sim_cfg = None
    if args.similarity_engine:
        import tempfile
        _tmp_sim_cfg = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="sim_override_"
        )
        _tmp_sim_cfg.write(f"engine: {args.similarity_engine}\n")
        _tmp_sim_cfg.close()
        args.similarity_config = _tmp_sim_cfg.name

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    print(f"Loading papers from {input_dir} ...")
    papers = load_papers_from_markdown(
        input_dir,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    print(f"Loaded {len(papers)} papers ({args.start_month} to {args.end_month})")
    if not papers:
        print("No papers found. Exiting.")
        return 1

    topics = load_topics()
    print(f"\nTopics configured: {len(topics)}")
    grouped = classify_papers_by_topic(papers, topics)
    for topic_id, topic_papers in grouped.items():
        print(f"  {topic_id}: {len(topic_papers)} papers")

    strategy_obj = create_strategy(
        strategy_name=args.strategy,
        recent_months=args.recent_months,
        min_keyword_freq=args.min_keyword_freq,
        model_name=args.model_name,
        similarity_config=args.similarity_config,
    )
    bt_config = BacktestConfig(
        top_k=args.top_k,
        horizon_months=args.horizon_months,
        min_train_papers=args.min_train_papers,
        start_month=args.start_month,
        end_month=args.end_month,
        similarity_config=args.similarity_config,
    )

    print(f"\nRunning per-topic backtest (horizon={args.horizon_months}m) ...\n")
    topic_results = {}
    total_windows = 0

    for topic in topics:
        scoped = grouped.get(topic.id, [])
        if not scoped:
            print(f"  [{topic.id}] 0 papers — skipped")
            topic_results[topic.id] = {
                "topic_name": topic.name,
                "paper_count": 0,
                "backtest": None,
            }
            continue

        result = backtest(
            papers=scoped,
            strategy=strategy_obj,
            config=bt_config,
        )
        summary = result.get("summary", {})
        windows = summary.get("windows", 0)
        total_windows += windows
        topic_results[topic.id] = {
            "topic_name": topic.name,
            "paper_count": len(scoped),
            "backtest": result,
        }
        print(
            f"  [{topic.id}] {len(scoped)} papers, {windows} windows — "
            f"hit@k={summary.get('avg_hit_at_k', 0):.4f}, "
            f"mrr={summary.get('avg_mrr', 0):.4f}, "
            f"novelty={summary.get('avg_novelty', 0):.4f}, "
            f"diversity={summary.get('avg_diversity', 0):.4f}"
        )

    # Weighted-average summary across topics
    weighted_metrics: dict[str, float] = {}
    for metric in ("avg_hit_at_k", "avg_recall_at_k", "avg_precision_at_k", "avg_mrr", "avg_novelty", "avg_diversity"):
        numerator = 0.0
        denominator = 0
        for tid, tr in topic_results.items():
            bt = tr.get("backtest")
            if not bt:
                continue
            s = bt.get("summary", {})
            w = s.get("windows", 0)
            if w > 0:
                numerator += float(s.get(metric, 0)) * w
                denominator += w
        weighted_metrics[metric] = round(numerator / denominator, 4) if denominator else 0.0

    print(f"\n{'='*60}")
    print(f"Aggregate: {total_windows} windows across {len(topics)} topics")
    for k, v in weighted_metrics.items():
        print(f"  {k}: {v:.4f}")

    output_path = Path(args.output)
    # Resolve the actual model name used (for predictor_llm)
    resolved_model_name: str | None = None
    if args.strategy == "predictor_llm":
        from live_idea_bench.config import load_predictor_config, load_runtime_config
        _pc = load_predictor_config()
        _rc = load_runtime_config()
        resolved_model_name = args.model_name or _pc.default_model or _rc.model_name

    payload = {
        "mode": "domain_backtest",
        "strategy": args.strategy,
        "model_name": resolved_model_name,
        "config": {
            "top_k": args.top_k,
            "horizon_months": args.horizon_months,
            "min_train_papers": args.min_train_papers,
            "start_month": args.start_month,
            "end_month": args.end_month,
        },
        "total_papers": len(papers),
        "total_windows": total_windows,
        "aggregate_summary": weighted_metrics,
        "topic_results": topic_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {output_path}")

    if _tmp_sim_cfg:
        import os
        os.unlink(_tmp_sim_cfg.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
