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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        "--eval-model",
        type=str,
        default=None,
        help="Model to use for LLM-based similarity evaluation (e.g. gpt-5.4). "
             "Only relevant when --similarity-engine llm is used.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="reasoning_effort for OpenAI reasoning models (low/medium/high). "
             "Only applies to gpt-5* models.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="Max future papers evaluated per prediction (LLM engine only). "
             "e.g. 20 means each prediction is compared against at most 20 papers. "
             "Reduces cost significantly; set to None for exhaustive evaluation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/domain_backtest.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of topics to process in parallel (default 1). "
             "Higher values increase throughput at the cost of more concurrent API calls.",
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

    import hashlib
    import pickle

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    # Cache key: input_dir + date range (topics config is stable)
    _cache_key = hashlib.md5(f"{input_dir}|{args.start_month}|{args.end_month}".encode()).hexdigest()[:12]
    _cache_path = PROJECT_ROOT / "data" / f".papers_cache_{_cache_key}.pkl"

    if _cache_path.exists():
        print(f"Loading papers+topics from cache ({_cache_path.name}) ...")
        with open(_cache_path, "rb") as _f:
            _cached = pickle.load(_f)
        papers = _cached["papers"]
        topics = _cached["topics"]
        grouped = _cached["grouped"]
        print(f"Loaded {len(papers)} papers ({args.start_month} to {args.end_month}) [cached]")
    else:
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
        print("Classifying papers by topic (this may take a minute) ...")
        grouped = classify_papers_by_topic(papers, topics)
        try:
            with open(_cache_path, "wb") as _f:
                pickle.dump({"papers": papers, "topics": topics, "grouped": grouped}, _f)
            print(f"  [cache] saved to {_cache_path.name}")
        except Exception as _ce:
            print(f"  [cache] could not save: {_ce}")

    if not papers:
        print("No papers found. Exiting.")
        return 1
    for topic_id, topic_papers in grouped.items():
        print(f"  {topic_id}: {len(topic_papers)} papers")

    strategy_obj = create_strategy(
        strategy_name=args.strategy,
        recent_months=args.recent_months,
        min_keyword_freq=args.min_keyword_freq,
        model_name=args.model_name,
        similarity_config=args.similarity_config,
        reasoning_effort=args.reasoning_effort,
    )
    bt_config = BacktestConfig(
        top_k=args.top_k,
        horizon_months=args.horizon_months,
        min_train_papers=args.min_train_papers,
        start_month=args.start_month,
        end_month=args.end_month,
        similarity_config=args.similarity_config,
        candidate_limit=args.candidate_limit,
    )

    # ── Resume support: load existing partial results ────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    topic_results: dict = {}
    _saved_payload: dict = {}
    if output_path.exists():
        try:
            _saved_payload = json.loads(output_path.read_text(encoding="utf-8"))
            topic_results = _saved_payload.get("topic_results", {})
            _resumed = sum(1 for v in topic_results.values() if v.get("backtest") is not None)
            print(f"  [resume] found {_resumed} completed topics in {output_path}")
        except Exception as _e:
            print(f"  [resume] could not load existing output ({_e}), starting fresh")
            topic_results = {}

    print(f"\nRunning per-topic backtest (horizon={args.horizon_months}m) ...\n")
    total_windows = sum(
        (v.get("backtest") or {}).get("summary", {}).get("windows", 0)
        for v in topic_results.values()
    )

    _lock = threading.Lock()

    # Helper: write current state to disk after every topic (must hold _lock)
    def _save_checkpoint() -> None:
        resolved: str | None = None
        if args.strategy == "predictor_llm":
            from live_idea_bench.config import load_predictor_config, load_runtime_config
            _pc = load_predictor_config()
            _rc = load_runtime_config()
            resolved = args.model_name or _pc.default_model or _rc.model_name
        weighted: dict[str, float] = {}
        for metric in ("avg_hit_at_k", "avg_recall_at_k", "avg_precision_at_k", "avg_mrr", "avg_novelty", "avg_diversity"):
            num, den = 0.0, 0
            for tr in topic_results.values():
                bt = tr.get("backtest")
                if not bt:
                    continue
                s = bt.get("summary", {})
                w = s.get("windows", 0)
                if w > 0:
                    num += float(s.get(metric, 0)) * w
                    den += w
            weighted[metric] = round(num / den, 4) if den else 0.0
        payload = {
            "mode": "domain_backtest",
            "strategy": args.strategy,
            "model_name": resolved,
            "eval_model": args.eval_model,
            "reasoning_effort": args.reasoning_effort,
            "similarity_engine": args.similarity_engine,
            "config": {
                "top_k": args.top_k,
                "horizon_months": args.horizon_months,
                "min_train_papers": args.min_train_papers,
                "start_month": args.start_month,
                "end_month": args.end_month,
            },
            "total_papers": len(papers),
            "total_windows": total_windows,
            "aggregate_summary": weighted,
            "topic_results": topic_results,
        }
        try:
            output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except TypeError as _e:
            import traceback
            print(f"[checkpoint ERROR] JSON serialization failed: {_e}", flush=True)
            traceback.print_exc()
            raise

    def _process_topic(topic):
        """Process one topic; returns (topic_id, result_dict) or None if skipped."""
        # Skip already-completed topics
        with _lock:
            if topic.id in topic_results and topic_results[topic.id].get("backtest") is not None:
                existing = topic_results[topic.id]
                w = (existing.get("backtest") or {}).get("summary", {}).get("windows", 0)
                print(f"  [{topic.id}] skipped (already done, {w} windows)", flush=True)
                return None

        scoped = grouped.get(topic.id, [])
        if not scoped:
            print(f"  [{topic.id}] 0 papers — skipped", flush=True)
            return topic.id, {"topic_name": topic.name, "paper_count": 0, "backtest": None}

        result = backtest(
            papers=scoped,
            strategy=strategy_obj,
            config=bt_config,
            model_name=args.eval_model,
            reasoning_effort=args.reasoning_effort,
        )
        summary = result.get("summary", {})
        windows = summary.get("windows", 0)
        print(
            f"  [{topic.id}] {len(scoped)} papers, {windows} windows — "
            f"hit@k={summary.get('avg_hit_at_k', 0):.4f}, "
            f"mrr={summary.get('avg_mrr', 0):.4f}, "
            f"novelty={summary.get('avg_novelty', 0):.4f}, "
            f"diversity={summary.get('avg_diversity', 0):.4f}",
            flush=True,
        )
        return topic.id, {"topic_name": topic.name, "paper_count": len(scoped), "backtest": result}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_topic, t): t for t in topics}
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            tid, entry = res
            with _lock:
                topic_results[tid] = entry
                windows = (entry.get("backtest") or {}).get("summary", {}).get("windows", 0)
                total_windows += windows
                try:
                    _save_checkpoint()
                except Exception as _ckpt_e:
                    print(f"[checkpoint WARN] could not write checkpoint: {_ckpt_e}", flush=True)

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

    _save_checkpoint()
    print(f"\nSaved to {output_path}")

    if _tmp_sim_cfg:
        import os
        os.unlink(_tmp_sim_cfg.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
