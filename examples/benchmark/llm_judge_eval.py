#!/usr/bin/env python3
"""LLM-as-Judge evaluation for IdeaForecastBench predictions -- the CLI.

The protocol itself lives in ``idea_forecast_bench.judge``; this file parses flags,
loads the corpus, fans out over topics, and writes the report.

Pipeline:
  1. Embed predictions + future papers with voyage-3-large
  2. Retrieve top-R candidates per prediction (cosine similarity)
  3. Call the LLM judge for each candidate (multi-dimensional scoring)
  4. A prediction "hits" if (P+M >= 5) AND (S >= 2), scale 0-3
  5. Compute diversity (cluster coverage) and novelty (embedding distance)

Usage:
    idea-forecast-bench judge-eval \\
        --input-json output/baselines/topic_trend.json \\
        --papers-dir data/csml/raw_markdown \\
        --output output/judge/topic_trend_judged.json

    # Quick check on one topic (2 windows):
    idea-forecast-bench judge-eval \\
        --input-json output/baselines/topic_trend.json \\
        --papers-dir data/csml/raw_markdown \\
        --output output/judge/smoke.json \\
        --topics llm_pretraining --max-windows 2
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import openai

from idea_forecast_bench.backtest import weighted_mean_over_topics
from idea_forecast_bench.config import load_topics
from idea_forecast_bench.judge.config import (
    DEFAULT_CLUSTER_K,
    DEFAULT_JUDGE,
    DEFAULT_TOP_R,
    EMBED_MODEL,
    MATCH_PM_THRESHOLD,
    MATCH_S_THRESHOLD,
    VOYAGE_BASE_URL,
)
from idea_forecast_bench.judge.identity import embed_fingerprint, judge_fingerprint
from idea_forecast_bench.judge.state import RunState
from idea_forecast_bench.judge.topics import process_topic
from idea_forecast_bench.papers import load_papers_from_markdown
from idea_forecast_bench.topics import classify_papers_by_topic


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions with retrieve+LLM-judge pipeline."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--papers-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--state-file",
        default=None,
        help="Checkpoint file (default: <output>.state.json)",
    )
    parser.add_argument(
        "--top-r",
        type=int,
        default=DEFAULT_TOP_R,
        help="Number of candidates to retrieve per prediction",
    )
    parser.add_argument(
        "--cluster-k",
        type=int,
        default=DEFAULT_CLUSTER_K,
        help="Number of clusters for future-paper diversity coverage",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="Base URL for judge model API (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel LLM judge threads per window"
    )
    parser.add_argument(
        "--topic-workers",
        type=int,
        default=4,
        help="Parallel topics processed simultaneously",
    )
    parser.add_argument(
        "--topics",
        default=None,
        help="Comma-separated topic IDs to process (default: all)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Max windows per topic (for testing)",
    )
    parser.add_argument("--topics-config", default=None)
    args = parser.parse_args()

    # Embedding backend: Voyage-only, no fallback. A missing key fails loud
    # rather than silently swapping in a different embedding geometry (which
    # would make scores incomparable to Voyage-embedded runs).
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    if not voyage_key:
        print(
            "ERROR: Set VOYAGE_API_KEY — the judge embeds with Voyage and has no "
            "local fallback (mixing embedding models corrupts score comparability).",
            file=sys.stderr,
        )
        return 1
    # Same override as the benchmark matcher (idea_forecast_bench/similarity.py):
    # without it, VOYAGE_BASE_URL would silently apply to `baselines` but not to
    # `judge-eval`, and the two would embed against different endpoints.
    embed_base_url = os.environ.get("VOYAGE_BASE_URL") or VOYAGE_BASE_URL
    embed_client = openai.OpenAI(api_key=voyage_key, base_url=embed_base_url)

    # Judge client: prefer env-var endpoint so we can target a local
    # vLLM OpenAI-compatible server when OPENAI_API_KEY is absent.
    judge_base_url = args.judge_base_url or os.environ.get("JUDGE_BASE_URL")
    openai_key = os.environ.get("OPENAI_API_KEY")
    judge_api_key = os.environ.get("JUDGE_API_KEY", openai_key or "EMPTY")
    if not judge_base_url and not openai_key:
        print(
            "ERROR: Set OPENAI_API_KEY (for hosted judge) or "
            "JUDGE_BASE_URL/--judge-base-url (for local vLLM judge).",
            file=sys.stderr,
        )
        return 1
    judge_client = openai.OpenAI(
        api_key=judge_api_key,
        base_url=judge_base_url or None,
        timeout=60.0,
    )

    out_path = Path(args.output)
    state_path = (
        Path(args.state_file)
        if args.state_file
        else out_path.with_suffix(".state.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    judge_fp = judge_fingerprint(args.judge_model)
    embed_fp = embed_fingerprint(args.embed_model)
    state = RunState(state_path, judge_fingerprint=judge_fp, embed_fingerprint=embed_fp)
    atexit.register(state.force_flush)

    saved = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    cfg = saved.get("config", {})
    start_month = cfg.get("start_month", "2023-01")
    end_month = cfg.get("end_month", "2025-06")
    horizon_months = cfg.get("horizon_months", 3)
    top_k = cfg.get("top_k", 5)

    print(f"Loading papers from {args.papers_dir} ...", flush=True)
    papers = load_papers_from_markdown(
        Path(args.papers_dir), start_month=start_month, end_month=end_month
    )
    print(f"Loaded {len(papers)} papers", flush=True)

    topics = load_topics(args.topics_config)
    grouped = classify_papers_by_topic(papers, topics)

    if args.topics:
        topic_filter = {t.strip() for t in args.topics.split(",")}
        topics = [t for t in topics if t.id in topic_filter]
        print(
            f"Running on {len(topics)} topic(s): {[t.id for t in topics]}", flush=True
        )

    topic_results: dict[str, Any] = {}
    total_windows = 0

    def _run_topic(topic):
        saved_topic = saved.get("topic_results", {}).get(topic.id, {})
        scoped = grouped.get(topic.id, [])
        if not scoped or not saved_topic.get("backtest"):
            return topic.id, {
                "topic_name": topic.name,
                "paper_count": len(scoped),
                "backtest": None,
            }
        return process_topic(
            topic_id=topic.id,
            saved_topic=saved_topic,
            scoped_papers=scoped,
            horizon_months=horizon_months,
            top_k=top_k,
            top_r=args.top_r,
            cluster_k=args.cluster_k,
            embed_client=embed_client,
            judge_client=judge_client,
            judge_model=args.judge_model,
            state=state,
            workers=args.workers,
            max_windows=args.max_windows,
        )

    results_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.topic_workers) as pool:
        futs = {pool.submit(_run_topic, t): t for t in topics}
        for fut in as_completed(futs):
            tid, result = fut.result()
            with results_lock:
                topic_results[tid] = result
                bt = result.get("backtest")
                if bt:
                    total_windows += bt["summary"]["windows"]
                _write_output(out_path, args, cfg, topic_results, total_windows)

    aggregate = _compute_aggregate(topic_results)

    print(f"\n{'=' * 60}")
    print(f"Aggregate: {total_windows} windows | judge={args.judge_model}")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.4f}")

    _write_output(out_path, args, cfg, topic_results, total_windows, aggregate)
    state.force_flush()
    print(f"\nSaved → {out_path}")
    print(f"State  → {state_path}")
    return 0


def _compute_aggregate(topic_results: dict) -> dict[str, float]:
    return weighted_mean_over_topics(
        topic_results,
        (
            "avg_hit_at_k",
            "avg_mrr",
            "avg_precision_at_k",
            "avg_soft_score",
            "avg_cluster_coverage",
            "avg_novelty",
        ),
    )


def _write_output(
    out_path: Path,
    args: argparse.Namespace,
    cfg: dict,
    topic_results: dict,
    total_windows: int,
    aggregate: dict | None = None,
) -> None:
    payload = {
        "mode": "llm_judge_eval",
        "source_json": args.input_json,
        "embed_model": args.embed_model,
        "judge_model": args.judge_model,
        "top_r": args.top_r,
        "cluster_k": args.cluster_k,
        "match_pm_threshold": MATCH_PM_THRESHOLD,
        "match_s_threshold": MATCH_S_THRESHOLD,
        "config": cfg,
        "total_windows": total_windows,
        "aggregate_summary": aggregate or _compute_aggregate(topic_results),
        "topic_results": topic_results,
    }
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
