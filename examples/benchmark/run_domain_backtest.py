"""Run a domain-separated backtest from the command line.

This exercises the same per-topic backtest pipeline used by the backend
(strategy_store.run_backtest_sync) but without requiring Flask or a
persisted strategy JSON file.

Usage::

    python examples/run_domain_backtest.py \
        --input-dir data/csml/raw_markdown \
        --strategy topic_trend \
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

from live_idea_bench.backtest import (
    BacktestConfig,
    backtest,
    weighted_mean_over_topics,
)
from live_idea_bench.paper_cache import load_papers_and_topics
from live_idea_bench.strategy import create_strategy

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run domain-separated backtest across configured topics.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/csml/raw_markdown",
        help="Directory with markdown papers.",
    )
    parser.add_argument("--strategy", type=str, default="topic_trend")
    parser.add_argument("--recent-months", type=int, default=3)
    parser.add_argument("--min-keyword-freq", type=int, default=1)
    parser.add_argument(
        "--model-name", type=str, help="Model override for predictor_llm."
    )
    parser.add_argument(
        "--prior-checkpoint",
        type=str,
        default=None,
        help="Path to trained prior SFT checkpoint (forecaster strategy).",
    )
    parser.add_argument(
        "--realization-checkpoint",
        type=str,
        default=None,
        help="Path to trained GRPO realization checkpoint (forecaster strategy).",
    )
    parser.add_argument(
        "--memory-path",
        type=str,
        default=None,
        help="Path to memory snapshot (forecaster strategy).",
    )
    parser.add_argument(
        "--policy-manifest-path",
        type=str,
        default=None,
        help="Path to policy manifest JSON (policy_rl strategy).",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--horizon-months",
        type=int,
        default=3,
        help="Months past the cutoff month. The cutoff month itself counts as "
        "future, so N=3 spans four calendar months.",
    )
    parser.add_argument("--min-train-papers", type=int, default=2)
    parser.add_argument("--start-month", type=str, default="2024-01")
    parser.add_argument("--end-month", type=str, default="2025-06")
    parser.add_argument(
        "--min-cutoff-month",
        type=str,
        default=None,
        help="Earliest cutoff to EVALUATE (>= this). Lets start-month load "
        "earlier papers as context while only scoring test-period cutoffs.",
    )
    parser.add_argument("--similarity-config", type=str, default="similarity.yaml")
    parser.add_argument(
        "--eval-model",
        type=str,
        default=None,
        help="Model to use for LLM-based similarity evaluation (e.g. gpt-5.4). ",
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
        default=None,
        help="Output JSON path. Defaults to output/backtest/<strategy>.json -- "
        "per strategy, because a shared default plus resume silently merged "
        "runs of different strategies into one artifact.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of topics to process in parallel (default 1). "
        "Threads, so this only speeds up the API-bound part -- the lexical "
        "prefilter in similarity.py is pure Python and GIL-bound. Shard with "
        "--topics across processes to parallelise that.",
    )
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Generate predictions but do not score them here. `judge-eval` reads "
        "only cutoff_month/cutoff_date/predictions from this artifact and "
        "re-embeds everything itself, so when the judge supplies the reported "
        "numbers this run's embedding match is duplicated work -- it costs "
        "O(top_k * candidate_limit) Voyage calls plus a SequenceMatcher "
        "prefilter over every future paper, per window. Metrics come out NaN "
        "(never 0.0) and the artifact is stamped matching_skipped.",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Comma-separated topic ids to score (default: all). Exists to shard "
        "one strategy across processes: ~95%% of a window's non-API time is "
        "difflib.SequenceMatcher inside the prefilter, which is pure Python and "
        "therefore GIL-bound, so --workers cannot parallelise it. Topics are "
        "independent, so N processes over disjoint --topics sets scale linearly. "
        "Each shard needs its own --output; merge the topic_results afterwards.",
    )
    args = parser.parse_args()

    # Apply runtime engine override: write a minimal temp YAML so the caller
    # never needs to touch similarity.yaml just to change the engine.

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    papers, topics, grouped = load_papers_and_topics(
        input_dir, args.start_month, args.end_month
    )

    if not papers:
        print("No papers found. Exiting.")
        return 1

    if args.topics:
        wanted = [t.strip() for t in args.topics.split(",") if t.strip()]
        known = {t.id for t in topics}
        # Reject unknown ids rather than silently scoring fewer topics: a typo in
        # one shard of a sharded run would otherwise drop those topics from the
        # merged result, and the aggregate would look complete.
        unknown = [t for t in wanted if t not in known]
        if unknown:
            print(
                f"Unknown topic id(s): {', '.join(unknown)}.\n"
                f"Known ids: {', '.join(sorted(known))}",
                file=sys.stderr,
            )
            return 2
        selected = set(wanted)
        topics = [t for t in topics if t.id in selected]
        print(f"Sharded to {len(topics)} topic(s): {', '.join(t.id for t in topics)}")

    for topic_id, topic_papers in grouped.items():
        print(f"  {topic_id}: {len(topic_papers)} papers")

    strategy_obj = create_strategy(
        strategy_name=args.strategy,
        recent_months=args.recent_months,
        min_keyword_freq=args.min_keyword_freq,
        model_name=args.model_name,
        similarity_config=args.similarity_config,
        reasoning_effort=args.reasoning_effort,
        prior_checkpoint=args.prior_checkpoint,
        realization_checkpoint=args.realization_checkpoint,
        memory_path=args.memory_path,
        policy_manifest_path=args.policy_manifest_path,
    )
    bt_config = BacktestConfig(
        top_k=args.top_k,
        horizon_months=args.horizon_months,
        min_train_papers=args.min_train_papers,
        start_month=args.start_month,
        end_month=args.end_month,
        min_cutoff_month=args.min_cutoff_month,
        similarity_config=args.similarity_config,
        candidate_limit=args.candidate_limit,
        skip_matching=args.skip_matching,
    )
    if args.skip_matching:
        print(
            "\n"
            + "=" * 72
            + "\n  --skip-matching: predictions only, NOT scored here.\n"
            "  Every metric in this artifact is NaN. Score it with `judge-eval`.\n"
            + "=" * 72,
            flush=True,
        )

    # ── Resume support: load existing partial results ────────────────
    output_path = Path(
        args.output or PROJECT_ROOT / "output" / "backtest" / f"{args.strategy}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_fingerprint = {
        "strategy": args.strategy,
        "top_k": args.top_k,
        "horizon_months": args.horizon_months,
        "min_train_papers": args.min_train_papers,
        "start_month": args.start_month,
        "end_month": args.end_month,
    }
    topic_results: dict = {}
    _saved_payload: dict = {}
    if output_path.exists():
        try:
            _saved_payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as _e:
            print(f"  [resume] could not read existing output ({_e}), starting fresh")
            _saved_payload = {}
        if _saved_payload:
            prior = {
                "strategy": _saved_payload.get("strategy"),
                **{
                    k: (_saved_payload.get("config") or {}).get(k)
                    for k in (
                        "top_k",
                        "horizon_months",
                        "min_train_papers",
                        "start_month",
                        "end_month",
                    )
                },
            }
            if prior != run_fingerprint:
                # Resuming across configurations merges incompatible windows and
                # then stamps the artifact with the CURRENT header, so the file
                # claims a provenance its contents do not have.
                differing = [
                    k for k in run_fingerprint if prior.get(k) != run_fingerprint[k]
                ]
                raise SystemExit(
                    f"{output_path} was produced by a different configuration "
                    f"(differs on: {', '.join(differing)}).\n"
                    "Refusing to resume: merging them would report one set of "
                    "numbers under another set's settings.\n"
                    "Pass a different --output, or delete that file to start fresh."
                )
            topic_results = _saved_payload.get("topic_results", {})
            _resumed = sum(
                1 for v in topic_results.values() if v.get("backtest") is not None
            )
            print(f"  [resume] found {_resumed} completed topics in {output_path}")
            topic_results = {}

    print(f"\nRunning per-topic backtest (horizon={args.horizon_months}m) ...\n")
    total_windows = sum(
        (v.get("backtest") or {}).get("summary", {}).get("windows", 0)
        for v in topic_results.values()
    )

    _lock = threading.Lock()

    # Helper: write current state to disk after every topic (must hold _lock)
    def _save_checkpoint() -> None:
        # Which model produced these predictions. This used to resolve only for
        # predictor_llm and be written as null for the other four strategies, so
        # an artifact could not say which backbone generated it -- only its file
        # path could, and that is lost the moment results are merged. Every
        # strategy takes --model-name, so record it for every strategy.
        resolved: str | None = args.model_name
        if args.strategy == "predictor_llm" and not resolved:
            from live_idea_bench.config import (
                load_predictor_config,
                load_runtime_config,
            )

            _pc = load_predictor_config()
            _rc = load_runtime_config()
            resolved = _pc.default_model or _rc.model_name
        weighted = weighted_mean_over_topics(
            topic_results,
            (
                "avg_hit_at_k",
                "avg_precision_at_k",
                "avg_mrr",
                "avg_novelty",
                "avg_diversity",
            ),
        )
        payload = {
            "mode": "domain_backtest",
            "strategy": args.strategy,
            "model_name": resolved,
            "eval_model": args.eval_model,
            "reasoning_effort": args.reasoning_effort,
            "config": {
                "top_k": args.top_k,
                "horizon_months": args.horizon_months,
                "min_train_papers": args.min_train_papers,
                "start_month": args.start_month,
                "end_month": args.end_month,
                "min_cutoff_month": args.min_cutoff_month,
                "candidate_limit": args.candidate_limit,
            },
            # Stamped on the artifact so a consumer cannot read NaN metrics as
            # a failed run: they were never computed here by design.
            "matching_skipped": bool(args.skip_matching),
            "topics_shard": args.topics,
            "total_papers": len(papers),
            "total_windows": total_windows,
            "aggregate_summary": weighted,
            "topic_results": topic_results,
        }
        # PaperRecord / IdeaPrediction are dataclasses, not JSON-native.
        # default=_jsonable handles them (and any other dataclass) by falling
        # back to asdict, so a single non-serializable value can't silently
        # kill every checkpoint write.
        import dataclasses

        def _jsonable(obj):
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return dataclasses.asdict(obj)
            if hasattr(obj, "model_dump"):  # pydantic v2 escape hatch
                return obj.model_dump()
            if hasattr(obj, "__dict__"):
                return vars(obj)
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable"
            )

        try:
            output_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=_jsonable),
                encoding="utf-8",
            )
        except Exception as _e:
            import traceback

            print(f"[checkpoint ERROR] write failed: {_e}", flush=True)
            traceback.print_exc()
            raise

    def _process_topic(topic):
        """Process one topic; returns (topic_id, result_dict) or None if skipped."""
        # Skip already-completed topics
        with _lock:
            if (
                topic.id in topic_results
                and topic_results[topic.id].get("backtest") is not None
            ):
                existing = topic_results[topic.id]
                w = (
                    (existing.get("backtest") or {})
                    .get("summary", {})
                    .get("windows", 0)
                )
                print(f"  [{topic.id}] skipped (already done, {w} windows)", flush=True)
                return None

        scoped = grouped.get(topic.id, [])
        if not scoped:
            print(f"  [{topic.id}] 0 papers — skipped", flush=True)
            return topic.id, {
                "topic_name": topic.name,
                "paper_count": 0,
                "backtest": None,
            }

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
        return topic.id, {
            "topic_name": topic.name,
            "paper_count": len(scoped),
            "backtest": result,
        }

    failed_topics: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_topic, t): t for t in topics}
        for fut in as_completed(futures):
            topic = futures[fut]
            try:
                res = fut.result()
            except Exception as _topic_exc:
                # Isolate one topic's failure from the whole run. Common
                # cause: openai.APITimeoutError on a stuck stream.
                print(
                    f"  [{topic.id}] FAILED: {type(_topic_exc).__name__}: {_topic_exc}",
                    flush=True,
                )
                import traceback

                traceback.print_exc()
                failed_topics.append(topic.id)
                continue
            if res is None:
                continue
            tid, entry = res
            with _lock:
                topic_results[tid] = entry
                windows = (
                    (entry.get("backtest") or {}).get("summary", {}).get("windows", 0)
                )
                total_windows += windows
                try:
                    _save_checkpoint()
                except Exception as _ckpt_e:
                    print(
                        f"[checkpoint WARN] could not write checkpoint: {_ckpt_e}",
                        flush=True,
                    )

    # Weighted-average summary across topics
    weighted_metrics = weighted_mean_over_topics(
        topic_results,
        (
            "avg_hit_at_k",
            "avg_precision_at_k",
            "avg_mrr",
            "avg_novelty",
            "avg_diversity",
        ),
    )

    print(f"\n{'=' * 60}")
    print(f"Aggregate: {total_windows} windows across {len(topics)} topics")
    for k, v in weighted_metrics.items():
        print(f"  {k}: {v:.4f}")

    _save_checkpoint()
    print(f"\nSaved to {output_path}")

    # A run that scored nothing must not look like a run that scored zero.
    # Without this an all-topics-failed run exits 0 and leaves behind a
    # well-formed artifact full of 0.0000 -- indistinguishable from a model
    # that simply predicted badly.
    if total_windows == 0:
        print(
            f"\nNO WINDOWS SCORED. {len(failed_topics)} topic(s) errored"
            + (f": {', '.join(failed_topics[:5])}" if failed_topics else "")
            + ".\nThe metrics above are not results -- nothing was evaluated. "
            "Check the corpus covers the window, and that required API keys are set.",
            flush=True,
        )
        return 1
    if failed_topics:
        print(
            f"\nWARNING: {len(failed_topics)} of {len(topics)} topics errored and "
            f"are absent from the aggregate: {', '.join(failed_topics[:10])}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
