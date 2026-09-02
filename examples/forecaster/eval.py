"""Single forecaster eval entry point — runs the trained model on a held-out test window.

Reads the prior and realization checkpoints from one training output dir
(``prior_sft/final_checkpoint`` and the GRPO checkpoint, see ``--output-dir``),
runs the per-topic backtest across the test window (default ``2024-10 ~ 2025-03`` — clean held-out from the ``2023-01 ~ 2024-09``
training window), writes a JSON report in the same shape as
``examples/run_domain_backtest.py`` so downstream tools (Voyage re-eval, etc.) keep
working unchanged.

When ``SGLANG_PRIOR_URL`` and ``SGLANG_URL`` are set (pointing at OpenAI-compatible
servers for the prior and the realization policy), the underlying inference
helpers automatically route through them — see
``forecaster/prior/sampler.py:_sglang_prior_url`` and
``forecaster/realization/proposal_generator.py:_sglang_api_url``. Those helpers speak
plain OpenAI ``/v1/completions`` and ``/v1/chat/completions``, so any OpenAI-compatible
server (SGLang, vLLM, llama.cpp's server, …) works — the env-var names are historical.

Usage:
    python examples/forecaster/eval.py \\
        --model qwen3.5-2b \\
        --output-dir output/forecaster_qwen3.5-2b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from forecaster.realization.model_zoo import resolve_small_model
from live_idea_bench.backtest import (
    BacktestConfig,
    backtest,
    weighted_mean_over_topics,
)
from live_idea_bench.paper_cache import load_papers_and_topics
from live_idea_bench.strategy import create_strategy

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("forecaster.eval")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run held-out eval on trained forecaster (Prior + Realization LoRAs).",
    )
    p.add_argument(
        "--model", default="qwen3.5-2b", help="Model preset alias (see model_zoo)."
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="The same dir train.py wrote to. Auto-finds prior_sft/final_checkpoint and "
        "realization_grpo/grpo/checkpoints/final_checkpoint underneath, unless overridden.",
    )
    p.add_argument(
        "--prior-checkpoint",
        default=None,
        help="Override path to prior LoRA adapter (default: <output-dir>/prior_sft/final_checkpoint).",
    )
    p.add_argument(
        "--realization-checkpoint",
        default=None,
        help="Override path to realization LoRA adapter "
        "(default: <output-dir>/realization_grpo/grpo/checkpoints/final_checkpoint).",
    )
    p.add_argument(
        "--papers",
        default=str(PROJECT_ROOT.parent / "md_mineru"),
        help="Directory of markdown papers (default: ../md_mineru relative to repo root).",
    )
    p.add_argument(
        "--start-month",
        default="2024-10",
        help="Test window start month (default: 2024-10 — clean held-out from training).",
    )
    p.add_argument(
        "--end-month",
        default="2025-03",
        help="Test window end month (default: 2025-03).",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--horizon-months",
        type=int,
        default=3,
        help="Months past the cutoff month. The cutoff month itself counts as "
        "future, so N=3 spans four calendar months.",
    )
    p.add_argument("--min-train-papers", type=int, default=2)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Topics processed in parallel (default 1; vLLM does its own batching).",
    )
    p.add_argument("--similarity-config", default="similarity.yaml")
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <output-dir>/eval_test_<start>_<end>.json).",
    )
    p.add_argument(
        "--max-topics",
        type=int,
        default=None,
        help="Cap number of topics processed (smoke testing).",
    )
    p.add_argument(
        "--voyage",
        action="store_true",
        help="After main eval, also run examples/benchmark/reeval_voyage.py to produce "
        "paper-comparable scores. Requires VOYAGE_API_KEY in environment.",
    )
    p.add_argument(
        "--no-vllm-server",
        action="store_true",
        help="Force the slow HF generate path (clears SGLANG_*_URL env vars before running). "
        "Useful for debugging the wiring without a vLLM server.",
    )
    return p


def _resolve_checkpoints(
    args: argparse.Namespace, output_dir: Path
) -> tuple[Path, Path]:
    prior = (
        Path(args.prior_checkpoint)
        if args.prior_checkpoint
        else (output_dir / "prior_sft" / "final_checkpoint")
    )
    real = (
        Path(args.realization_checkpoint)
        if args.realization_checkpoint
        else (
            output_dir
            / "realization_grpo"
            / "grpo"
            / "checkpoints"
            / "final_checkpoint"
        )
    )
    for label, p in [("prior", prior), ("realization", real)]:
        cfg = p / "adapter_config.json"
        if not cfg.exists():
            raise SystemExit(
                f"ERROR: {label} checkpoint missing adapter_config.json at {cfg}\n"
                f"       Did training finish? Pass --{label}-checkpoint to override."
            )
    return prior.resolve(), real.resolve()


def _load_papers_and_topics(
    papers_dir: Path,
    start_month: str,
    end_month: str,
) -> tuple[list, list, dict]:
    """Load + cache papers/topics; see live_idea_bench.paper_cache."""
    return load_papers_and_topics(papers_dir, start_month, end_month, verbose=False)


def _aggregate(topic_results: dict) -> dict:
    """Weighted aggregate across topics (same metric set as run_domain_backtest.py)."""
    return weighted_mean_over_topics(
        topic_results,
        (
            "avg_hit_at_k",
            "avg_precision_at_k",
            "avg_mrr",
            "avg_novelty",
            "avg_diversity",
        ),
    )


def _save_payload(
    output_path: Path,
    args: argparse.Namespace,
    base_model_id: str,
    prior_ckpt: Path,
    real_ckpt: Path,
    topic_results: dict,
    total_papers: int,
) -> None:
    total_windows = sum(
        (v.get("backtest") or {}).get("summary", {}).get("windows", 0)
        for v in topic_results.values()
    )
    payload = {
        "mode": "forecaster_eval",
        "strategy": "forecaster",
        "model_name": base_model_id,
        "prior_checkpoint": str(prior_ckpt),
        "realization_checkpoint": str(real_ckpt),
        "config": {
            "top_k": args.top_k,
            "horizon_months": args.horizon_months,
            "min_train_papers": args.min_train_papers,
            "start_month": args.start_month,
            "end_month": args.end_month,
        },
        "total_papers": total_papers,
        "total_windows": total_windows,
        "aggregate_summary": _aggregate(topic_results),
        "topic_results": topic_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()

    if args.no_vllm_server:
        # Force HF generate path for debugging.
        for var in ("SGLANG_PRIOR_URL", "SGLANG_URL"):
            if var in os.environ:
                log.info("Clearing %s (--no-vllm-server)", var)
                del os.environ[var]

    prior_ckpt, real_ckpt = _resolve_checkpoints(args, output_dir)
    base_model_id = resolve_small_model(args.model).model_id

    papers_dir = Path(args.papers)
    if not papers_dir.is_absolute():
        papers_dir = PROJECT_ROOT / papers_dir
    if not papers_dir.exists():
        raise SystemExit(f"ERROR: papers dir does not exist: {papers_dir}")

    output_path = (
        Path(args.output)
        if args.output
        else (output_dir / f"eval_test_{args.start_month}_{args.end_month}.json")
    )

    log.info("=" * 60)
    log.info("Forecaster eval")
    log.info("  model              : %s -> %s", args.model, base_model_id)
    log.info("  prior checkpoint   : %s", prior_ckpt)
    log.info("  realization ckpt   : %s", real_ckpt)
    log.info("  papers             : %s", papers_dir)
    log.info("  test window        : %s ~ %s", args.start_month, args.end_month)
    log.info("  top-k / horizon    : %d / %d months", args.top_k, args.horizon_months)
    log.info("  workers            : %d", args.workers)
    log.info("  output             : %s", output_path)
    if os.environ.get("SGLANG_PRIOR_URL") or os.environ.get("SGLANG_URL"):
        log.info("  vLLM/SGLang URLs   :")
        log.info(
            "    SGLANG_PRIOR_URL : %s", os.environ.get("SGLANG_PRIOR_URL", "<unset>")
        )
        log.info("    SGLANG_URL       : %s", os.environ.get("SGLANG_URL", "<unset>"))
    else:
        log.info("  vLLM/SGLang URLs   : <unset> — using HF generate fallback (slow)")
    log.info("=" * 60)
    papers, topics, grouped = _load_papers_and_topics(
        papers_dir,
        args.start_month,
        args.end_month,
    )
    if args.max_topics:
        topics = topics[: args.max_topics]
        log.info("--max-topics: limited to first %d topics", len(topics))

    for topic in topics:
        log.info("  topic [%s]: %d papers", topic.id, len(grouped.get(topic.id, [])))

    strategy_obj = create_strategy(
        strategy_name="forecaster",
        model_name=base_model_id,
        similarity_config=args.similarity_config,
        prior_checkpoint=str(prior_ckpt),
        realization_checkpoint=str(real_ckpt),
    )
    bt_config = BacktestConfig(
        top_k=args.top_k,
        horizon_months=args.horizon_months,
        min_train_papers=args.min_train_papers,
        start_month=args.start_month,
        end_month=args.end_month,
        similarity_config=args.similarity_config,
    )

    # ─── Resume support ───────────────────────────────────────────
    topic_results: dict = {}
    if output_path.exists():
        try:
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            topic_results = saved.get("topic_results", {})
            done = sum(
                1 for v in topic_results.values() if v.get("backtest") is not None
            )
            log.info("[resume] found %d completed topics in %s", done, output_path)
        except Exception as exc:
            log.warning(
                "[resume] could not load existing output (%s); starting fresh", exc
            )
            topic_results = {}

    lock = threading.Lock()

    def _process_topic(topic):
        with lock:
            if (
                topic.id in topic_results
                and topic_results[topic.id].get("backtest") is not None
            ):
                w = (
                    (topic_results[topic.id].get("backtest") or {})
                    .get("summary", {})
                    .get("windows", 0)
                )
                log.info("[%s] skipped (already done, %d windows)", topic.id, w)
                return None
        scoped = grouped.get(topic.id, [])
        if not scoped:
            log.info("[%s] 0 papers — skipped", topic.id)
            return topic.id, {
                "topic_name": topic.name,
                "paper_count": 0,
                "backtest": None,
            }
        result = backtest(papers=scoped, strategy=strategy_obj, config=bt_config)
        summary = result.get("summary", {})
        log.info(
            "[%s] %d papers, %d windows — hit@k=%.4f mrr=%.4f novelty=%.4f diversity=%.4f",
            topic.id,
            len(scoped),
            summary.get("windows", 0),
            summary.get("avg_hit_at_k", 0),
            summary.get("avg_mrr", 0),
            summary.get("avg_novelty", 0),
            summary.get("avg_diversity", 0),
        )
        return topic.id, {
            "topic_name": topic.name,
            "paper_count": len(scoped),
            "backtest": result,
        }

    log.info("Running per-topic backtest (workers=%d) ...", args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_topic, t): t for t in topics}
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            tid, entry = res
            with lock:
                topic_results[tid] = entry
                try:
                    _save_payload(
                        output_path,
                        args,
                        base_model_id,
                        prior_ckpt,
                        real_ckpt,
                        topic_results,
                        len(papers),
                    )
                except Exception as exc:
                    log.warning("[checkpoint] could not write payload: %s", exc)

    _save_payload(
        output_path,
        args,
        base_model_id,
        prior_ckpt,
        real_ckpt,
        topic_results,
        len(papers),
    )
    aggregate = _aggregate(topic_results)

    log.info("=" * 60)
    log.info(
        "Aggregate (%d topics):",
        len([t for t in topic_results.values() if t.get("backtest")]),
    )
    for k, v in aggregate.items():
        log.info("  %s = %.4f", k, v)
    log.info("Saved: %s", output_path)
    log.info("=" * 60)

    # Optional Voyage re-eval
    if args.voyage:
        if not os.environ.get("VOYAGE_API_KEY"):
            log.warning(
                "--voyage set but VOYAGE_API_KEY not in environment; skipping re-eval."
            )
        else:
            voyage_path = output_path.with_name(output_path.stem + "_voyage.json")
            log.info("Running Voyage re-eval -> %s", voyage_path)
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "examples" / "benchmark" / "reeval_voyage.py"),
                "--input-json",
                str(output_path),
                "--papers-dir",
                str(papers_dir),
                "--output",
                str(voyage_path),
                "--threshold",
                "0.80",
            ]
            subprocess.run(cmd, check=True)
            log.info("Voyage re-eval complete: %s", voyage_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
