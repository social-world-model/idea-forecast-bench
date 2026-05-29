"""Batch-mode backtest runner using OpenAI Batch API (50% cheaper than real-time).

Workflow
--------
For single-call strategies (predictor_llm, topic_trend):

  Phase 1 — collect : dry-run the backtest recording all LLM requests
  Phase 2 — submit  : upload requests JSONL to OpenAI Batch API
  Phase 3 — poll    : wait for completion, download responses
  Phase 4 — replay  : re-run backtest with cached responses → write output JSON

For memory_prompting (compress → forecast, two dependent LLM calls per window):

  Phase 1a — collect compress requests
  Phase 2a+3a — submit + poll compress batch
  Phase 1b — collect forecast requests (using compress cache so prompts are real)
  Phase 2b+3b — submit + poll forecast batch
  Phase 4    — replay with full cache → write output JSON

All intermediate files are written to --batch-dir so a crashed run can be
resumed by re-running with the same --batch-dir (already-completed phases
are skipped via a state.json checkpoint).

Usage
-----
  python examples/run_batch_backtest.py \\
      --input-dir data/csml_v2/raw_markdown \\
      --strategy predictor_llm \\
      --model-name gpt-5.4 --reasoning-effort medium \\
      --start-month 2023-01 --end-month 2025-06 \\
      --horizon-months 3 --top-k 5 --min-train-papers 5 \\
      --similarity-engine hybrid \\
      --output logs/baselines/predictor_llm_raw.json \\
      --batch-dir logs/batch/predictor_llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from live_idea_bench.backtest import BacktestConfig, backtest  # noqa: E402
from live_idea_bench.config import load_topics  # noqa: E402
from live_idea_bench.llm import (  # noqa: E402
    batch_clear,
    batch_set_collect,
    batch_set_replay,
)
from live_idea_bench.papers import load_papers_from_markdown  # noqa: E402
from live_idea_bench.strategy import create_strategy  # noqa: E402
from live_idea_bench.topics import classify_papers_by_topic  # noqa: E402

MAX_BATCH_TOKENS = 4096

# System-message prefixes used to distinguish the two LLM calls in two-round strategies.
# memory_prompting:  compress -> "You are a research analyst."
#                    forecast -> "You are a research forecasting assistant with memory ..."
# summary_prompting: compress -> "You are a research summarizer."
#                    forecast -> "You are a research forecasting assistant working from a compressed historical summary."
# Forecast prefix is shared across strategies — both forecast calls start with
# "You are a research forecasting assistant", which is enough to separate them from compress calls.
_MEMORY_COMPRESS_SYS_PREFIX = "You are a research analyst."
_SUMMARY_COMPRESS_SYS_PREFIX = "You are a research summarizer."
_FORECAST_SYS_PREFIX = "You are a research forecasting assistant"

_TWO_ROUND_STRATEGIES = {
    "memory_prompting": _MEMORY_COMPRESS_SYS_PREFIX,
    "summary_prompting": _SUMMARY_COMPRESS_SYS_PREFIX,
}


# ── Paper loading ──────────────────────────────────────────────────────────────

def _load_papers_cached(args, input_dir: Path):
    import hashlib
    import pickle

    key = hashlib.md5(
        f"{input_dir}|{args.start_month}|{args.end_month}".encode()
    ).hexdigest()[:12]
    cache_path = PROJECT_ROOT / "data" / f".papers_cache_{key}.pkl"

    if cache_path.exists():
        print(f"Loading papers+topics from cache ({cache_path.name}) ...")
        with open(cache_path, "rb") as f:
            c = pickle.load(f)
        papers, topics, grouped = c["papers"], c["topics"], c["grouped"]
        print(f"Loaded {len(papers)} papers [{args.start_month}–{args.end_month}] (cached)")
    else:
        print(f"Loading papers from {input_dir} ...")
        papers = load_papers_from_markdown(
            input_dir, start_month=args.start_month, end_month=args.end_month
        )
        if not papers:
            print("No papers found. Exiting.")
            sys.exit(1)
        print(f"Loaded {len(papers)} papers [{args.start_month}–{args.end_month}]")
        topics = load_topics()
        print(f"Topics: {len(topics)}. Classifying ...")
        grouped = classify_papers_by_topic(papers, topics)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"papers": papers, "topics": topics, "grouped": grouped}, f)
            print(f"  [cache] saved to {cache_path.name}")
        except Exception as e:
            print(f"  [cache] could not save: {e}")

    return papers, topics, grouped


# ── Collect phase ──────────────────────────────────────────────────────────────

def _collect_requests(
    strategy_obj,
    topics,
    grouped,
    bt_config: BacktestConfig,
    args,
    replay_cache: dict | None = None,
    system_prefix: str | None = None,
    workers: int = 8,
) -> dict:
    """Dry-run the backtest in collect mode; return the captured request dict.

    Parameters
    ----------
    replay_cache:
        Pre-seeded responses served directly (no collection, no live API).
        Used in round-2 of memory_prompting to provide real compress outputs
        so that forecast prompts are built with the correct memory text.
    system_prefix:
        Only collect calls whose system_message starts with this string.
        All other calls still return "" without touching the live API.
    """
    import tempfile

    # Use heuristic similarity during collect — we only care about capturing LLM
    # request prompts, not about evaluation quality.  Heuristic is CPU-only and
    # ~10× faster than hybrid/embedding engines, so collect finishes in minutes.
    _tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="sim_collect_"
    )
    _tmp.write("engine: heuristic\n")
    _tmp.close()
    collect_bt_config = BacktestConfig(
        top_k=bt_config.top_k,
        horizon_months=bt_config.horizon_months,
        min_train_papers=bt_config.min_train_papers,
        start_month=bt_config.start_month,
        end_month=bt_config.end_month,
        similarity_config=_tmp.name,
        candidate_limit=bt_config.candidate_limit,
    )

    collector: dict = {}

    def _worker(topic):
        scoped = grouped.get(topic.id, [])
        if not scoped:
            return
        batch_set_collect(
            collector=collector,
            cache=replay_cache or {},
            system_prefix=system_prefix,
        )
        try:
            backtest(
                papers=scoped,
                strategy=strategy_obj,
                config=collect_bt_config,
                model_name=args.eval_model,
                reasoning_effort=args.reasoning_effort,
            )
        except Exception as e:
            print(f"  [collect] {topic.id}: {e}", flush=True)
        finally:
            batch_clear()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, t) for t in topics]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"  [collect] worker error: {e}", flush=True)

    os.unlink(_tmp.name)
    print(f"  [collect] captured {len(collector)} unique requests", flush=True)
    return collector


# ── Batch API helpers ──────────────────────────────────────────────────────────

def _build_batch_jsonl(
    requests_dict: dict,
    default_model: str,
    default_reasoning_effort: str | None,
) -> str:
    """Convert collected requests dict into an OpenAI Batch API JSONL string."""
    lines = []
    for key, req in requests_dict.items():
        model = req.get("model") or default_model
        body: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": req["system_message"]},
                {"role": "user", "content": req["msg"]},
            ],
        }
        if model.startswith("gpt-5"):
            body["max_completion_tokens"] = MAX_BATCH_TOKENS
            re_val = req.get("reasoning_effort") or default_reasoning_effort
            if re_val:
                body["reasoning_effort"] = re_val
        else:
            body["max_tokens"] = MAX_BATCH_TOKENS
            if req.get("temperature") is not None:
                body["temperature"] = req["temperature"]
        lines.append(json.dumps({
            "custom_id": key,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }))
    return "\n".join(lines)


def _submit_batch(
    jsonl_content: str,
    oai_client,
    batch_dir: Path,
    label: str,
) -> str:
    """Upload JSONL and create a batch job.  Returns the batch_id."""
    jsonl_path = batch_dir / f"{label}_requests.jsonl"
    jsonl_path.write_text(jsonl_content, encoding="utf-8")
    size_kb = len(jsonl_content.encode()) // 1024
    print(f"  [batch] uploading {jsonl_path.name} ({size_kb} KB, "
          f"{jsonl_content.count(chr(10)) + 1} requests) ...")

    with open(jsonl_path, "rb") as f:
        file_obj = oai_client.files.create(file=f, purpose="batch")

    batch = oai_client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  [batch] submitted  id={batch.id}  status={batch.status}")
    (batch_dir / f"{label}_batch_id.txt").write_text(batch.id)
    return batch.id


def _poll_batch(
    batch_id: str,
    oai_client,
    poll_interval: int = 60,
    max_wait_hours: int = 24,
) -> str:
    """Poll until the batch completes.  Returns the output_file_id."""
    deadline = time.time() + max_wait_hours * 3600
    while time.time() < deadline:
        b = oai_client.batches.retrieve(batch_id)
        counts = b.request_counts
        total = counts.total if counts else "?"
        done = counts.completed if counts else 0
        failed = counts.failed if counts else 0
        print(
            f"  [poll] {b.status}  completed={done}/{total}  failed={failed}",
            flush=True,
        )
        if b.status == "completed":
            print(f"  [poll] done  output_file_id={b.output_file_id}")
            return b.output_file_id
        if b.status in ("failed", "expired", "cancelled", "cancelling"):
            raise RuntimeError(f"Batch {batch_id} ended with status: {b.status}")
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Batch {batch_id} did not complete within {max_wait_hours} hours"
    )


def _download_results(
    output_file_id: str,
    oai_client,
    batch_dir: Path,
    label: str,
) -> dict:
    """Download batch output and return {custom_id: response_text} dict."""
    print(f"  [batch] downloading results (file_id={output_file_id}) ...")
    content = oai_client.files.content(output_file_id).text
    (batch_dir / f"{label}_responses.jsonl").write_text(content, encoding="utf-8")

    cache: dict = {}
    error_count = 0
    for line in content.strip().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("custom_id")
        if not cid:
            continue
        if obj.get("error"):
            print(f"  [batch] request {cid[:12]}… failed: {obj['error']}", flush=True)
            error_count += 1
            continue
        resp = obj.get("response", {})
        if resp.get("status_code") == 200:
            choices = (resp.get("body") or {}).get("choices") or []
            if choices:
                text = (choices[0].get("message") or {}).get("content") or ""
                cache[cid] = text
        else:
            error_count += 1

    print(
        f"  [batch] {len(cache)} ok  /  {error_count} errors  "
        f"(total {len(cache) + error_count})"
    )
    return cache


# ── Replay phase ───────────────────────────────────────────────────────────────

def _replay_backtest(
    strategy_obj,
    topics,
    grouped,
    bt_config: BacktestConfig,
    args,
    full_cache: dict,
    workers: int = 8,
) -> tuple[dict, int]:
    """Re-run the backtest using cached LLM responses.  Returns (topic_results, total_windows)."""
    topic_results: dict = {}
    total_windows = 0
    _lock = threading.Lock()

    def _worker(topic):
        scoped = grouped.get(topic.id, [])
        if not scoped:
            return topic.id, {"topic_name": topic.name, "paper_count": 0, "backtest": None}
        batch_set_replay(full_cache)
        try:
            result = backtest(
                papers=scoped,
                strategy=strategy_obj,
                config=bt_config,
                model_name=args.eval_model,
                reasoning_effort=args.reasoning_effort,
            )
        except Exception as e:
            print(f"  [replay] {topic.id}: {e}", flush=True)
            return topic.id, {"topic_name": topic.name, "paper_count": len(scoped), "backtest": None}
        finally:
            batch_clear()

        s = result.get("summary", {})
        print(
            f"  [{topic.id}] {len(scoped)} papers, {s.get('windows', 0)} windows — "
            f"hit@k={s.get('avg_hit_at_k', 0):.4f}  mrr={s.get('avg_mrr', 0):.4f}",
            flush=True,
        )
        return topic.id, {"topic_name": topic.name, "paper_count": len(scoped), "backtest": result}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, t) for t in topics]
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                print(f"  [replay] worker error: {e}", flush=True)
                continue
            if res is None:
                continue
            tid, entry = res
            with _lock:
                topic_results[tid] = entry
                w = (entry.get("backtest") or {}).get("summary", {}).get("windows", 0)
                total_windows += w

    return topic_results, total_windows


# ── Output saving ──────────────────────────────────────────────────────────────

def _save_output(
    topic_results: dict,
    total_windows: int,
    papers: list,
    args,
    output_path: Path,
) -> None:
    """Write final output JSON in the same format as run_domain_backtest.py."""
    resolved = args.model_name
    if args.strategy == "predictor_llm":
        from live_idea_bench.config import load_predictor_config, load_runtime_config
        pc = load_predictor_config()
        rc = load_runtime_config()
        resolved = args.model_name or pc.default_model or rc.model_name

    weighted: dict = {}
    for metric in (
        "avg_hit_at_k", "avg_recall_at_k", "avg_precision_at_k",
        "avg_mrr", "avg_novelty", "avg_diversity",
    ):
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
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved to {output_path}")

    print(f"\n{'='*60}")
    print(f"Aggregate: {total_windows} windows")
    for k, v in weighted.items():
        print(f"  {k}: {v:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-mode domain backtest (OpenAI Batch API, 50% cheaper).",
    )
    # ── same args as run_domain_backtest.py ───────────────────────────────────
    parser.add_argument("--input-dir", type=str, default="data/csml_v2/raw_markdown")
    parser.add_argument("--strategy", type=str, default="predictor_llm")
    parser.add_argument("--recent-months", type=int, default=3)
    parser.add_argument("--min-keyword-freq", type=int, default=1)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--horizon-months", type=int, default=3)
    parser.add_argument("--min-train-papers", type=int, default=2)
    parser.add_argument("--start-month", type=str, default="2024-01")
    parser.add_argument("--end-month", type=str, default="2025-06")
    parser.add_argument("--similarity-config", type=str, default="similarity.yaml")
    parser.add_argument("--similarity-engine", type=str, default=None)
    parser.add_argument("--eval-model", type=str, default=None)
    parser.add_argument("--reasoning-effort", type=str, default=None)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="/tmp/batch_backtest.json")
    parser.add_argument("--workers", type=int, default=8)
    # ── batch-specific args ───────────────────────────────────────────────────
    parser.add_argument(
        "--batch-dir", type=str, default=None,
        help="Directory for intermediate batch files.  "
             "Default: logs/batch/{strategy}_{timestamp}/",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=60,
        help="Seconds between batch status polls (default 60).",
    )
    parser.add_argument(
        "--max-wait-hours", type=int, default=24,
        help="Max hours to wait for batch completion (default 24).",
    )
    args = parser.parse_args()

    # ── similarity engine override ────────────────────────────────────────────
    _tmp_sim_cfg = None
    if args.similarity_engine:
        import tempfile
        _tmp_sim_cfg = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="sim_override_"
        )
        _tmp_sim_cfg.write(f"engine: {args.similarity_engine}\n")
        _tmp_sim_cfg.close()
        args.similarity_config = _tmp_sim_cfg.name

    # ── setup paths ───────────────────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = (
        Path(args.batch_dir)
        if args.batch_dir
        else PROJECT_ROOT / "logs" / "batch" / f"{args.strategy}_{timestamp}"
    )
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"Batch working dir: {batch_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    # ── load data ─────────────────────────────────────────────────────────────
    papers, topics, grouped = _load_papers_cached(args, input_dir)

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

    # ── OpenAI client ─────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return 1
    import openai
    oai_client = openai.OpenAI(api_key=api_key)
    model_for_batch = args.model_name or "gpt-5.4"

    # ── state (resume support) ────────────────────────────────────────────────
    state_path = batch_dir / "state.json"
    state: dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            print(f"  [resume] loaded state: {list(state.keys())}")
        except Exception as e:
            print(f"  [resume] could not load state ({e}), starting fresh")

    def _save_state() -> None:
        state_path.write_text(json.dumps(state, indent=2))

    # ── helper: run one batch round (collect → submit → poll → download) ──────
    def _run_batch_round(
        label: str,
        replay_cache: dict | None,
        system_prefix: str | None,
    ) -> dict:
        """Run one collect+submit+poll+download cycle.  Returns the response cache dict."""
        req_key = f"{label}_requests"
        bid_key = f"{label}_batch_id"
        resp_key = f"{label}_responses"

        # collect
        if req_key not in state:
            print(f"\n[{label}] Phase 1 — collecting requests ...")
            requests = _collect_requests(
                strategy_obj, topics, grouped, bt_config, args,
                replay_cache=replay_cache,
                system_prefix=system_prefix,
                workers=args.workers,
            )
            rpath = batch_dir / f"{label}_requests.json"
            rpath.write_text(json.dumps(requests), encoding="utf-8")
            state[req_key] = str(rpath)
            state[f"{label}_n_requests"] = len(requests)
            _save_state()
        else:
            n = state.get(f"{label}_n_requests", "?")
            print(f"\n[{label}] Phase 1 — skipped ({n} requests already collected)")
            requests = json.loads(Path(state[req_key]).read_text())

        if not requests:
            print(f"  [{label}] No requests collected — skipping batch round")
            return {}

        # submit
        if bid_key not in state:
            print(f"\n[{label}] Phase 2 — submitting batch ...")
            jsonl = _build_batch_jsonl(requests, model_for_batch, args.reasoning_effort)
            batch_id = _submit_batch(jsonl, oai_client, batch_dir, label)
            state[bid_key] = batch_id
            _save_state()
        else:
            batch_id = state[bid_key]
            print(f"\n[{label}] Phase 2 — skipped (batch_id={batch_id})")

        # poll + download
        if resp_key not in state:
            print(f"\n[{label}] Phase 3 — polling batch {batch_id} ...")
            output_fid = _poll_batch(
                batch_id, oai_client, args.poll_interval, args.max_wait_hours
            )
            cache = _download_results(output_fid, oai_client, batch_dir, label)
            cpath = batch_dir / f"{label}_cache.json"
            cpath.write_text(json.dumps(cache), encoding="utf-8")
            state[resp_key] = str(cpath)
            state[f"{label}_n_responses"] = len(cache)
            _save_state()
        else:
            n = state.get(f"{label}_n_responses", "?")
            print(f"\n[{label}] Phase 3 — skipped ({n} responses already downloaded)")
            cache = json.loads(Path(state[resp_key]).read_text())

        return cache

    # ── strategy dispatch ─────────────────────────────────────────────────────
    compress_prefix = _TWO_ROUND_STRATEGIES.get(args.strategy)

    if compress_prefix is None:
        # Single-round: collect all LLM requests, batch them, replay
        full_cache = _run_batch_round(
            label="main",
            replay_cache=None,
            system_prefix=None,
        )

    else:
        # Two-round: compress first, then forecast
        # Round A — compress
        compress_cache = _run_batch_round(
            label="compress",
            replay_cache=None,
            system_prefix=compress_prefix,
        )

        # Round B — forecast (prompts depend on compress outputs)
        forecast_cache = _run_batch_round(
            label="forecast",
            replay_cache=compress_cache,
            system_prefix=_FORECAST_SYS_PREFIX,
        )

        full_cache = {**compress_cache, **forecast_cache}

    # ── Phase 4: replay ───────────────────────────────────────────────────────
    print(f"\n[replay] Running backtest with {len(full_cache)} cached responses ...")
    topic_results, total_windows = _replay_backtest(
        strategy_obj, topics, grouped, bt_config, args, full_cache, args.workers
    )
    _save_output(topic_results, total_windows, papers, args, output_path)
    print(f"\nBatch artifacts: {batch_dir}")

    if _tmp_sim_cfg:
        os.unlink(_tmp_sim_cfg.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
