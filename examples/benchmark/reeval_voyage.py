#!/usr/bin/env python3
"""Re-evaluate saved backtest predictions with Voyage AI batch embeddings.

Usage:
    python examples/reeval_voyage.py \
        --input-json logs/baselines/keyword_trend_raw.json \
        --papers-dir data/csml_v2/raw_markdown \
        --output logs/baselines/keyword_trend_voyage.json

Loads predictions from a previous run_domain_backtest.py output, batch-embeds
all relevant papers and predictions with voyage-4-large (one API call per 128
texts), then computes cosine similarity locally — no per-pair API calls.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import openai

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from live_idea_bench.backtest import split_train_future_by_cutoff  # noqa: E402
from live_idea_bench.config import load_topics  # noqa: E402
from live_idea_bench.models import IdeaPrediction  # noqa: E402
from live_idea_bench.papers import load_papers_from_markdown  # noqa: E402
from live_idea_bench.similarity import paper_text, _sanitize  # noqa: E402
from live_idea_bench.topics import classify_papers_by_topic  # noqa: E402

VOYAGE_BASE_URL = "https://api.voyageai.com/v1"
DEFAULT_MODEL   = "voyage-4-large"
DEFAULT_THRESHOLD = 0.80
BATCH_SIZE   = 128
MAX_CHARS    = 4000
MAX_RETRIES  = 7


def _make_client(api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key, base_url=VOYAGE_BASE_URL)


def _embed_batch(texts: list[str], client: openai.OpenAI, model: str) -> list[list[float]]:
    """Embed texts in batches of BATCH_SIZE with exponential-backoff retry."""
    results: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vecs = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
                results.extend(vecs)
                break
            except Exception as exc:
                wait = 2 ** attempt
                if attempt < MAX_RETRIES - 1:
                    print(f"\n[embed retry {attempt+1}/{MAX_RETRIES}, wait {wait}s] {exc}", flush=True)
                    time.sleep(wait)
                else:
                    raise
    return results


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return max(0.0, min(1.0, dot / (na * nb))) if na and nb else 0.0


def _pred_text(p: IdeaPrediction) -> str:
    parts = [p.title]
    if p.rationale:
        parts.append(p.rationale)
    if p.key_terms:
        parts.append(", ".join(p.key_terms))
    return _sanitize(" ".join(parts))


def _load_predictions(raw: list[dict]) -> list[IdeaPrediction]:
    return [
        IdeaPrediction(
            rank=p["rank"], title=p["title"],
            rationale=p.get("rationale", ""),
            approach=p.get("approach", ""),
            score=p.get("score", 0.0),
            confidence=p.get("confidence", 0.0),
            key_terms=p.get("key_terms", []),
        )
        for p in raw
    ]


def _evaluate_window(
    predictions: list[IdeaPrediction],
    future_paper_ids: list[str],
    paper_vecs: dict[str, list[float]],
    pred_vecs: list[list[float]],
    k: int,
    threshold: float,
) -> dict[str, Any]:
    from live_idea_bench.similarity import _token_jaccard

    used: set[str] = set()
    matched_ranks: list[int] = []
    per_pred: list[dict] = []

    for rank, (pred, pvec) in enumerate(zip(predictions[:k], pred_vecs[:k]), start=1):
        best_score   = 0.0
        best_pid     = None
        all_scores   = []
        for pid in future_paper_ids:
            fvec = paper_vecs.get(pid)
            if fvec is None:
                continue
            score = _cosine(pvec, fvec)
            all_scores.append(score)
            if score > best_score and pid not in used:
                best_score = score
                best_pid   = pid
        is_match = best_score >= threshold and best_pid is not None
        if is_match:
            matched_ranks.append(rank)
            used.add(best_pid)
        per_pred.append({
            "rank": rank, "title": pred.title,
            "best_score": round(best_score, 4),
            "best_paper_id": best_pid,
            "is_match": is_match,
            "score_mean": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
            "score_max":  round(max(all_scores), 4) if all_scores else 0.0,
        })

    hit  = 1.0 if matched_ranks else 0.0
    mrr  = 1.0 / matched_ranks[0] if matched_ranks else 0.0
    prec = len(matched_ranks) / k if k else 0.0
    rec  = len(matched_ranks) / len(future_paper_ids) if future_paper_ids else 0.0

    # Novelty & diversity use token-level (no embedding)
    pred_texts = [_pred_text(p) for p in predictions[:k]]
    return {
        "hit_at_k": hit, "recall_at_k": round(rec, 4),
        "precision_at_k": round(prec, 4), "mrr": round(mrr, 4),
        "per_prediction": per_pred,
    }


def _process_topic(
    topic_id: str,
    saved_topic: dict,
    scoped_papers,
    horizon_months: int,
    top_k: int,
    threshold: float,
    client: openai.OpenAI,
    model: str,
) -> tuple[str, dict]:
    saved_bt = saved_topic.get("backtest")
    if not saved_bt or not saved_bt.get("windows"):
        return topic_id, {"topic_name": saved_topic.get("topic_name", ""), "paper_count": 0, "backtest": None}

    # Batch-embed all papers in this topic once
    paper_ids   = [p.paper_id for p in scoped_papers]
    paper_texts = [_sanitize(paper_text(p)[:MAX_CHARS]) for p in scoped_papers]
    print(f"  [{topic_id}] embedding {len(scoped_papers)} papers ...", flush=True)
    paper_vecs_list = _embed_batch(paper_texts, client, model)
    paper_vecs: dict[str, list[float]] = dict(zip(paper_ids, paper_vecs_list))

    windows_out = []
    for w in saved_bt["windows"]:
        cutoff  = w["cutoff_month"]
        cut_date = w["cutoff_date"]
        predictions = _load_predictions(w["predictions"])
        if not predictions:
            continue

        train, future, future_end, future_end_date = split_train_future_by_cutoff(
            papers=scoped_papers, cutoff_month=cutoff,
            horizon_months=horizon_months, cutoff_date=cut_date,
        )
        if not future:
            continue

        future_ids  = [p.paper_id for p in future]
        pred_texts  = [_sanitize(_pred_text(p)) for p in predictions[:top_k]]
        pred_vecs   = _embed_batch(pred_texts, client, model)

        evaluation = _evaluate_window(
            predictions=predictions, future_paper_ids=future_ids,
            paper_vecs=paper_vecs, pred_vecs=pred_vecs,
            k=top_k, threshold=threshold,
        )
        windows_out.append({
            "cutoff_month": cutoff, "future_end_month": future_end,
            "train_papers": len(train), "future_papers": len(future),
            "evaluation": evaluation,
        })

    if not windows_out:
        return topic_id, {"topic_name": saved_topic.get("topic_name", ""), "paper_count": len(scoped_papers), "backtest": None}

    def avg(k: str) -> float:
        vals = [w["evaluation"][k] for w in windows_out]
        return round(sum(vals) / len(vals), 4)

    summary = {
        "windows": len(windows_out),
        "avg_hit_at_k":       avg("hit_at_k"),
        "avg_recall_at_k":    avg("recall_at_k"),
        "avg_precision_at_k": avg("precision_at_k"),
        "avg_mrr":            avg("mrr"),
    }
    print(
        f"  [{topic_id}] {len(windows_out)} windows — "
        f"hit@k={summary['avg_hit_at_k']:.4f}  mrr={summary['avg_mrr']:.4f}",
        flush=True,
    )
    return topic_id, {
        "topic_name": saved_topic.get("topic_name", ""),
        "paper_count": len(scoped_papers),
        "backtest": {"summary": summary, "windows": windows_out},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-evaluate backtest predictions with Voyage embeddings.")
    parser.add_argument("--input-json",  required=True, help="Path to run_domain_backtest.py output JSON")
    parser.add_argument("--papers-dir",  required=True, help="Papers directory (data/csml_v2/raw_markdown)")
    parser.add_argument("--output",      default="/tmp/reeval_voyage.json")
    parser.add_argument("--model",       default=DEFAULT_MODEL, help="Voyage embedding model")
    parser.add_argument("--threshold",   type=float, default=DEFAULT_THRESHOLD, help="Cosine similarity match threshold")
    parser.add_argument("--topics-config", default=None, help="Path to topics config YAML (defaults to config/config.yaml)")
    args = parser.parse_args()

    api_key = os.environ.get("VOYAGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set VOYAGE_API_KEY env var", file=sys.stderr)
        return 1

    client = _make_client(api_key)
    saved  = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    cfg    = saved.get("config", {})
    start_month    = cfg.get("start_month", "2023-01")
    end_month      = cfg.get("end_month",   "2025-06")
    horizon_months = cfg.get("horizon_months", 3)
    top_k          = cfg.get("top_k", 5)

    print(f"Loading papers from {args.papers_dir} ...", flush=True)
    papers = load_papers_from_markdown(
        Path(args.papers_dir),
        start_month=start_month,
        end_month=end_month,
    )
    print(f"Loaded {len(papers)} papers", flush=True)

    topics  = load_topics(args.topics_config)
    grouped = classify_papers_by_topic(papers, topics)

    topic_results: dict[str, Any] = {}
    total_windows = 0

    for topic in topics:
        saved_topic = saved.get("topic_results", {}).get(topic.id, {})
        scoped      = grouped.get(topic.id, [])
        if not scoped or not saved_topic.get("backtest"):
            topic_results[topic.id] = {
                "topic_name": topic.name, "paper_count": len(scoped), "backtest": None
            }
            continue

        tid, result = _process_topic(
            topic_id=topic.id, saved_topic=saved_topic,
            scoped_papers=scoped, horizon_months=horizon_months,
            top_k=top_k, threshold=args.threshold,
            client=client, model=args.model,
        )
        topic_results[tid] = result
        bt = result.get("backtest")
        if bt:
            total_windows += bt["summary"]["windows"]

    # Weighted aggregate
    weighted: dict[str, float] = {}
    for metric in ("avg_hit_at_k", "avg_recall_at_k", "avg_precision_at_k", "avg_mrr"):
        num, den = 0.0, 0
        for tr in topic_results.values():
            bt = tr.get("backtest")
            if not bt:
                continue
            s = bt["summary"]
            w = s.get("windows", 0)
            if w > 0:
                num += s[metric] * w
                den += w
        weighted[metric] = round(num / den, 4) if den else 0.0

    print(f"\n{'='*60}")
    print(f"Aggregate: {total_windows} windows | model={args.model} | threshold={args.threshold}")
    for k, v in weighted.items():
        print(f"  {k}: {v:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "reeval_voyage",
        "source_json": args.input_json,
        "embedding_model": args.model,
        "threshold": args.threshold,
        "config": cfg,
        "total_windows": total_windows,
        "aggregate_summary": weighted,
        "topic_results": topic_results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
