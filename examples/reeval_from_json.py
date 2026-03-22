"""Re-evaluate saved backtest predictions using batch OpenAI embeddings.

Loads predictions from a previous backtest JSON (no LLM calls needed),
reloads the paper corpus, pre-embeds every paper and prediction in bulk
(one API call per batch of 2048 texts), then computes cosine similarity
locally — instead of one API call per (prediction, paper) pair.

Usage::

    python examples/reeval_from_json.py \
        --input-json /tmp/predictor_llm_domain_backtest_v4.json \
        --papers-dir /tmp/papers_2024_2025 \
        --output /tmp/reeval_embedding.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import openai  # noqa: E402
from tqdm import tqdm  # noqa: E402

from live_idea_bench.backtest import split_train_future_by_cutoff  # noqa: E402
from live_idea_bench.config import load_runtime_config, load_topics  # noqa: E402
from live_idea_bench.models import IdeaPrediction, PaperRecord  # noqa: E402
from live_idea_bench.papers import load_papers_from_markdown  # noqa: E402
from live_idea_bench.similarity import paper_text, _sanitize  # noqa: E402
from live_idea_bench.topics import classify_papers_by_topic  # noqa: E402

EMBED_MODEL = "text-embedding-3-small"
EMBED_THRESHOLD = 0.4
BATCH_SIZE = 100
MAX_CHARS_PER_TEXT = 2000


def _prediction_text(p: IdeaPrediction) -> str:
    parts = [p.title]
    if p.rationale:
        parts.append(p.rationale)
    if p.key_terms:
        parts.append(", ".join(p.key_terms))
    return _sanitize(" ".join(parts))


def _embed_batch(texts: list[str], client: openai.OpenAI) -> list[list[float]]:
    """Embed a list of texts in batches with retry + exponential backoff."""
    import time
    results: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for attempt in range(7):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                batch_vecs = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
                results.extend(batch_vecs)
                break
            except Exception as e:
                wait = 2 ** attempt
                if attempt < 6:
                    print(f"\n[embed retry {attempt+1}/7, wait {wait}s] {e}", flush=True)
                    time.sleep(wait)
                else:
                    raise
    return results


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return max(0.0, min(1.0, dot / (na * nb))) if na and nb else 0.0


def _evaluate_window(
    predictions: list[IdeaPrediction],
    train_papers: list[PaperRecord],
    future_papers: list[PaperRecord],
    paper_vecs: dict[str, list[float]],
    pred_vecs: list[list[float]],
    k: int,
) -> dict[str, Any]:
    """Compute hit@k, mrr, precision, recall using pre-computed embeddings.
    Novelty and diversity use token-level metrics (no embeddings needed)."""
    from live_idea_bench.similarity import _token_jaccard

    used_paper_ids: set[str] = set()
    matched_ranks: list[int] = []
    per_prediction_scores: list[dict[str, Any]] = []

    for rank, (pred, pvec) in enumerate(zip(predictions[:k], pred_vecs[:k]), start=1):
        best_score = 0.0
        best_paper_id = None
        all_scores = []
        for fp in future_papers:
            fvec = paper_vecs.get(fp.paper_id)
            if fvec is None:
                continue
            score = _cosine(pvec, fvec)
            all_scores.append(score)
            if score > best_score and fp.paper_id not in used_paper_ids:
                best_score = score
                best_paper_id = fp.paper_id
        is_match = best_score >= EMBED_THRESHOLD and best_paper_id is not None
        if is_match:
            matched_ranks.append(rank)
            used_paper_ids.add(best_paper_id)
        per_prediction_scores.append({
            "rank": rank,
            "title": pred.title,
            "best_score": round(best_score, 4),
            "best_paper_id": best_paper_id,
            "is_match": is_match,
            "score_mean": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
            "score_max": round(max(all_scores), 4) if all_scores else 0.0,
            "score_min": round(min(all_scores), 4) if all_scores else 0.0,
        })

    hit_at_k = 1.0 if matched_ranks else 0.0
    mrr = (1.0 / matched_ranks[0]) if matched_ranks else 0.0
    precision = len(matched_ranks) / k if k else 0.0
    recall = len(matched_ranks) / len(future_papers) if future_papers else 0.0

    # Novelty: avg dissimilarity from training papers (token jaccard)
    train_texts = [paper_text(p) for p in train_papers[-20:]]
    novelty_scores = []
    for pred in predictions[:k]:
        pt = _prediction_text(pred)
        sims = [_token_jaccard(pt, tt) for tt in train_texts]
        novelty_scores.append(1.0 - (max(sims) if sims else 0.0))
    novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0

    # Diversity: avg pairwise dissimilarity among predictions
    pred_texts = [_prediction_text(p) for p in predictions[:k]]
    diversity_scores = []
    for i in range(len(pred_texts)):
        for j in range(i + 1, len(pred_texts)):
            diversity_scores.append(1.0 - _token_jaccard(pred_texts[i], pred_texts[j]))
    diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0

    return {
        "hit_at_k": hit_at_k,
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "mrr": round(mrr, 4),
        "novelty": round(novelty, 4),
        "diversity": round(diversity, 4),
        "matched_prediction_ranks": matched_ranks,
        "matched_paper_ids": list(used_paper_ids),
        "per_prediction_scores": per_prediction_scores,
    }


def _load_predictions(raw: list[dict]) -> list[IdeaPrediction]:
    return [
        IdeaPrediction(
            rank=p["rank"],
            title=p["title"],
            rationale=p.get("rationale", ""),
            approach=p.get("approach", ""),
            score=p.get("score", 0.0),
            confidence=p.get("confidence", 0.0),
            key_terms=p.get("key_terms", []),
        )
        for p in raw
    ]


def main() -> int:
    global EMBED_THRESHOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--papers-dir", required=True)
    parser.add_argument("--output", default="/tmp/reeval_embedding.json")
    parser.add_argument("--threshold", type=float, default=EMBED_THRESHOLD,
                        help="Cosine similarity threshold for a match (default: 0.4)")
    args = parser.parse_args()

    EMBED_THRESHOLD = args.threshold

    api_key = os.environ.get("OPENAI_API_KEY", "no-key")
    client = openai.OpenAI(api_key=api_key)

    saved = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    cfg = saved.get("config", {})
    start_month = cfg.get("start_month", "2024-01")
    end_month = cfg.get("end_month", "2025-06")
    horizon_months = cfg.get("horizon_months", 3)
    top_k = cfg.get("top_k", 5)

    print(f"Loading papers from {args.papers_dir} ...")
    papers = load_papers_from_markdown(
        Path(args.papers_dir),
        start_month=start_month,
        end_month=end_month,
    )
    print(f"Loaded {len(papers)} papers")

    topics = load_topics()
    grouped = classify_papers_by_topic(papers, topics)

    topic_results: dict[str, Any] = {}
    total_windows = 0

    for topic in tqdm(topics, desc="Topics", unit="topic"):
        saved_topic = saved.get("topic_results", {}).get(topic.id, {})
        saved_bt = saved_topic.get("backtest")
        if not saved_bt or not saved_bt.get("windows"):
            tqdm.write(f"  [{topic.id}] no saved windows — skipped")
            topic_results[topic.id] = {"topic_name": topic.name, "paper_count": 0, "backtest": None}
            continue

        scoped = grouped.get(topic.id, [])
        if not scoped:
            topic_results[topic.id] = {"topic_name": topic.name, "paper_count": 0, "backtest": None}
            continue

        # --- Batch embed ALL papers in this topic at once ---
        tqdm.write(f"  [{topic.id}] embedding {len(scoped)} papers ...")
        paper_ids = [p.paper_id for p in scoped]
        paper_texts_list = [_sanitize(paper_text(p)[:MAX_CHARS_PER_TEXT]) for p in scoped]
        paper_vecs_list = _embed_batch(paper_texts_list, client)
        paper_vecs: dict[str, list[float]] = dict(zip(paper_ids, paper_vecs_list))

        saved_windows = saved_bt["windows"]
        windows_out = []

        for w in tqdm(saved_windows, desc=f"  {topic.id[:20]}", unit="win", leave=False):
            cutoff = w["cutoff_month"]
            cutoff_date = w["cutoff_date"]
            predictions = _load_predictions(w["predictions"])

            train, future, future_end, future_end_date = split_train_future_by_cutoff(
                papers=scoped,
                cutoff_month=cutoff,
                horizon_months=horizon_months,
                cutoff_date=cutoff_date,
            )

            if not future or not predictions:
                continue

            # Embed predictions for this window
            pred_texts = [_sanitize(_prediction_text(p)) for p in predictions[:top_k]]
            pred_vecs = _embed_batch(pred_texts, client)

            evaluation = _evaluate_window(
                predictions=predictions,
                train_papers=train,
                future_papers=future,
                paper_vecs=paper_vecs,
                pred_vecs=pred_vecs,
                k=top_k,
            )

            windows_out.append({
                "cutoff_month": cutoff,
                "future_end_month": future_end,
                "train_papers": len(train),
                "future_papers": len(future),
                "evaluation": evaluation,
            })

        if not windows_out:
            topic_results[topic.id] = {"topic_name": topic.name, "paper_count": len(scoped), "backtest": None}
            continue

        def avg(key: str) -> float:
            vals = [win["evaluation"][key] for win in windows_out]
            return round(sum(vals) / len(vals), 4)

        summary = {
            "windows": len(windows_out),
            "avg_hit_at_k": avg("hit_at_k"),
            "avg_recall_at_k": avg("recall_at_k"),
            "avg_precision_at_k": avg("precision_at_k"),
            "avg_mrr": avg("mrr"),
            "avg_novelty": avg("novelty"),
            "avg_diversity": avg("diversity"),
        }

        total_windows += len(windows_out)
        topic_results[topic.id] = {
            "topic_name": topic.name,
            "paper_count": len(scoped),
            "backtest": {"summary": summary, "windows": windows_out},
        }
        tqdm.write(
            f"  [{topic.id}] {len(windows_out)} windows — "
            f"hit@k={summary['avg_hit_at_k']:.4f}, mrr={summary['avg_mrr']:.4f}, "
            f"novelty={summary['avg_novelty']:.4f}, diversity={summary['avg_diversity']:.4f}"
        )

    # Weighted aggregate
    weighted: dict[str, float] = {}
    for metric in ("avg_hit_at_k", "avg_recall_at_k", "avg_precision_at_k", "avg_mrr", "avg_novelty", "avg_diversity"):
        num, den = 0.0, 0
        for tr in topic_results.values():
            bt = tr.get("backtest")
            if not bt:
                continue
            s = bt["summary"]
            w_count = s.get("windows", 0)
            if w_count > 0:
                num += s[metric] * w_count
                den += w_count
        weighted[metric] = round(num / den, 4) if den else 0.0

    print(f"\n{'='*60}")
    print(f"Aggregate ({total_windows} windows), threshold={EMBED_THRESHOLD}")
    for k, v in weighted.items():
        print(f"  {k}: {v:.4f}")

    out = {
        "mode": "reeval_batch_embedding",
        "embedding_model": EMBED_MODEL,
        "threshold": EMBED_THRESHOLD,
        "source_json": args.input_json,
        "config": cfg,
        "total_windows": total_windows,
        "aggregate_summary": weighted,
        "topic_results": topic_results,
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
