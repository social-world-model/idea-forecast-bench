"""Re-run a single backtest window and patch the saved reeval JSON.

Usage::

    python examples/rerun_window.py \\
        --reeval-json /tmp/reeval_t06.json \\
        --source-json /tmp/predictor_llm_domain_backtest_v4.json \\
        --papers-dir /tmp/papers_2024_2025 \\
        --topic optimizer \\
        --cutoff 2024-11 \\
        --output /tmp/reeval_t06_patched.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import openai
from tqdm import tqdm

from live_idea_bench.backtest import split_train_future_by_cutoff
from live_idea_bench.config import load_topics
from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.papers import load_papers_from_markdown
from live_idea_bench.strategy import create_strategy
from live_idea_bench.topics import classify_papers_by_topic
from examples.reeval_from_json import (
    _embed_batch, _evaluate_window, _prediction_text, _sanitize,
    MAX_CHARS_PER_TEXT, EMBED_THRESHOLD,
)
from live_idea_bench.similarity import paper_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reeval-json",  required=True, help="Existing reeval JSON to patch")
    parser.add_argument("--source-json",  required=True, help="Original backtest JSON (for config)")
    parser.add_argument("--papers-dir",   required=True)
    parser.add_argument("--topic",        required=True)
    parser.add_argument("--cutoff",       required=True, help="e.g. 2024-11")
    parser.add_argument("--model-name",   default=None)
    parser.add_argument("--output",       required=True)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "no-key")
    client = openai.OpenAI(api_key=api_key)

    source = json.loads(Path(args.source_json).read_text())
    cfg = source.get("config", {})
    start_month    = cfg.get("start_month",    "2024-01")
    end_month      = cfg.get("end_month",      "2025-06")
    horizon_months = cfg.get("horizon_months", 3)
    top_k          = cfg.get("top_k",          5)

    print(f"Loading papers from {args.papers_dir} ...")
    papers = load_papers_from_markdown(
        Path(args.papers_dir), start_month=start_month, end_month=end_month,
    )
    topics  = load_topics()
    grouped = classify_papers_by_topic(papers, topics)
    scoped  = grouped.get(args.topic, [])
    if not scoped:
        print(f"No papers found for topic {args.topic}")
        return 1

    # Find cutoff_date from original backtest
    orig_topic = source["topic_results"].get(args.topic, {})
    orig_bt    = orig_topic.get("backtest") or {}
    orig_win   = next(
        (w for w in orig_bt.get("windows", []) if w["cutoff_month"] == args.cutoff), None
    )
    if orig_win is None:
        print(f"Window cutoff={args.cutoff} not found in source JSON for topic {args.topic}")
        return 1
    cutoff_date = orig_win["cutoff_date"]

    train, future, future_end, future_end_date = split_train_future_by_cutoff(
        papers=scoped,
        cutoff_month=args.cutoff,
        horizon_months=horizon_months,
        cutoff_date=cutoff_date,
    )
    print(f"train={len(train)}, future={len(future)}")

    # Re-run prediction
    strategy = create_strategy(
        strategy_name="predictor_llm",
        model_name=args.model_name,
    )
    print(f"Generating predictions for cutoff={args.cutoff} ...")
    predictions = strategy.generate(train_papers=train, cutoff_month=args.cutoff, top_k=top_k)
    print(f"Got {len(predictions)} predictions:")
    for p in predictions:
        print(f"  [{p.rank}] {p.title}")

    # Embed papers for this topic
    print(f"Embedding {len(scoped)} papers ...")
    paper_ids   = [p.paper_id for p in scoped]
    paper_texts = [_sanitize(paper_text(p)[:MAX_CHARS_PER_TEXT]) for p in scoped]
    paper_vecs_list = _embed_batch(paper_texts, client)
    paper_vecs  = dict(zip(paper_ids, paper_vecs_list))

    # Embed predictions
    pred_texts = [_sanitize(_prediction_text(p)) for p in predictions[:top_k]]
    pred_vecs  = _embed_batch(pred_texts, client)

    evaluation = _evaluate_window(
        predictions=predictions,
        train_papers=train,
        future_papers=future,
        paper_vecs=paper_vecs,
        pred_vecs=pred_vecs,
        k=top_k,
    )
    print(f"hit@k={evaluation['hit_at_k']}, mrr={evaluation['mrr']:.4f}, "
          f"matched={evaluation['matched_prediction_ranks']}")

    new_window = {
        "cutoff_month":      args.cutoff,
        "future_end_month":  future_end,
        "train_papers":      len(train),
        "future_papers":     len(future),
        "evaluation":        evaluation,
    }

    # Patch reeval JSON
    reeval = json.loads(Path(args.reeval_json).read_text())
    topic_bt = reeval["topic_results"][args.topic]["backtest"]

    # Remove old incomplete window and insert new one
    old_windows = topic_bt["windows"]
    old_window  = next((w for w in old_windows if w["cutoff_month"] == args.cutoff), None)
    if old_window:
        old_windows.remove(old_window)
        print(f"Replaced window with {len(old_window['evaluation']['per_prediction_scores'])} predictions "
              f"→ now {len(new_window['evaluation']['per_prediction_scores'])} predictions")
    old_windows.append(new_window)
    old_windows.sort(key=lambda w: w["cutoff_month"])

    # Recompute topic summary
    def avg(key: str) -> float:
        vals = [w["evaluation"][key] for w in old_windows]
        return round(sum(vals) / len(vals), 4)

    topic_bt["summary"] = {
        "windows":             len(old_windows),
        "avg_hit_at_k":        avg("hit_at_k"),
        "avg_recall_at_k":     avg("recall_at_k"),
        "avg_precision_at_k":  avg("precision_at_k"),
        "avg_mrr":             avg("mrr"),
        "avg_novelty":         avg("novelty"),
        "avg_diversity":       avg("diversity"),
    }

    Path(args.output).write_text(json.dumps(reeval, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved patched reeval JSON → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
