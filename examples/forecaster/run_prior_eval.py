#!/usr/bin/env python3
"""Sample innovations from a prior model and format for voyage evaluation.

Usage (trained checkpoint):
    python examples/run_prior_eval.py \
        --model-path output/prior_sft/final_checkpoint \
        --hindsight output/hindsight_samples.jsonl \
        --papers-dir data/csml/raw_markdown \
        --output-dir output/prior_sft/eval

Usage (untrained base model by HF id):
    python examples/run_prior_eval.py \
        --model-path Qwen/Qwen3.5-2B \
        --hindsight output/hindsight_samples.jsonl \
        --papers-dir data/csml/raw_markdown \
        --output-dir output/prior_baseline/eval
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from forecaster.config import InferenceConfig
from forecaster.hindsight.dataset_builder import load_hindsight_samples_jsonl
from forecaster.models import innovation_to_dict
from forecaster.prior.memory import build_memory_store_from_hindsight_samples
from forecaster.prior.sampler import sample_innovations
from live_idea_bench.backtest import split_train_future_by_cutoff
from live_idea_bench.config import load_topics
from live_idea_bench.papers import load_papers_from_markdown
from live_idea_bench.topics import classify_papers_by_topic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("prior_eval")


def main() -> int:
    p = argparse.ArgumentParser(description="Sample from prior and prepare for voyage eval.")
    p.add_argument("--model-path", required=True, help="Checkpoint dir or HF model id")
    p.add_argument("--hindsight", required=True, help="Path to hindsight_samples.jsonl")
    p.add_argument("--papers-dir", required=True, help="Papers markdown directory")
    p.add_argument("--output-dir", required=True, help="Output directory for predictions")
    p.add_argument("--num-candidates", type=int, default=16)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--horizon-months", type=int, default=3)
    p.add_argument("--start-month", default="2023-01")
    p.add_argument("--end-month", default="2025-06")
    args = p.parse_args()

    # Build memory at last cutoff
    samples = load_hindsight_samples_jsonl(args.hindsight)
    last_cutoff = sorted(set(s.cutoff_month for s in samples))[-1]
    log.info("Eval cutoff: %s", last_cutoff)

    memory = build_memory_store_from_hindsight_samples(samples, last_cutoff)
    memory = memory.decay_recency(last_cutoff)
    log.info("Memory: %d entries", memory.size)

    # Sample innovations
    cfg = InferenceConfig(
        num_candidates=args.num_candidates,
        prior_temperature=args.temperature,
        runtime_mode="flexible",
    )
    log.info("Sampling %d candidates from %s", args.num_candidates, args.model_path)
    innovations = sample_innovations(args.model_path, memory, cfg)
    log.info("Got %d valid innovations", len(innovations))

    if not innovations:
        log.error("No innovations sampled.")
        return 1

    # Save raw innovations
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sampled_innovations.json").write_text(
        json.dumps([innovation_to_dict(i) for i in innovations], indent=2, ensure_ascii=False)
    )

    # Build predictions in reeval_voyage format
    papers = load_papers_from_markdown(Path(args.papers_dir), start_month=args.start_month, end_month=args.end_month)
    log.info("Loaded %d papers", len(papers))

    topics = load_topics()
    grouped = classify_papers_by_topic(papers, topics)

    topic_results = {}
    for topic in topics:
        scoped = grouped.get(topic.id, [])
        if not scoped:
            topic_results[topic.id] = {"topic_name": topic.name, "paper_count": 0, "backtest": None}
            continue
        train, future, future_end, _ = split_train_future_by_cutoff(
            papers=scoped, cutoff_month=last_cutoff, horizon_months=args.horizon_months,
        )
        if not future:
            topic_results[topic.id] = {"topic_name": topic.name, "paper_count": len(scoped), "backtest": None}
            continue
        preds = [
            {
                "rank": rank,
                "title": f"{inn.operator}: {inn.base_direction}",
                "rationale": inn.gap,
                "approach": f"{inn.operator} on {inn.base_direction}",
                "score": 0.0,
                "confidence": 0.0,
                "key_terms": [inn.base_direction, inn.operator],
            }
            for rank, inn in enumerate(innovations[: args.top_k], start=1)
        ]
        topic_results[topic.id] = {
            "topic_name": topic.name,
            "paper_count": len(scoped),
            "backtest": {
                "summary": {"windows": 1},
                "windows": [
                    {
                        "cutoff_month": last_cutoff,
                        "cutoff_date": f"{last_cutoff}-01",
                        "future_end_month": future_end,
                        "train_papers": len(train),
                        "future_papers": len(future),
                        "predictions": preds,
                    }
                ],
            },
        }

    pred_path = out / "predictions_for_eval.json"
    pred_path.write_text(
        json.dumps(
            {
                "config": {
                    "start_month": args.start_month,
                    "end_month": args.end_month,
                    "horizon_months": args.horizon_months,
                    "top_k": args.top_k,
                },
                "topic_results": topic_results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    log.info("Saved predictions: %s", pred_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
