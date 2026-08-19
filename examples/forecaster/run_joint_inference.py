#!/usr/bin/env python3
"""Run joint inference (Algorithm 1): prior sampling + realization + scoring.

All-local inference using trained prior (SFT) and realization (GRPO) checkpoints.
No LLM API calls — both models run on local GPU.

Produces proposals in reeval_voyage.py-compatible format.

Usage:
    python examples/run_joint_inference.py \
        --prior-checkpoint output/prior_sft/final_checkpoint \
        --realization-checkpoint output/realization_grpo/grpo \
        --hindsight output/hindsight_samples.jsonl \
        --papers-dir data/csml/raw_markdown \
        --output-dir output/joint_inference
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


from forecaster.config import InferenceConfig, RealizationConfig
from forecaster.hindsight.dataset_builder import load_hindsight_samples_jsonl
from forecaster.inference.algorithm import run_joint_inference
from forecaster.models import innovation_to_dict
from forecaster.prior.memory import build_memory_store_from_hindsight_samples
from forecaster.prior.sampler import sample_innovations
from live_idea_bench.backtest import split_train_future_by_cutoff
from live_idea_bench.config import load_topics
from live_idea_bench.papers import load_papers_from_markdown
from live_idea_bench.topics import classify_papers_by_topic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("joint_inference")


def main() -> int:
    p = argparse.ArgumentParser(description="Joint inference: prior + realization → proposals.")
    p.add_argument("--prior-checkpoint", required=True, help="Path to trained prior SFT checkpoint")
    p.add_argument("--realization-checkpoint", required=True, help="Path to trained GRPO realization checkpoint")
    p.add_argument("--hindsight", required=True, help="Path to hindsight_samples.jsonl")
    p.add_argument("--papers-dir", required=True, help="Papers markdown directory")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--num-candidates", type=int, default=16)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--horizon-months", type=int, default=3)
    p.add_argument("--start-month", default="2023-01")
    p.add_argument("--end-month", default="2025-06")
    args = p.parse_args()

    # Load hindsight and build memory
    samples = load_hindsight_samples_jsonl(args.hindsight)
    last_cutoff = sorted(set(s.cutoff_month for s in samples))[-1]
    log.info("Eval cutoff: %s", last_cutoff)

    memory = build_memory_store_from_hindsight_samples(samples, last_cutoff)
    memory = memory.decay_recency(last_cutoff)
    log.info("Memory: %d entries", memory.size)

    # Sample innovations from trained prior
    inf_cfg = InferenceConfig(
        num_candidates=args.num_candidates,
        prior_temperature=args.temperature,
        runtime_mode="flexible",
    )
    log.info("Sampling %d innovations from prior", args.num_candidates)
    innovations = sample_innovations(args.prior_checkpoint, memory, inf_cfg)
    log.info("Got %d valid innovations", len(innovations))
    if not innovations:
        log.error("No innovations sampled.")
        return 1

    # Load papers
    papers = load_papers_from_markdown(
        Path(args.papers_dir), start_month=args.start_month, end_month=args.end_month,
    )
    training_papers = [p for p in papers if p.month <= last_cutoff]
    log.info("Loaded %d papers (%d before cutoff)", len(papers), len(training_papers))

    # Run joint inference (Algorithm 1) — all local, no LLM API fallback
    real_cfg = RealizationConfig(allow_artifact_fallback_to_llm=False)

    log.info("Running joint inference with realization_checkpoint=%s", args.realization_checkpoint)
    proposals = run_joint_inference(
        innovations=innovations,
        papers=training_papers,
        memory_store=memory,
        llm_client=None,
        model="",
        inference_config=inf_cfg,
        realization_config=real_cfg,
        prior_model_path=args.prior_checkpoint,
        realization_model_path=args.realization_checkpoint,
    )
    log.info("Got %d scored proposals", len(proposals))

    # Save outputs
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save raw innovations
    (out / "sampled_innovations.json").write_text(json.dumps(
        [innovation_to_dict(i) for i in innovations], indent=2, ensure_ascii=False,
    ))

    # Save scored proposals
    proposals_data = [
        {
            "rank": sp.rank,
            "innovation": innovation_to_dict(sp.innovation),
            "proposal_text": sp.proposal_text,
            "prior_score": sp.prior_score,
            "realization_score": sp.realization_score,
            "joint_score": sp.joint_score,
            "evidence_paper_ids": list(sp.evidence_paper_ids),
        }
        for sp in proposals
    ]
    (out / "scored_proposals.json").write_text(json.dumps(proposals_data, indent=2, ensure_ascii=False))

    # Build predictions in reeval_voyage format
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
                "rank": i + 1,
                "title": sp.proposal_text.splitlines()[0] if sp.proposal_text.strip() else f"{sp.innovation.operator}: {sp.innovation.base_direction}",
                "rationale": sp.innovation.gap,
                "approach": f"{sp.innovation.operator} on {sp.innovation.base_direction}",
                "score": sp.joint_score,
                "confidence": sp.joint_score,
                "key_terms": [sp.innovation.base_direction, sp.innovation.operator],
            }
            for i, sp in enumerate(proposals[: args.top_k])
        ]
        topic_results[topic.id] = {
            "topic_name": topic.name,
            "paper_count": len(scoped),
            "backtest": {
                "summary": {"windows": 1},
                "windows": [{
                    "cutoff_month": last_cutoff,
                    "cutoff_date": f"{last_cutoff}-01",
                    "future_end_month": future_end,
                    "train_papers": len(train),
                    "future_papers": len(future),
                    "predictions": preds,
                }],
            },
        }

    pred_path = out / "predictions_for_eval.json"
    pred_path.write_text(json.dumps({
        "config": {
            "start_month": args.start_month, "end_month": args.end_month,
            "horizon_months": args.horizon_months, "top_k": args.top_k,
        },
        "topic_results": topic_results,
    }, indent=2, ensure_ascii=False))
    log.info("Saved predictions: %s", pred_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
