#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from forecaster.config import SFTTrainConfig
from forecaster.hindsight.dataset_builder import load_hindsight_samples_jsonl
from forecaster.prior.sft_dataset import build_sft_samples, save_sft_dataset
from forecaster.prior.trainer import train_prior

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("prior_sft")


def main() -> int:
    p = argparse.ArgumentParser(description="Train innovation prior via SFT.")
    p.add_argument("--hindsight", required=True, help="Path to hindsight_samples.jsonl")
    p.add_argument(
        "--output-dir", required=True, help="Output directory for checkpoint"
    )
    p.add_argument(
        "--model", default="qwen3.5-2b", help="Model alias (default: qwen3.5-2b)"
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--max-memory-entries", type=int, default=10)
    p.add_argument("--lora-r", type=int, default=16)
    args = p.parse_args()

    samples = load_hindsight_samples_jsonl(args.hindsight)
    log.info("Loaded %d hindsight samples", len(samples))

    sft_samples = build_sft_samples(samples, max_memory_entries=args.max_memory_entries)
    log.info("Built %d SFT samples", len(sft_samples))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_sft_dataset(sft_samples, f"{args.output_dir}/dataset.jsonl")

    config = SFTTrainConfig(
        model_alias=args.model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        max_memory_entries=args.max_memory_entries,
        lora_r=args.lora_r,
        output_dir=args.output_dir,
    )

    checkpoint = train_prior(sft_samples, config, output_dir=args.output_dir)
    log.info("Checkpoint: %s", checkpoint)

    meta = {
        "checkpoint_path": checkpoint,
        "model_alias": args.model,
        "num_samples": len(sft_samples),
    }
    Path(f"{args.output_dir}/train_result.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
