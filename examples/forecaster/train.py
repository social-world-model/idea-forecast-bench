"""Single forecaster training entry point (METHOD §3.2 + §3.3, Unsloth + TRL).

Phase 1 — Prior SFT (METHOD §3.2): trains ``p_θ(z | M_t)`` via SFT on the
hindsight dataset, using the memory module ``M_t`` from
``forecaster.prior.memory``.

Phase 2 — Dataset prep: builds the realization GRPO prompt rows and writes
``trainer_dataset.jsonl``.

Phase 3 — Realization GRPO (METHOD §3.3): trains ``p_ψ(y | z, X_{≤t})`` via
GRPO with the three METHOD §3.3 verifiable rewards (evidence accuracy,
operator adherence, scientific coherence). Warm-starts from the Phase 1
adapter so the two phases compose into the factorized model from METHOD §3.1.

Usage:
    python examples/forecaster/train.py \\
        --model qwen3.5-2b \\
        --hindsight output/hindsight_samples.jsonl \\
        --papers data/csml_v2/raw_markdown \\
        --output-dir output/forecaster_qwen3.5-2b
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from forecaster.config import SFTTrainConfig  # noqa: E402
from forecaster.hindsight.dataset_builder import load_hindsight_samples_jsonl  # noqa: E402
from forecaster.prior.sft_dataset import build_sft_samples, save_sft_dataset  # noqa: E402
from forecaster.prior.trainer import train_prior  # noqa: E402
from forecaster.realization import (  # noqa: E402
    load_candidate_generation_config,
    load_episode_build_config,
    load_grpo_train_config,
    load_reward_config,
    load_selection_config,
)
from forecaster.realization.model_zoo import resolve_small_model  # noqa: E402
from forecaster.realization.pipeline import run_policy_rl_pipeline  # noqa: E402
from live_idea_bench.papers import load_papers_from_markdown  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("forecaster.train")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the forecaster (Prior SFT + Realization GRPO) with Unsloth."
    )
    p.add_argument("--model", default="qwen3.5-2b", help="Model preset alias (see model_zoo).")
    p.add_argument("--hindsight", required=True, help="Path to hindsight_samples.jsonl.")
    p.add_argument(
        "--papers",
        default=str(PROJECT_ROOT.parent / "md_mineru"),
        help="Directory of markdown papers (default: ../md_mineru relative to repo root).",
    )
    p.add_argument("--output-dir", required=True, help="Top-level output directory.")
    p.add_argument("--start-month", default=None, help="Lower bound month for loading papers.")
    p.add_argument("--end-month", default=None, help="Upper bound month for loading papers.")
    p.add_argument("--max-episodes", type=int, default=None, help="Cap GRPO episodes for quick runs.")
    p.add_argument(
        "--max-grpo-rows", type=int, default=None,
        help="Hard cap on the number of GRPO training rows (one row per hindsight sample). "
             "Use this for smoke tests — --max-episodes only caps episode count, not rows.",
    )
    # Prior SFT overrides
    p.add_argument("--prior-epochs", type=int, default=3)
    p.add_argument("--prior-lr", type=float, default=2e-5)
    p.add_argument("--prior-batch-size", type=int, default=4)
    p.add_argument("--prior-max-seq-length", type=int, default=4096)
    p.add_argument("--prior-lora-r", type=int, default=16)
    p.add_argument("--max-memory-entries", type=int, default=10)
    # GRPO overrides
    p.add_argument("--grpo-epochs", type=int, default=None, help="Override num_train_epochs for GRPO.")
    p.add_argument("--grpo-lr", type=float, default=None, help="Override learning_rate for GRPO.")
    p.add_argument("--num-generations", type=int, default=None, help="GRPO group size G (must fit in VRAM).")
    p.add_argument("--max-completion-length", type=int, default=None, help="Override GRPO max_completion_length.")
    p.add_argument(
        "--max-prompt-length", type=int, default=None,
        help="Override prompt length budget (used for FastLanguageModel max_seq_length sizing). "
             "TRL 1.0.0 GRPOConfig itself no longer takes this — it's only used for the model loader.",
    )
    # vLLM server-mode acceleration
    p.add_argument(
        "--use-vllm-server", action="store_true",
        help="Talk to a separately-started vLLM server (vllm_mode='server') for fast generation. "
             "The wrapper script scripts/forecaster/train.sh starts the server before invoking this script.",
    )
    p.add_argument("--vllm-server-host", default="localhost")
    p.add_argument("--vllm-server-port", type=int, default=8765)
    # Skips
    p.add_argument("--skip-prior-sft", action="store_true", help="Reuse existing prior_sft/final_checkpoint.")
    p.add_argument("--skip-grpo", action="store_true", help="Skip the realization GRPO phase.")
    p.add_argument("--skip-alignment-check", action="store_true", help="Skip GRPO online reward alignment gate.")
    return p


def _phase1_prior_sft(args: argparse.Namespace, output_dir: Path) -> str:
    sft_dir = output_dir / "prior_sft"
    final_ckpt = sft_dir / "final_checkpoint"
    if args.skip_prior_sft and final_ckpt.exists():
        log.info("Phase 1 skipped (reusing %s).", final_ckpt)
        return str(final_ckpt)

    log.info("Phase 1: loading hindsight samples from %s", args.hindsight)
    samples = load_hindsight_samples_jsonl(args.hindsight)
    log.info("Loaded %d hindsight samples.", len(samples))

    sft_samples = build_sft_samples(samples, max_memory_entries=args.max_memory_entries)
    log.info("Built %d SFT samples (memory-augmented per METHOD §3.2).", len(sft_samples))

    sft_dir.mkdir(parents=True, exist_ok=True)
    save_sft_dataset(sft_samples, str(sft_dir / "dataset.jsonl"))

    sft_config = SFTTrainConfig(
        model_alias=args.model,
        num_epochs=args.prior_epochs,
        learning_rate=args.prior_lr,
        per_device_batch_size=args.prior_batch_size,
        max_seq_length=args.prior_max_seq_length,
        max_memory_entries=args.max_memory_entries,
        lora_r=args.prior_lora_r,
        output_dir=str(sft_dir),
    )

    checkpoint = train_prior(sft_samples, sft_config, output_dir=str(sft_dir))
    log.info("Phase 1 done. Prior checkpoint: %s", checkpoint)

    meta = {
        "checkpoint_path": checkpoint,
        "model_alias": args.model,
        "num_samples": len(sft_samples),
    }
    (sft_dir / "train_result.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return checkpoint


def _phase2_3_grpo(
    args: argparse.Namespace,
    output_dir: Path,
    prior_ckpt: str | None,
) -> dict | None:
    if args.skip_grpo:
        log.info("Phase 2/3 skipped (--skip-grpo).")
        return None

    log.info("Phase 2: loading papers from %s", args.papers)
    papers = load_papers_from_markdown(
        input_dir=Path(args.papers),
        start_month=args.start_month,
        end_month=args.end_month,
    )
    if not papers:
        log.error("No papers loaded from %s", args.papers)
        return None

    episode_config = load_episode_build_config()
    if args.start_month:
        episode_config = replace(episode_config, start_month=args.start_month)
    if args.end_month:
        episode_config = replace(episode_config, end_month=args.end_month)
    candidate_config = load_candidate_generation_config()
    reward_config = load_reward_config()
    selection_config = load_selection_config()
    grpo_config = load_grpo_train_config()
    grpo_overrides: dict = {}
    if args.grpo_epochs is not None:
        grpo_overrides["num_train_epochs"] = args.grpo_epochs
    if args.grpo_lr is not None:
        grpo_overrides["learning_rate"] = args.grpo_lr
    if args.num_generations is not None:
        grpo_overrides["num_generations"] = args.num_generations
    if args.max_completion_length is not None:
        grpo_overrides["max_completion_length"] = args.max_completion_length
    if args.max_prompt_length is not None:
        grpo_overrides["max_prompt_length"] = args.max_prompt_length
    if grpo_overrides:
        grpo_config = replace(grpo_config, **grpo_overrides)

    samples = load_hindsight_samples_jsonl(args.hindsight)

    grpo_dir = output_dir / "realization_grpo"
    log.info("Phase 3: running GRPO with Unsloth (output=%s)", grpo_dir)
    manifest = run_policy_rl_pipeline(
        papers,
        model_name=resolve_small_model(args.model).model_id,
        output_dir=str(grpo_dir),
        episode_config=episode_config,
        candidate_config=candidate_config,
        reward_config=reward_config,
        selection_config=selection_config,
        trainer_config=grpo_config,
        trainer_config_path="grpo_train.yaml",
        selection_config_path="selection.yaml",
        max_episodes=args.max_episodes,
        init_policy_path=prior_ckpt,
        skip_alignment_check=args.skip_alignment_check,
        hindsight_samples=samples,
        max_grpo_rows=args.max_grpo_rows,
        use_vllm_server=args.use_vllm_server,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
    )
    log.info("Phase 3 done. Pipeline manifest: %s", grpo_dir / "pipeline_manifest.json")
    return manifest


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Forecaster training (Unsloth + TRL)")
    log.info("  model:      %s", args.model)
    log.info("  hindsight:  %s", args.hindsight)
    log.info("  papers:     %s", args.papers)
    log.info("  output:     %s", output_dir)
    log.info("=" * 60)

    prior_ckpt = _phase1_prior_sft(args, output_dir)
    manifest = _phase2_3_grpo(args, output_dir, prior_ckpt)

    log.info("Done.")
    if manifest:
        ckpt = manifest.get("trainer_policy_manifest_path", "")
        log.info("  GRPO policy manifest: %s", ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
