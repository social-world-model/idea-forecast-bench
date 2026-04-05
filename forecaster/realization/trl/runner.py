"""TRL-based GRPO training runner.

Single-process alternative to the veRL backend. Loads model directly to GPU
via device_map, avoiding the multi-GB CPU overhead of Ray. Produces the same
training results — both implement standard GRPO (Eq. 2 in the paper).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from forecaster.realization.config import GRPOTrainConfig
from forecaster.realization.io import _write_json
from forecaster.realization.trainers.base import (
    PreparedRLContext,
    TrainerPreparedArtifacts,
    build_policy_manifest,
    write_policy_manifest,
)
from forecaster.realization.verl.dataset import build_verl_dataset_rows

logger = logging.getLogger(__name__)


def _auto_batch_size(num_generations: int, max_completion_length: int, model_name: str = "") -> int:
    """Pick the largest batch size that fits in GPU memory.

    Called BEFORE model loading. Reserves memory for model weights + optimizer
    then fills the rest with generation batches.
    """
    import torch

    if not torch.cuda.is_available():
        return num_generations

    free_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

    # Reserve memory for model weights + ref copy + optimizer + gradients.
    # For a 2B param model in bf16: ~4GB model + ~4GB ref + ~4GB opt = ~12GB.
    # Use 15GB as a safe default; scale with free memory for larger models.
    reserved_mb = min(15_000, free_mb * 0.50)
    available_for_batch_mb = max(0, free_mb - reserved_mb)

    # Memory per sample includes KV cache, activations, lm_head logits AND
    # backward gradients. For Qwen with 151k vocab, backward through lm_head
    # allocates batch × seq × vocab × 4 bytes (fp32 grads). With 1024 tokens:
    # ~600MB per sample for forward+backward. Use 800MB as safe margin.
    mb_per_sample = max(400, max_completion_length * 0.8)

    max_samples = int(available_for_batch_mb / mb_per_sample)

    # Must be a multiple of num_generations
    batch_size = (max_samples // num_generations) * num_generations
    batch_size = max(batch_size, num_generations)  # at least one group
    batch_size = min(batch_size, num_generations * 2)  # cap at 2 groups

    print(f"[auto_batch] GPU free={free_mb:.0f}MB, reserved={reserved_mb:.0f}MB, "
          f"available={available_for_batch_mb:.0f}MB, {mb_per_sample:.0f}MB/sample → batch_size={batch_size}")
    return batch_size


def prepare_trl_artifacts(
    common_context: PreparedRLContext,
    *,
    trainer_name: str,
    dry_run: bool,
) -> TrainerPreparedArtifacts:
    """Prepare dataset for TRL training (JSONL format, no Parquet needed)."""
    output_dir = common_context.shared_dir.parent / trainer_name
    dataset_rows = build_verl_dataset_rows(
        common_context.prompt_rows, data_source=f"live_idea_bench::{trainer_name}"
    )
    dataset_path = output_dir / "trainer_dataset.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    from forecaster.realization.io import _write_jsonl
    _write_jsonl(dataset_path, dataset_rows)

    return TrainerPreparedArtifacts(
        trainer_name=trainer_name,
        output_dir=output_dir,
        dataset_path=dataset_path,
        dataset_rows=dataset_rows,
        metadata={
            "backend": "trl",
            "dataset_format": "jsonl",
            "dataset_row_count": len(dataset_rows),
        },
    )


def train_with_trl(
    *,
    trainer_name: str,
    prepared_artifacts: TrainerPreparedArtifacts | None,
    config: GRPOTrainConfig,
    model_name: str,
    predictor_config: str,
    output_dir: str,
    reward_config: Any,
    reward_config_path: str = "reward.yaml",
    realization_config_path: str = "realization.yaml",
    trainer_config_path: str,
    selection_config: Any,
    selection_config_path: str,
    dataset_rows: list[dict[str, Any]] | None = None,
    dataset_path: str | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    init_policy_path: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run GRPO training using TRL's GRPOTrainer (single-process, GPU-direct)."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = list(
        prepared_artifacts.dataset_rows if prepared_artifacts else (dataset_rows or [])
    )
    if not dataset_rows:
        raise ValueError("train_with_trl: dataset_rows is empty.")

    training_model_name = str(init_policy_path or model_name)

    # --- Build reward function closure ---
    from forecaster.realization.verl.reward_fn import compute_score

    def reward_fn(completions: list[str], **kwargs: Any) -> list[float]:
        """TRL-compatible reward function wrapping the existing compute_score."""
        prompts = kwargs.get("prompts", [""] * len(completions))
        extra_infos = kwargs.get("extra_info", ["{}"] * len(completions))
        scores = []
        for completion, extra_info in zip(completions, extra_infos):
            score = compute_score(
                data_source=f"live_idea_bench::{trainer_name}",
                solution_str=completion,
                ground_truth="",
                extra_info=extra_info,
                reward_config_path=reward_config_path,
                realization_config_path=realization_config_path,
                similarity_config_path=similarity_config_path,
                runtime_config_path=runtime_config_path,
                model_name=model_name,
            )
            scores.append(score)
        return scores

    # --- Build dataset ---
    # TRL 1.0 expects conversational format (list of message dicts) for proper
    # tokenization with variable-length prompts.
    ds_records = []
    for row in dataset_rows:
        ds_records.append({
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "extra_info": row.get("extra_info", "{}"),
        })
    dataset = Dataset.from_list(ds_records)

    # --- Configure training ---
    num_gen = config.num_generations
    batch_size = _auto_batch_size(num_gen, config.max_completion_length)
    logger.info("Auto-selected batch_size=%d (num_generations=%d)", batch_size, num_gen)

    grpo_config = GRPOConfig(
        output_dir=str(target_dir / "checkpoints"),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_generations=num_gen,
        max_completion_length=config.max_completion_length,
        beta=config.kl_coef,
        logging_steps=config.logging_steps,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    logger.info("Loading model %s with LoRA (r=%d, alpha=%d)...", training_model_name, config.lora_r, config.lora_alpha)

    # --- Load model + tokenizer explicitly ---
    # Qwen3.5 config registers as ForConditionalGeneration (VLM) by default.
    # Force AutoModelForCausalLM to get the text-only CausalLM head, avoiding
    # the VLM's 3D position embedding code path that breaks on text-only input.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        training_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(training_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Train ---
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO training with TRL (%d examples, %d epochs)...", len(dataset), config.num_train_epochs)
    trainer.train()

    # Save final checkpoint
    final_ckpt_dir = target_dir / "checkpoints" / "final_checkpoint"
    trainer.save_model(str(final_ckpt_dir))
    logger.info("Saved final checkpoint to %s", final_ckpt_dir)

    # --- Build manifest ---
    manifest = build_policy_manifest(
        trainer=trainer_name,
        base_model_name=model_name,
        inference_model_name=model_name,
        init_policy_path=init_policy_path,
        predictor_config=predictor_config,
        trainer_config_path=trainer_config_path,
        output_dir=target_dir,
        dataset_path=Path(dataset_rows[0].get("__path__", str(target_dir / "trainer_dataset.jsonl"))),
        dataset_size=len(dataset_rows),
        dry_run=config.dry_run,
        selection_config_path=selection_config_path,
        candidate_pool_size=selection_config.candidate_pool_size,
        output_top_k=selection_config.output_top_k,
        diagnostics=diagnostics,
        backend="trl",
    )
    write_policy_manifest(target_dir / "policy_manifest.json", manifest)
    return manifest
