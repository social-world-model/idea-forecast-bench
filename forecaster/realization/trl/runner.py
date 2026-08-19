"""TRL-based GRPO training runner.

Single-process alternative to the veRL backend. Loads model directly to GPU
via device_map, avoiding the multi-GB CPU overhead of Ray. Produces the same
training results — both implement standard GRPO (Eq. 2 in the paper).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from forecaster.realization.config import GRPOTrainConfig
from forecaster.realization.trainers.base import (
    PreparedRLContext,
    TrainerPreparedArtifacts,
    build_policy_manifest,
    write_policy_manifest,
)
from forecaster.realization.verl.dataset import build_verl_dataset_rows

logger = logging.getLogger(__name__)


def _vllm_available() -> bool:
    """Check if vLLM is installed and compatible with TRL.

    Set DISABLE_VLLM=1 to force HF generate (useful when vLLM's worker
    process would exceed the cgroup RAM limit).
    """
    import os

    if os.environ.get("DISABLE_VLLM", "").strip() in ("1", "true", "yes"):
        logger.info("vLLM disabled via DISABLE_VLLM env var")
        return False
    try:
        import vllm  # noqa: F401
        from trl.import_utils import is_vllm_available

        return is_vllm_available()
    except Exception:
        return False


def _auto_batch_size(
    num_generations: int, max_completion_length: int, model_name: str = ""
) -> int:
    """Pick a safe batch size based on actual free GPU memory.

    Conservative: on shared GPUs other users may occupy most VRAM. The backward
    pass needs ~2-3x the forward memory, so we budget generously. Better to run
    slower than OOM.
    """
    import os

    import torch

    # Explicit override: BATCH_SIZE env wins over memory-based auto-sizing.
    # (Auto-sizing measures free VRAM *before* the colocate vLLM reserves its
    # share, so it over-commits when USE_VLLM=1. Allow a hard cap.)
    _bs_env = os.environ.get("BATCH_SIZE", "").strip()
    if _bs_env.isdigit() and int(_bs_env) > 0:
        return max(num_generations, (int(_bs_env) // num_generations) * num_generations)

    if not torch.cuda.is_available():
        return num_generations

    free_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

    # Reserve 60% of free memory for model weights + ref model + optimizer +
    # backward pass activation spikes. Previous 50% was too aggressive — the
    # backward pass on lm_head alone can spike batch×seq×vocab×4 bytes.
    reserved_mb = free_mb * 0.65
    available_for_batch_mb = max(0, free_mb - reserved_mb)

    # ~500MB per sample accounts for generation KV cache + backward pass peak.
    # Previous 250MB estimate caused OOM on shared GPUs.
    mb_per_sample = max(500, max_completion_length * 0.5)

    max_samples = int(available_for_batch_mb / mb_per_sample)

    # Must be a multiple of num_generations
    batch_size = (max_samples // num_generations) * num_generations
    batch_size = max(batch_size, num_generations)  # at least one group
    batch_size = min(batch_size, num_generations * 2)  # cap at 2 groups (not 4)

    # Allow override via env var
    override = os.environ.get("BATCH_SIZE", "").strip()
    if override:
        batch_size = int(override)

    print(
        f"[auto_batch] GPU free={free_mb:.0f}MB (total={total_mb:.0f}MB), "
        f"reserved={reserved_mb:.0f}MB, available={available_for_batch_mb:.0f}MB, "
        f"{mb_per_sample:.0f}MB/sample → batch_size={batch_size}"
    )
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
    # Routed through forecaster.foresight.trainer_wiring; switches between
    # the legacy composite reward and the Phase-4 foresight reward via
    # config.reward_mode ("legacy" | "foresight"). Keeping the call in
    # one place lets ablation runs flip a single field.
    from forecaster.foresight.trainer_wiring import make_reward_fn

    reward_fn = make_reward_fn(
        config,
        trainer_name=trainer_name,
        reward_config_path=reward_config_path,
        realization_config_path=realization_config_path,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
        model_name=model_name,
    )
    logger.info(
        "reward backend: %s (mode=%s)",
        reward_fn.__name__,
        getattr(config, "reward_mode", "legacy"),
    )

    # --- Build dataset ---
    # TRL 1.0 expects conversational format (list of message dicts) for proper
    # tokenization with variable-length prompts.
    ds_records = []
    for row in dataset_rows:
        ds_records.append(
            {
                "prompt": [{"role": "user", "content": row["prompt"]}],
                "extra_info": row.get("extra_info", "{}"),
            }
        )
    dataset = Dataset.from_list(ds_records)

    # --- Configure training ---
    num_gen = config.num_generations
    batch_size = _auto_batch_size(num_gen, config.max_completion_length)
    logger.info("Auto-selected batch_size=%d (num_generations=%d)", batch_size, num_gen)

    # With larger batch sizes, drop gradient accumulation to minimize steps
    grad_accum = max(1, config.gradient_accumulation_steps // (batch_size // num_gen))

    # vLLM is env-gated. On a single 47GB A6000 it OOMs (colocate needs TWO 9B
    # copies; server mode deadlocks on Qwen3.5 NCCL sync) so default OFF → HF
    # generate (slow but reliable). On a 96GB card (e.g. RTX Pro 6000) export
    # USE_VLLM=1 for colocate vLLM — ~10x faster generation. Tune memory via
    # VLLM_GPU_MEM_UTIL (fraction left to the vLLM engine after the trainer copy).
    if os.environ.get("USE_VLLM", "0") == "1":
        vllm_kwargs = {
            "use_vllm": True,
            "vllm_mode": os.environ.get("VLLM_MODE", "colocate"),
            "vllm_gpu_memory_utilization": float(
                os.environ.get("VLLM_GPU_MEM_UTIL", "0.45")
            ),
            # Cap the colocate vLLM context to what we actually use
            # (max_prompt_length + max_completion_length). The Qwen3.5 default
            # max_model_len is 262144, whose KV cache (~8GB) won't fit at low
            # gpu_memory_utilization -> "available KV cache memory" ValueError.
            "vllm_max_model_length": int(
                os.environ.get(
                    "VLLM_MAX_LEN",
                    str(config.max_prompt_length + config.max_completion_length + 1024),
                )
            ),
        }
        logger.info("vLLM ENABLED: %s", vllm_kwargs)
    else:
        vllm_kwargs = {"use_vllm": False}

    grpo_config = GRPOConfig(
        output_dir=str(target_dir / "checkpoints"),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=config.learning_rate,
        num_generations=num_gen,
        max_completion_length=config.max_completion_length,
        beta=config.kl_coef,
        logging_steps=config.logging_steps,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
        **vllm_kwargs,
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    logger.info(
        "Loading model %s with LoRA (r=%d, alpha=%d)...",
        training_model_name,
        config.lora_r,
        config.lora_alpha,
    )

    # --- Load model + tokenizer explicitly ---
    # Qwen3.5 config registers as ForConditionalGeneration (VLM) by default.
    # Force AutoModelForCausalLM to get the text-only CausalLM head, avoiding
    # the VLM's 3D position embedding code path that breaks on text-only input.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # SFT->GRPO: if init_policy_path points at a PEFT adapter (the SFT output),
    # load the real base model and CONTINUE that adapter through GRPO instead of
    # starting a fresh LoRA from base. Detection = adapter_config.json present.
    sft_adapter_dir = None
    if init_policy_path and (Path(init_policy_path) / "adapter_config.json").exists():
        sft_adapter_dir = init_policy_path
        load_from = model_name  # base weights; SFT adapter applied below
    else:
        load_from = training_model_name

    model = AutoModelForCausalLM.from_pretrained(
        load_from,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config_arg = lora_config
    if sft_adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, sft_adapter_dir, is_trainable=True)
        model.enable_input_require_grads()  # required for gradient checkpointing
        peft_config_arg = None  # train the existing SFT adapter; don't add a new one
        logger.info(
            "SFT->GRPO: continuing trainable SFT adapter from %s", sft_adapter_dir
        )

    # --- Train ---
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config_arg,
        processing_class=tokenizer,
    )

    logger.info(
        "Starting GRPO training with TRL (%d examples, %d epochs)...",
        len(dataset),
        config.num_train_epochs,
    )
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
        dataset_path=Path(
            dataset_rows[0].get("__path__", str(target_dir / "trainer_dataset.jsonl"))
        ),
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
