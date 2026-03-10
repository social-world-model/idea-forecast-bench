from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from live_idea_bench.rl.config import DPOTrainConfig, GRPOTrainConfig, RewardConfig
from live_idea_bench.rl.io import _write_json, _write_jsonl
from live_idea_bench.rl.reward import build_grpo_reward_function


def _require_trl_stack() -> dict[str, Any]:
    try:
        datasets = importlib.import_module("datasets")
        peft = importlib.import_module("peft")
        trl = importlib.import_module("trl")
    except ImportError as exc:
        raise RuntimeError(
            "TRL training dependencies are not installed. Install torch, datasets, transformers, peft, "
            "accelerate, and trl to run non-dry-run RL training."
        ) from exc

    return {
        "Dataset": getattr(datasets, "Dataset"),
        "LoraConfig": getattr(peft, "LoraConfig"),
        "DPOConfig": getattr(trl, "DPOConfig"),
        "DPOTrainer": getattr(trl, "DPOTrainer"),
        "GRPOConfig": getattr(trl, "GRPOConfig"),
        "GRPOTrainer": getattr(trl, "GRPOTrainer"),
    }


def _build_manifest(
    *,
    stage: str,
    model_name: str,
    predictor_config: str,
    output_dir: Path,
    dataset_path: Path,
    dataset_size: int,
    dry_run: bool,
) -> dict[str, Any]:
    checkpoint_path = output_dir / "artifacts"
    return {
        "policy_manifest_version": 1,
        "policy_type": "policy_rl",
        "stage": stage,
        "inference_model_name": model_name,
        "predictor_config": predictor_config,
        "output_dir": str(output_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_size": dataset_size,
        "dry_run": dry_run,
    }


def train_dpo_with_trl(
    dataset_rows: list[dict[str, Any]],
    config: DPOTrainConfig,
    *,
    model_name: str,
    predictor_config: str,
    output_dir: str,
) -> dict[str, Any]:
    target_dir = Path(output_dir).resolve()
    dataset_path = target_dir / "dpo_dataset.jsonl"
    _write_jsonl(dataset_path, dataset_rows)
    manifest = _build_manifest(
        stage="dpo",
        model_name=model_name,
        predictor_config=predictor_config,
        output_dir=target_dir,
        dataset_path=dataset_path,
        dataset_size=len(dataset_rows),
        dry_run=config.dry_run,
    )

    if config.dry_run:
        _write_json(target_dir / "policy_manifest.json", manifest)
        return manifest

    deps = _require_trl_stack()
    peft_config = deps["LoraConfig"](
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    train_dataset = deps["Dataset"].from_list(
        [
            {
                "prompt": row["prompt"],
                "chosen": json.dumps(row["chosen"], ensure_ascii=False),
                "rejected": json.dumps(row["rejected"], ensure_ascii=False),
            }
            for row in dataset_rows
        ]
    )
    args = deps["DPOConfig"](
        output_dir=str(target_dir / "artifacts"),
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        beta=config.beta,
        max_length=config.max_length,
        logging_steps=config.logging_steps,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = deps["DPOTrainer"](
        model=model_name,
        args=args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(target_dir / "artifacts"))
    _write_json(target_dir / "policy_manifest.json", manifest)
    return manifest


def train_grpo_with_trl(
    dataset_rows: list[dict[str, Any]],
    config: GRPOTrainConfig,
    *,
    model_name: str,
    predictor_config: str,
    output_dir: str,
    reward_config: RewardConfig,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
) -> dict[str, Any]:
    target_dir = Path(output_dir).resolve()
    dataset_path = target_dir / "grpo_dataset.jsonl"
    _write_jsonl(dataset_path, dataset_rows)
    manifest = _build_manifest(
        stage="grpo",
        model_name=model_name,
        predictor_config=predictor_config,
        output_dir=target_dir,
        dataset_path=dataset_path,
        dataset_size=len(dataset_rows),
        dry_run=config.dry_run,
    )

    if config.dry_run:
        _write_json(target_dir / "policy_manifest.json", manifest)
        return manifest

    deps = _require_trl_stack()
    peft_config = deps["LoraConfig"](
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    train_dataset = deps["Dataset"].from_list(dataset_rows)
    args = deps["GRPOConfig"](
        output_dir=str(target_dir / "artifacts"),
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        use_vllm=config.use_vllm,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        logging_steps=config.logging_steps,
        report_to="none",
    )
    reward_func = build_grpo_reward_function(
        reward_config,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
        model_name=model_name,
    )
    trainer = deps["GRPOTrainer"](
        model=model_name,
        args=args,
        reward_funcs=reward_func,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(target_dir / "artifacts"))
    _write_json(target_dir / "policy_manifest.json", manifest)
    return manifest
