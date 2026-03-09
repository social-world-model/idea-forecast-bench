from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from live_idea_bench.rl.config import DPOTrainConfig, GRPOTrainConfig


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _require_trl_stack() -> dict[str, Any]:
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import DPOTrainer, GRPOTrainer
    except ImportError as exc:  # pragma: no cover - exercised only with external deps installed
        raise RuntimeError(
            "TRL training dependencies are not installed. Install datasets, transformers, peft, and trl "
            "to run non-dry-run RL training."
        ) from exc

    return {
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "TrainingArguments": TrainingArguments,
        "DPOTrainer": DPOTrainer,
        "GRPOTrainer": GRPOTrainer,
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
    tokenizer = deps["AutoTokenizer"].from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = deps["AutoModelForCausalLM"].from_pretrained(model_name)
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
    args = deps["TrainingArguments"](
        output_dir=str(target_dir / "artifacts"),
        per_device_train_batch_size=config.per_device_batch_size,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        logging_steps=1,
        remove_unused_columns=False,
    )
    trainer = deps["DPOTrainer"](
        model=model,
        ref_model=None,
        args=args,
        beta=config.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
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
    tokenizer = deps["AutoTokenizer"].from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = deps["AutoModelForCausalLM"].from_pretrained(model_name)
    peft_config = deps["LoraConfig"](
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    train_dataset = deps["Dataset"].from_list(dataset_rows)
    args = deps["TrainingArguments"](
        output_dir=str(target_dir / "artifacts"),
        per_device_train_batch_size=config.per_device_batch_size,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        logging_steps=1,
        remove_unused_columns=False,
    )
    trainer = deps["GRPOTrainer"](
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(target_dir / "artifacts"))
    _write_json(target_dir / "policy_manifest.json", manifest)
    return manifest
