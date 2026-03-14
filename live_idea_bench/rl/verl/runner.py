from __future__ import annotations

from dataclasses import asdict, is_dataclass
import importlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from live_idea_bench.rl.config import GRPOTrainConfig, PPOTrainConfig, RewardConfig, SelectionConfig
from live_idea_bench.rl.io import _write_json
from live_idea_bench.rl.trainers.base import (
    PreparedRLContext,
    TrainerPreparedArtifacts,
    build_policy_manifest,
    write_policy_manifest,
)
from live_idea_bench.rl.verl.dataset import build_verl_dataset_rows, write_verl_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REWARD_FN_PATH = (Path(__file__).resolve().parent / "reward_fn.py").resolve()


def _require_verl_stack() -> None:
    try:
        importlib.import_module("verl")
        importlib.import_module("verl.trainer.main_ppo")
    except ImportError as exc:
        raise RuntimeError(
            "veRL training dependencies are not installed. Install verl, pandas, and pyarrow "
            "to run non-dry-run PPO/GRPO training."
        ) from exc


def prepare_verl_artifacts(
    common_context: PreparedRLContext,
    *,
    trainer_name: str,
    dry_run: bool,
) -> TrainerPreparedArtifacts:
    output_dir = common_context.shared_dir.parent / trainer_name
    dataset_rows = build_verl_dataset_rows(common_context.prompt_rows, data_source=f"live_idea_bench::{trainer_name}")
    dataset_path = output_dir / "trainer_dataset.parquet"
    dataset_artifacts = write_verl_dataset(dataset_rows, dataset_path, allow_preview_only=dry_run)
    return TrainerPreparedArtifacts(
        trainer_name=trainer_name,
        output_dir=output_dir,
        dataset_path=dataset_artifacts.primary_path,
        dataset_rows=dataset_rows,
        metadata={
            "backend": "verl",
            "dataset_format": "parquet" if dataset_artifacts.parquet_ready else "jsonl_preview",
            "parquet_ready": dataset_artifacts.parquet_ready,
            "parquet_engine": dataset_artifacts.parquet_engine or "",
            "prepared_parquet_path": str(dataset_artifacts.parquet_path.resolve()),
            "dataset_preview_path": str(dataset_artifacts.preview_path.resolve()),
            "dataset_row_count": dataset_artifacts.row_count,
        },
    )


def _coerce_config_payload(config: PPOTrainConfig | GRPOTrainConfig) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _build_overrides(
    *,
    trainer_name: str,
    config: PPOTrainConfig | GRPOTrainConfig,
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    reward_config_path: str,
    similarity_config_path: str,
    runtime_config_path: str | None,
    init_policy_path: str | None,
) -> dict[str, Any]:
    training_model_name = str(init_policy_path or model_name)
    batch_size = max(1, config.per_device_batch_size * config.gradient_accumulation_steps)
    overrides: dict[str, Any] = {
        "data.train_files": str(dataset_path.resolve()),
        "data.val_files": str(dataset_path.resolve()),
        "data.train_batch_size": batch_size,
        "data.val_batch_size": batch_size,
        "data.max_prompt_length": config.max_prompt_length,
        "data.max_response_length": config.max_completion_length,
        "data.prompt_key": "prompt",
        "data.return_raw_chat": False,
        "actor_rollout_ref.model.path": training_model_name,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.lora_rank": config.lora_r,
        "actor_rollout_ref.model.lora_alpha": config.lora_alpha,
        "actor_rollout_ref.model.target_modules": "all-linear",
        "actor_rollout_ref.actor.optim.lr": config.learning_rate,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": config.per_device_batch_size,
        "actor_rollout_ref.actor.ppo_mini_batch_size": batch_size,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.rollout.n": config.num_generations,
        "actor_rollout_ref.rollout.name": "vllm" if config.use_vllm else "hf",
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.gpu_memory_utilization": config.vllm_gpu_memory_utilization,
        "custom_reward_function.path": str(REWARD_FN_PATH),
        "custom_reward_function.name": "compute_score",
        "custom_reward_function.reward_kwargs.reward_config_path": reward_config_path,
        "custom_reward_function.reward_kwargs.similarity_config_path": similarity_config_path,
        "custom_reward_function.reward_kwargs.model_name": model_name,
        "trainer.project_name": "live-idea-bench",
        "trainer.experiment_name": f"{trainer_name}-{output_dir.name}",
        "trainer.default_local_dir": str((output_dir / "artifacts").resolve()),
        "trainer.total_epochs": config.num_train_epochs,
        "trainer.total_training_steps": max(1, config.num_train_epochs),
        "algorithm.adv_estimator": "grpo" if trainer_name == "grpo" else "gae",
    }
    if runtime_config_path:
        overrides["custom_reward_function.reward_kwargs.runtime_config_path"] = runtime_config_path
    if trainer_name == "ppo" and isinstance(config, PPOTrainConfig):
        overrides.update(
            {
                "critic.model.path": training_model_name,
                "critic.optim.lr": config.critic_learning_rate,
                "critic.ppo_micro_batch_size_per_gpu": config.critic_micro_batch_size,
                "algorithm.gamma": config.gamma,
                "algorithm.lam": config.lam,
                "algorithm.kl_ctrl.kl_coef": config.kl_coef,
            }
        )
    else:
        overrides.update(
            {
                "actor_rollout_ref.actor.use_kl_loss": True,
                "actor_rollout_ref.actor.kl_loss_coef": config.kl_coef,
                "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
            }
        )
    return overrides


def _command_from_overrides(overrides: dict[str, Any]) -> list[str]:
    command = [sys.executable, "-m", "verl.trainer.main_ppo"]
    for key, value in overrides.items():
        command.append(f"{key}={_hydra_value(value)}")
    return command


def train_with_verl(
    *,
    trainer_name: str,
    prepared_artifacts: TrainerPreparedArtifacts | None,
    config: PPOTrainConfig | GRPOTrainConfig,
    model_name: str,
    predictor_config: str,
    output_dir: str,
    reward_config: RewardConfig,
    reward_config_path: str = "reward.yaml",
    trainer_config_path: str,
    selection_config: SelectionConfig,
    selection_config_path: str,
    dataset_rows: list[dict[str, Any]] | None = None,
    dataset_path: str | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    init_policy_path: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_dir = Path(output_dir).resolve()
    dataset_rows = list(prepared_artifacts.dataset_rows if prepared_artifacts else (dataset_rows or []))
    dataset_metadata = dict(prepared_artifacts.metadata if prepared_artifacts else (dataset_metadata or {}))
    dataset_path_value = Path(
        dataset_path or (str(prepared_artifacts.dataset_path) if prepared_artifacts else str(target_dir / "trainer_dataset.parquet"))
    )
    prepared_parquet_path = Path(dataset_metadata.get("prepared_parquet_path", dataset_path_value))
    overrides = _build_overrides(
        trainer_name=trainer_name,
        config=config,
        model_name=model_name,
        dataset_path=prepared_parquet_path if dataset_metadata.get("parquet_ready", False) else dataset_path_value,
        output_dir=target_dir,
        reward_config_path=reward_config_path,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
        init_policy_path=init_policy_path,
    )
    launch_payload = {
        "module": "verl.trainer.main_ppo",
        "trainer": trainer_name,
        "backend": "verl",
        "overrides": overrides,
        "dataset_metadata": dataset_metadata,
        "config": _coerce_config_payload(config),
    }
    config_path = target_dir / "verl_launch_config.json"
    command_path = target_dir / "verl_launch_command.txt"
    _write_json(config_path, launch_payload)
    command = _command_from_overrides(overrides)
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")

    manifest = build_policy_manifest(
        trainer=trainer_name,
        base_model_name=model_name,
        inference_model_name=model_name,
        init_policy_path=init_policy_path,
        predictor_config=predictor_config,
        trainer_config_path=trainer_config_path,
        output_dir=target_dir,
        dataset_path=dataset_path_value,
        dataset_size=len(dataset_rows),
        dry_run=config.dry_run,
        selection_config_path=selection_config_path,
        candidate_pool_size=selection_config.candidate_pool_size,
        output_top_k=selection_config.output_top_k,
        diagnostics=diagnostics,
        backend="verl",
        launch_config_path=str(config_path.resolve()),
        launch_command=shlex.join(command),
        launch_command_path=str(command_path.resolve()),
        prepared_parquet_path=str(prepared_parquet_path.resolve()),
    )

    if config.dry_run:
        write_policy_manifest(target_dir / "policy_manifest.json", manifest)
        return manifest

    if not dataset_metadata.get("parquet_ready", False):
        raise RuntimeError(
            "veRL training requires a real Parquet dataset. Install pyarrow or fastparquet and rerun preparation."
        )

    _require_verl_stack()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        failure_manifest = {**manifest, "failed": True, "returncode": completed.returncode}
        write_policy_manifest(target_dir / "policy_manifest.json", failure_manifest)
        raise RuntimeError(f"veRL {trainer_name} training failed with exit code {completed.returncode}")

    write_policy_manifest(target_dir / "policy_manifest.json", manifest)
    return manifest
