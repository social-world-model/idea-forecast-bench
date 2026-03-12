from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from live_idea_bench.rl.config import GRPOTrainConfig, RewardConfig, SelectionConfig
from live_idea_bench.rl.io import _write_jsonl

logger = logging.getLogger(__name__)
from live_idea_bench.rl.reward import build_online_rl_reward_function
from live_idea_bench.rl.trainers.base import (
    PreparedRLContext,
    RLTrainerRunner,
    TrainerPreparedArtifacts,
    _peft_config,
    _require_trl_stack,
    build_policy_manifest,
    create_trl_config,
    create_trl_trainer,
    write_policy_manifest,
)


class GRPOTrainerRunner(RLTrainerRunner):
    trainer_name = "grpo"
    default_config_filename = "grpo_train.yaml"

    def prepare(self, common_context: PreparedRLContext, **_: Any) -> TrainerPreparedArtifacts:
        output_dir = common_context.shared_dir.parent / self.trainer_name
        dataset_path = output_dir / "trainer_dataset.jsonl"
        _write_jsonl(dataset_path, common_context.prompt_rows)
        return TrainerPreparedArtifacts(
            trainer_name=self.trainer_name,
            output_dir=output_dir,
            dataset_path=dataset_path,
            dataset_rows=list(common_context.prompt_rows),
        )

    def train(self, prepared_artifacts: TrainerPreparedArtifacts, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("output_dir", None)
        return train_grpo_with_trl(prepared_artifacts.dataset_rows, output_dir=str(prepared_artifacts.output_dir), **kwargs)


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
    trainer_config_path: str,
    selection_config: SelectionConfig,
    selection_config_path: str,
    init_policy_path: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_dir = Path(output_dir).resolve()
    dataset_path = target_dir / "trainer_dataset.jsonl"
    _write_jsonl(dataset_path, dataset_rows)
    training_model_name = str(init_policy_path or model_name)
    manifest = build_policy_manifest(
        trainer="grpo",
        base_model_name=model_name,
        inference_model_name=model_name,
        init_policy_path=init_policy_path,
        predictor_config=predictor_config,
        trainer_config_path=trainer_config_path,
        output_dir=target_dir,
        dataset_path=dataset_path,
        dataset_size=len(dataset_rows),
        dry_run=config.dry_run,
        selection_config_path=selection_config_path,
        candidate_pool_size=selection_config.candidate_pool_size,
        output_top_k=selection_config.output_top_k,
        diagnostics=diagnostics,
    )

    if config.dry_run:
        write_policy_manifest(target_dir / "policy_manifest.json", manifest)
        return manifest

    deps = _require_trl_stack()
    peft_config = _peft_config(deps, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout)
    train_dataset = deps["Dataset"].from_list(dataset_rows)
    args = create_trl_config(
        deps["GRPOConfig"],
        {
            "output_dir": str(target_dir / "artifacts"),
            "per_device_train_batch_size": config.per_device_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "num_train_epochs": config.num_train_epochs,
            "learning_rate": config.learning_rate,
            "num_generations": config.num_generations,
            "max_completion_length": config.max_completion_length,
            "beta": config.beta,
            "loss_type": config.loss_type,
            "scale_rewards": config.scale_rewards,
            "mask_truncated_completions": config.mask_truncated_completions,
            "use_vllm": config.use_vllm,
            "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
            "logging_steps": config.logging_steps,
            "report_to": "none",
        },
    )
    reward_func = build_online_rl_reward_function(
        reward_config,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
        model_name=model_name,
    )
    trainer = create_trl_trainer(
        deps["GRPOTrainer"],
        {
            "model": training_model_name,
            "args": args,
            "reward_funcs": reward_func,
            "train_dataset": train_dataset,
            "peft_config": peft_config,
        },
    )
    try:
        trainer.train()
        trainer.save_model(str(target_dir / "artifacts"))
    except Exception as exc:
        logger.error("GRPO training failed: %s", exc, exc_info=True)
        write_policy_manifest(target_dir / "policy_manifest.json", {**manifest, "failed": True, "error": str(exc)})
        raise
    write_policy_manifest(target_dir / "policy_manifest.json", manifest)
    return manifest
