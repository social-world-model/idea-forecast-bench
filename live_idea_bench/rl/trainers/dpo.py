from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from live_idea_bench.rl.config import CandidateGenerationConfig, DPOTrainConfig, RewardConfig, SelectionConfig
from live_idea_bench.rl.dpo import build_dpo_pairs
from live_idea_bench.rl.io import _write_json, _write_jsonl
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


class DPOTrainerRunner(RLTrainerRunner):
    trainer_name = "dpo"
    default_config_filename = "dpo_train.yaml"

    def prepare(
        self,
        common_context: PreparedRLContext,
        *,
        model_name: str,
        candidate_config: CandidateGenerationConfig,
        reward_config: RewardConfig,
        trainer_config: DPOTrainConfig,
    ) -> TrainerPreparedArtifacts:
        from live_idea_bench.rl.pipeline import generate_episode_candidate_lists, serialize_episode_candidate_lists

        output_dir = common_context.shared_dir.parent / self.trainer_name
        candidate_lists = generate_episode_candidate_lists(
            common_context.papers,
            common_context.selected_episodes,
            model_name=model_name,
            candidate_config=candidate_config,
            reward_config=reward_config,
            similarity_config_path=common_context.similarity_config_path,
            runtime_config_path=common_context.runtime_config_path,
        )
        rollout_path = output_dir / "candidate_rollouts.json"
        _write_json(rollout_path, {"episodes": serialize_episode_candidate_lists(candidate_lists)})
        invalid_candidate_count = sum(
            1
            for episode_batch in candidate_lists
            for candidate in episode_batch.candidates
            if candidate.reward.invalid_completion or not candidate.predictions
        )
        dpo_pairs = build_dpo_pairs(candidate_lists, trainer_config)
        dataset_path = output_dir / "trainer_dataset.jsonl"
        _write_jsonl(dataset_path, dpo_pairs)
        return TrainerPreparedArtifacts(
            trainer_name=self.trainer_name,
            output_dir=output_dir,
            dataset_path=dataset_path,
            dataset_rows=dpo_pairs,
            metadata={
                "candidate_rollout_path": str(rollout_path.resolve()),
                "dpo_pair_count": len(dpo_pairs),
                "invalid_candidate_count": invalid_candidate_count,
            },
        )

    def train(self, prepared_artifacts: TrainerPreparedArtifacts, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("output_dir", None)
        kwargs.pop("reward_config", None)
        kwargs.pop("similarity_config_path", None)
        kwargs.pop("runtime_config_path", None)
        return train_dpo_with_trl(prepared_artifacts.dataset_rows, output_dir=str(prepared_artifacts.output_dir), **kwargs)


def train_dpo_with_trl(
    dataset_rows: list[dict[str, Any]],
    config: DPOTrainConfig,
    *,
    model_name: str,
    predictor_config: str,
    output_dir: str,
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
        trainer="dpo",
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
    args = create_trl_config(
        deps["DPOConfig"],
        {
            "output_dir": str(target_dir / "artifacts"),
            "per_device_train_batch_size": config.per_device_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "num_train_epochs": config.num_train_epochs,
            "learning_rate": config.learning_rate,
            "beta": config.beta,
            "max_length": config.max_length,
            "logging_steps": config.logging_steps,
            "report_to": "none",
            "remove_unused_columns": False,
        },
    )
    trainer = create_trl_trainer(
        deps["DPOTrainer"],
        {
            "model": training_model_name,
            "args": args,
            "train_dataset": train_dataset,
            "peft_config": peft_config,
        },
    )
    trainer.train()
    trainer.save_model(str(target_dir / "artifacts"))
    write_policy_manifest(target_dir / "policy_manifest.json", manifest)
    return manifest
