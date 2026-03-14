from __future__ import annotations

from typing import Any

from live_idea_bench.rl.config import PPOTrainConfig
from live_idea_bench.rl.trainers.base import PreparedRLContext, RLTrainerRunner, TrainerPreparedArtifacts
from live_idea_bench.rl.verl.runner import prepare_verl_artifacts, train_with_verl


class PPOTrainerRunner(RLTrainerRunner):
    trainer_name = "ppo"
    default_config_filename = "ppo_train.yaml"
    backend_name = "verl"

    def prepare(
        self,
        common_context: PreparedRLContext,
        *,
        trainer_config: PPOTrainConfig,
        **_: Any,
    ) -> TrainerPreparedArtifacts:
        return prepare_verl_artifacts(
            common_context,
            trainer_name=self.trainer_name,
            dry_run=trainer_config.dry_run,
        )

    def train(self, prepared_artifacts: TrainerPreparedArtifacts, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("output_dir", None)
        return train_with_verl(
            trainer_name=self.trainer_name,
            prepared_artifacts=prepared_artifacts,
            output_dir=str(prepared_artifacts.output_dir),
            **kwargs,
        )


def train_ppo_with_verl(
    dataset_rows: list[dict[str, Any]],
    config: PPOTrainConfig,
    *,
    model_name: str,
    predictor_config: str,
    output_dir: str,
    reward_config: Any,
    reward_config_path: str = "reward.yaml",
    trainer_config_path: str,
    selection_config: Any,
    selection_config_path: str,
    dataset_path: str | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    init_policy_path: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return train_with_verl(
        trainer_name="ppo",
        prepared_artifacts=None,
        dataset_rows=dataset_rows,
        dataset_path=dataset_path,
        dataset_metadata=dataset_metadata,
        config=config,
        model_name=model_name,
        predictor_config=predictor_config,
        output_dir=output_dir,
        reward_config=reward_config,
        reward_config_path=reward_config_path,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
        trainer_config_path=trainer_config_path,
        selection_config=selection_config,
        selection_config_path=selection_config_path,
        init_policy_path=init_policy_path,
        diagnostics=diagnostics,
    )
