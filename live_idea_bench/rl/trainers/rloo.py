from __future__ import annotations

from typing import Any

from live_idea_bench.rl.config import RLOOTrainConfig
from live_idea_bench.rl.trainers.base import PreparedRLContext, RLTrainerRunner, TrainerPreparedArtifacts
from live_idea_bench.rl.verl.runner import prepare_verl_artifacts, train_with_verl


class RLOOTrainerRunner(RLTrainerRunner):
    trainer_name = "rloo"
    default_config_filename = "rloo_train.yaml"
    backend_name = "verl"

    def prepare(
        self,
        common_context: PreparedRLContext,
        *,
        trainer_config: RLOOTrainConfig,
        **_: Any,
    ) -> TrainerPreparedArtifacts:
        return prepare_verl_artifacts(
            common_context,
            trainer_name=self.trainer_name,
            dry_run=trainer_config.dry_run,
        )

    def train(self, prepared_artifacts: TrainerPreparedArtifacts, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("output_dir", None)
        return train_rloo_with_verl(
            prepared_artifacts.dataset_rows,
            output_dir=str(prepared_artifacts.output_dir),
            dataset_path=str(prepared_artifacts.dataset_path),
            dataset_metadata=prepared_artifacts.metadata,
            **kwargs,
        )


def train_rloo_with_verl(
    dataset_rows: list[dict[str, Any]],
    config: RLOOTrainConfig,
    *,
    output_dir: str,
    model_name: str,
    predictor_config: str,
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
        trainer_name="rloo",
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
