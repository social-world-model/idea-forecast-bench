from __future__ import annotations

from typing import Any

from forecaster.realization.config import GRPOTrainConfig
from forecaster.realization.trainers.base import (
    PreparedRLContext,
    RLTrainerRunner,
    TrainerPreparedArtifacts,
)
from forecaster.realization.trl.runner import prepare_trl_artifacts, train_with_trl


class GRPOTrainerRunner(RLTrainerRunner[GRPOTrainConfig]):
    trainer_name = "grpo"
    default_config_filename = "grpo_train.yaml"
    backend_name = "trl"

    def prepare(
        self,
        common_context: PreparedRLContext,
        *,
        trainer_config: GRPOTrainConfig,
        **_: Any,
    ) -> TrainerPreparedArtifacts:
        return prepare_trl_artifacts(
            common_context,
            trainer_name=self.trainer_name,
            dry_run=trainer_config.dry_run,
        )

    def train(
        self,
        prepared_artifacts: TrainerPreparedArtifacts,
        *,
        config: GRPOTrainConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        kwargs.pop("output_dir", None)
        return train_with_trl(
            trainer_name=self.trainer_name,
            prepared_artifacts=prepared_artifacts,
            output_dir=str(prepared_artifacts.output_dir),
            config=config,
            **kwargs,
        )
