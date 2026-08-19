from __future__ import annotations

from forecaster.realization.trainers.base import RLTrainerRunner
from forecaster.realization.trainers.grpo import GRPOTrainerRunner


def create_trainer_runner(trainer_name: str) -> RLTrainerRunner:
    normalized = trainer_name.strip().lower()
    if normalized == "grpo":
        return GRPOTrainerRunner()
    raise ValueError(f"Unsupported trainer: {trainer_name}")
