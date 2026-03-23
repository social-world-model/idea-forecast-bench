from __future__ import annotations

from forecaster.realization.trainers.base import RLTrainerRunner
from forecaster.realization.trainers.grpo import GRPOTrainerRunner
from forecaster.realization.trainers.ppo import PPOTrainerRunner
from forecaster.realization.trainers.rloo import RLOOTrainerRunner


def create_trainer_runner(trainer_name: str) -> RLTrainerRunner:
    normalized = trainer_name.strip().lower()
    if normalized == "ppo":
        return PPOTrainerRunner()
    if normalized == "grpo":
        return GRPOTrainerRunner()
    if normalized == "rloo":
        return RLOOTrainerRunner()
    raise ValueError(f"Unsupported trainer: {trainer_name}")
