from __future__ import annotations

from live_idea_bench.rl.trainers.base import RLTrainerRunner
from live_idea_bench.rl.trainers.dpo import DPOTrainerRunner
from live_idea_bench.rl.trainers.grpo import GRPOTrainerRunner
from live_idea_bench.rl.trainers.rloo import RLOOTrainerRunner


def create_trainer_runner(trainer_name: str) -> RLTrainerRunner:
    normalized = trainer_name.strip().lower()
    if normalized == "dpo":
        return DPOTrainerRunner()
    if normalized == "grpo":
        return GRPOTrainerRunner()
    if normalized == "rloo":
        return RLOOTrainerRunner()
    raise ValueError(f"Unsupported trainer: {trainer_name}")
