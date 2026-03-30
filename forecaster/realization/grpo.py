from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Protocol, runtime_checkable

from forecaster.realization.candidates import EpisodeCandidateLists
from forecaster.realization.config import GRPOTrainConfig
from forecaster.realization.reward import RLRewardEvaluation, spearman_correlation


@runtime_checkable
class HasAlignmentThreshold(Protocol):
    reward_alignment_threshold: float


@dataclass
class GRPOCandidateReward:
    prompt: str
    reward: float
    advantage: float
    reward_breakdown: dict[str, float]


@dataclass
class RewardAlignmentReport:
    correlation: float
    passed: bool


def compute_reward_alignment(
    evaluations: list[RLRewardEvaluation],
    config: HasAlignmentThreshold,
) -> RewardAlignmentReport:
    reward_scores = [evaluation.list_reward for evaluation in evaluations]
    benchmark_scores = [evaluation.benchmark_score for evaluation in evaluations]
    correlation = spearman_correlation(reward_scores, benchmark_scores)
    return RewardAlignmentReport(
        correlation=correlation,
        passed=correlation >= config.reward_alignment_threshold,
    )


def build_grpo_advantages(
    episodes: list[EpisodeCandidateLists],
    config: GRPOTrainConfig,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for episode_batch in episodes:
        rewards = [candidate.reward.list_reward for candidate in episode_batch.candidates]
        if not rewards:
            continue
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / max(1, len(rewards))
        stddev = sqrt(variance) if variance > 0 else 1.0

        candidates_payload: list[dict[str, Any]] = []
        for candidate in episode_batch.candidates:
            advantage = (candidate.reward.list_reward - mean_reward) / stddev
            candidates_payload.append(
                asdict(
                    GRPOCandidateReward(
                        prompt=episode_batch.prompt,
                        reward=round(candidate.reward.list_reward, 4),
                        advantage=round(advantage, 4),
                        reward_breakdown=candidate.reward.reward_breakdown,
                    )
                )
            )

        payloads.append(
            {
                "episode": asdict(episode_batch.episode),
                "prompt": episode_batch.prompt,
                "candidates": candidates_payload,
                "mean_reward": round(mean_reward, 4),
                "stddev_reward": round(stddev, 4),
            }
        )
    return payloads
