from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from live_idea_bench.models import IdeaPrediction
from live_idea_bench.rl.config import DPOTrainConfig
from live_idea_bench.rl.episodes import RLEpisode
from live_idea_bench.rl.reward import RLRewardEvaluation


@dataclass
class CandidateListSample:
    predictions: list[IdeaPrediction]
    reward: RLRewardEvaluation


@dataclass
class EpisodeCandidateLists:
    episode: RLEpisode
    prompt: str
    candidates: list[CandidateListSample] = field(default_factory=list)


def _serialize_prediction_value(predictions: list[IdeaPrediction]) -> dict[str, Any] | list[dict[str, Any]]:
    if len(predictions) == 1:
        return asdict(predictions[0])
    return [asdict(prediction) for prediction in predictions]


def build_dpo_pairs(
    episodes: list[EpisodeCandidateLists],
    config: DPOTrainConfig,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    quantile = min(0.49, max(0.05, config.quantile_fraction))

    for episode_batch in episodes:
        ranked = sorted(
            episode_batch.candidates,
            key=lambda candidate: candidate.reward.list_reward,
            reverse=True,
        )
        if len(ranked) < 2:
            continue

        sample_count = max(1, int(len(ranked) * quantile))
        chosen_candidates = ranked[:sample_count]
        rejected_candidates = ranked[-sample_count:]

        for chosen, rejected in zip(chosen_candidates, reversed(rejected_candidates)):
            pairs.append(
                {
                    "episode": asdict(episode_batch.episode),
                    "prompt": episode_batch.prompt,
                    "chosen": _serialize_prediction_value(chosen.predictions),
                    "rejected": _serialize_prediction_value(rejected.predictions),
                    "chosen_reward": chosen.reward.list_reward,
                    "rejected_reward": rejected.reward.list_reward,
                    "chosen_breakdown": chosen.reward.reward_breakdown,
                    "rejected_breakdown": rejected.reward.reward_breakdown,
                }
            )
    return pairs
