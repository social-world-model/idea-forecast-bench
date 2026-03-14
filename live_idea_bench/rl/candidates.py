from __future__ import annotations

from dataclasses import dataclass, field

from live_idea_bench.models import IdeaPrediction
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
