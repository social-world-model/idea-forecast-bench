from __future__ import annotations

from dataclasses import dataclass, field

from forecaster.realization.episodes import RLEpisode
from forecaster.realization.reward import RLRewardEvaluation
from idea_forecast_bench.models import IdeaPrediction


@dataclass
class CandidateListSample:
    predictions: list[IdeaPrediction]
    reward: RLRewardEvaluation


@dataclass
class EpisodeCandidateLists:
    episode: RLEpisode
    prompt: str
    candidates: list[CandidateListSample] = field(default_factory=list)
