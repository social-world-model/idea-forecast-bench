"""Scoring functions for joint inference."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from live_idea_bench.models import PaperRecord

from forecaster.models import Innovation
from forecaster.config import InferenceConfig, RealizationConfig
from forecaster.realization.realization_reward import compute_realization_reward

if TYPE_CHECKING:
    from forecaster.prior.memory import MemoryStore

_BASE_PRIOR_SCORE = -2.0
_LOG_EPSILON = 1e-6


def compute_prior_score(innovation: Innovation, memory_store: "MemoryStore") -> float:
    """Compute prior score for an innovation from memory.

    Proxy for log p_θ(z|M_t): uses the innovation's frequency and recency
    from the memory store as a log-likelihood proxy.

    If the innovation is not in memory, returns a small negative base score (-2.0).
    If it IS in memory, returns log(frequency * recency_score + 1).

    Returns:
        Float score (typically in range [-2, 3]).
    """
    for entry in memory_store.inventory.entries:
        inn = entry.innovation
        if (
            inn.base_direction == innovation.base_direction
            and inn.operator == innovation.operator
            and inn.gap == innovation.gap
        ):
            return math.log(entry.frequency * entry.recency_score + 1)
    return _BASE_PRIOR_SCORE


def compute_realization_score(
    proposal_text: str,
    innovation: Innovation,
    evidence: list[PaperRecord],
    config: RealizationConfig,
) -> float:
    """Compute realization score from the realization reward.

    Proxy for log p_ψ(y|z,X): wraps compute_realization_reward and
    scales to log space: log(reward + epsilon) where epsilon=1e-6.

    Returns:
        Float score in range [-14, 0] (log of [0,1] reward).
    """
    reward = compute_realization_reward(proposal_text, innovation, evidence, config)
    return math.log(reward + _LOG_EPSILON)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))


def compute_joint_score(
    prior_score: float,
    realization_score: float,
    config: InferenceConfig,
    *,
    popularity_bonus: float = 0.0,
) -> float:
    """Compute joint score as weighted combination.

    score = (prior_weight * sigmoid(prior_score)
             + realization_weight * sigmoid(realization_score)
             + popularity_weight * popularity_bonus) / total_weight

    When config.popularity_weight == 0.0 (default), this reduces to the
    original formula (prior + realization weights sum to 1.0).

    Args:
        popularity_bonus: A value in [0, 1] representing how popular/impactful
            the predicted topic is expected to be. Only applied when
            config.popularity_weight > 0.

    Returns:
        Float in [0, 1].
    """
    normalized_prior = _sigmoid(prior_score)
    normalized_realization = _sigmoid(realization_score)
    total_weight = config.prior_weight + config.realization_weight + config.popularity_weight
    if total_weight <= 0:
        return 0.0
    numerator = (
        config.prior_weight * normalized_prior
        + config.realization_weight * normalized_realization
        + config.popularity_weight * max(0.0, min(1.0, popularity_bonus))
    )
    return numerator / total_weight
