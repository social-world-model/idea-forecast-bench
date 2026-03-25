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
# Minimum cosine-like similarity to treat a memory entry as "relevant"
_SEMANTIC_MATCH_THRESHOLD = 0.1


def _innovation_text(innovation: Innovation) -> str:
    """Render an innovation triple as a single string for semantic comparison."""
    return f"{innovation.base_direction} {innovation.operator}: {innovation.gap}"


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute a lightweight semantic similarity score using hybrid text similarity.

    Uses the same hybrid engine as evidence retrieval (difflib + keyword overlap),
    avoiding any heavy ML dependency. Returns a value in [0, 1].
    """
    from live_idea_bench.similarity import _hybrid_similarity, _keyword_overlap
    semantic = _hybrid_similarity(text_a, text_b)
    keyword = _keyword_overlap(text_a, text_b)
    return max(semantic, keyword)


def compute_prior_score(innovation: Innovation, memory_store: "MemoryStore") -> float:
    """Compute prior score for an innovation from memory.

    Proxy for log p_θ(z|M_t): combines frequency/recency from the memory store
    with semantic similarity to the innovation text.

    Scoring strategy:
    - Exact match: returns log(frequency * recency_score + 1) directly.
    - Semantic match (similarity >= threshold): returns a weighted combination of
      the memory entry's log-score scaled by similarity.
    - No match: returns _BASE_PRIOR_SCORE (-2.0).

    This replaces the original exact-string-only matching which caused sampled
    innovations (rarely exact copies of memory entries) to always score -2.0.

    Returns:
        Float score (typically in range [-2, 3]).
    """
    if not memory_store.inventory.entries:
        return _BASE_PRIOR_SCORE

    query_text = _innovation_text(innovation)

    best_score = _BASE_PRIOR_SCORE
    for entry in memory_store.inventory.entries:
        inn = entry.innovation
        # Exact match: full credit
        if (
            inn.base_direction == innovation.base_direction
            and inn.operator == innovation.operator
            and inn.gap == innovation.gap
        ):
            return math.log(entry.frequency * entry.recency_score + 1)

        # Semantic match: scale memory score by similarity
        entry_text = _innovation_text(inn)
        similarity = _semantic_similarity(query_text, entry_text)
        if similarity >= _SEMANTIC_MATCH_THRESHOLD:
            entry_log_score = math.log(entry.frequency * entry.recency_score + 1)
            candidate = similarity * entry_log_score
            if candidate > best_score:
                best_score = candidate

    return best_score


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
