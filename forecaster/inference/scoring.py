"""Scoring functions for joint inference."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

from live_idea_bench.models import PaperRecord

from forecaster.models import Innovation
from forecaster.config import InferenceConfig, RealizationConfig
from forecaster.realization.realization_reward import compute_realization_reward
from forecaster.realization.proposal_generator import score_local_proposal

if TYPE_CHECKING:
    from forecaster.prior.memory import MemoryStore

_BASE_PRIOR_SCORE = math.log(1e-6)
_LOG_EPSILON = 1e-6
# Minimum cosine-like similarity to treat a memory entry as "relevant"
_SEMANTIC_MATCH_THRESHOLD = 0.1
_PRIOR_RECENCY_WEIGHT = 0.45
_PRIOR_FREQUENCY_WEIGHT = 0.35
_PRIOR_UTILITY_WEIGHT = 0.20


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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalized_utility(value: float) -> float:
    """Map EMA utility into a bounded calibration range used by fallback scoring."""
    return _clamp01(0.5 + (0.5 * math.tanh(value)))


def compute_prior_score(innovation: Innovation, memory_store: "MemoryStore") -> float:
    """Compute the explicit heuristic fallback prior score for an innovation.

    The fallback is calibrated into a log-like scale so it is comparable to the
    main-path per-token conditional log-probability used by trained checkpoints.
    """
    if not memory_store.inventory.entries:
        return _BASE_PRIOR_SCORE

    query_text = _innovation_text(innovation)
    entries = list(memory_store.inventory.entries)
    max_freq = max(entry.frequency for entry in entries) or 1

    best_score = _BASE_PRIOR_SCORE
    for entry in entries:
        inn = entry.innovation
        if (
            inn.base_direction == innovation.base_direction
            and inn.operator == innovation.operator
            and inn.gap == innovation.gap
        ):
            similarity = 1.0
        else:
            entry_text = _innovation_text(inn)
            similarity = _semantic_similarity(query_text, entry_text)
            if similarity < _SEMANTIC_MATCH_THRESHOLD:
                continue

        normalized_frequency = entry.frequency / max_freq
        memory_strength = (
            (_PRIOR_RECENCY_WEIGHT * _clamp01(entry.recency_score))
            + (_PRIOR_FREQUENCY_WEIGHT * _clamp01(normalized_frequency))
            + (_PRIOR_UTILITY_WEIGHT * _normalized_utility(entry.utility_score))
        )
        candidate_mass = max(_LOG_EPSILON, similarity * _clamp01(memory_strength))
        best_score = max(best_score, math.log(candidate_mass))

    return best_score


def compute_realization_score(
    proposal_text: str,
    innovation: Innovation,
    evidence: list[PaperRecord],
    config: RealizationConfig,
) -> float:
    """Compute the paper-faithful fallback realization score.

    When no realization checkpoint scorer is available, the fallback uses the
    frozen proposal reward contract and maps it into log space.
    """
    reward = compute_realization_reward(proposal_text, innovation, evidence, config)
    return math.log(reward + _LOG_EPSILON)


def build_realization_scorer(
    realization_model_path: str,
    papers: list[PaperRecord],
    realization_config: RealizationConfig,
    inference_config: InferenceConfig,
) -> Callable[[str, Innovation, list[PaperRecord]], float]:
    """Build a scorer for log p(y | z, X) under the served realization artifact."""
    normalization = str(getattr(inference_config, "score_normalization", "per_token")).strip().lower()
    score_temperature = float(getattr(inference_config, "score_temperature", 1.0) or 1.0)

    def score(
        proposal_text: str,
        innovation: Innovation,
        evidence: list[PaperRecord],
    ) -> float:
        return score_local_proposal(
            innovation,
            evidence,
            proposal_text,
            context_papers=papers,
            model_name_or_path=realization_model_path,
            config=realization_config,
            score_normalization=normalization,
            score_temperature=score_temperature,
        )

    return score


def compute_joint_score(
    prior_score: float,
    realization_score: float,
    config: InferenceConfig,
    *,
    popularity_bonus: float = 0.0,
) -> float:
    """Compute the calibrated linear joint score from Algorithm 1."""
    total_weight = config.prior_weight + config.realization_weight + config.popularity_weight
    if total_weight <= 0:
        return float("-inf")
    numerator = (
        config.prior_weight * prior_score
        + config.realization_weight * realization_score
    )
    if config.popularity_weight > 0:
        popularity_score = math.log(_clamp01(popularity_bonus) + _LOG_EPSILON)
        numerator += config.popularity_weight * popularity_score
    if config.popularity_weight > 0 or not math.isclose(total_weight, 1.0):
        return numerator / total_weight
    return numerator
