"""Phase-7 forecast(X_<=t) → top-K ideas.

A thin composer over Phase-1's build_memory + Phase-3's sample_z + the
existing realization stack. Designed so each step is injectable:

    forecast(
        papers_before_t=...,
        cutoff_t=...,
        n_candidates=..., top_k=...,
        sampler=...,                       # (memory, n, t) -> List[Innovation]
        realizer=...,                      # (memory, z, papers) -> RealizationResult
        scorer=...,                        # (z, real) -> (prior, real_score) tuple
    ) -> List[ScoredForecast]

Defaults route to forecaster.foresight.prior_api.sample_z and a thin
wrapper around forecaster.realization.proposal_generator. Tests inject
stubs to keep the path LLM-free.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from forecaster.foresight.memory import build_memory
from forecaster.foresight.prior_api import sample_z
from forecaster.inference.deduplication import _jaccard_similarity
from forecaster.models import Innovation
from live_idea_bench.models import PaperRecord

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- types


@dataclass
class RealizedIdea:
    """Output of the realization step."""
    proposal_text: str
    evidence_paper_ids: list[str] = field(default_factory=list)


@dataclass
class ScoredForecast:
    """Final ranked forecast row."""
    rank: int
    innovation: Innovation
    proposal_text: str
    prior_score: float
    realization_score: float
    joint_score: float
    evidence_paper_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "rank": self.rank,
            "innovation": {
                "base_direction": self.innovation.base_direction,
                "operator": self.innovation.operator,
                "gap": self.innovation.gap,
            },
            "proposal_text": self.proposal_text,
            "prior_score": round(self.prior_score, 4),
            "realization_score": round(self.realization_score, 4),
            "joint_score": round(self.joint_score, 4),
            "evidence_paper_ids": list(self.evidence_paper_ids),
            "metadata": dict(self.metadata),
        }


SamplerFn = Callable[[str, int, float], list[Innovation]]
RealizerFn = Callable[[str, Innovation, Sequence[PaperRecord]], RealizedIdea]
ScorerFn = Callable[[Innovation, RealizedIdea, str], tuple[float, float]]


# --------------------------------------------------------------------------- forecast


def _default_sampler(memory: str, n: int, t: float) -> list[Innovation]:
    return sample_z(memory_text=memory, n=n, temperature=t)


def _default_scorer(z: Innovation, real: RealizedIdea, memory: str) -> tuple[float, float]:
    """Cheap default scoring: prior = constant 1.0 (uniform), realization = length proxy."""
    if not real.proposal_text:
        return 0.0, 0.0
    length_score = min(1.0, len(real.proposal_text) / 1000.0)
    return 1.0, length_score


def _joint_score(
    prior: float,
    real: float,
    *,
    prior_weight: float,
    realization_weight: float,
) -> float:
    return prior_weight * prior + realization_weight * real


def _dedup_inplace(
    proposals: list[ScoredForecast],
    threshold: float,
) -> list[ScoredForecast]:
    kept: list[ScoredForecast] = []
    for cand in proposals:
        if any(
            _jaccard_similarity(cand.proposal_text, k.proposal_text) > threshold
            for k in kept
        ):
            continue
        kept.append(cand)
    return kept


def forecast(
    papers_before_t: Sequence[PaperRecord],
    *,
    cutoff_t: str,
    n_candidates: int = 16,
    top_k: int = 5,
    temperature: float = 0.9,
    prior_weight: float = 0.4,
    realization_weight: float = 0.6,
    dedup_threshold: float = 0.8,
    sampler: SamplerFn | None = None,
    realizer: RealizerFn | None = None,
    scorer: ScorerFn | None = None,
    memory_kwargs: dict[str, Any] | None = None,
) -> list[ScoredForecast]:
    """Produce a deduplicated top-K forecast at cutoff_t.

    Steps (per the plan):
      1. M_t = build_memory(papers_before_t)
      2. Z   = sampler(M_t, n_candidates, temperature)
      3. for each z_i: y_i = realizer(M_t, z_i, papers_before_t)
      4. (prior_i, real_i) = scorer(z_i, y_i, M_t)
      5. joint = α·prior + β·real; dedup; return top-K.

    Required components default to the existing infrastructure but each is
    injectable for unit-test and ablation paths.
    """
    if not realizer:
        raise ValueError("forecast(): a `realizer` callable is required")
    sampler = sampler or _default_sampler
    scorer = scorer or _default_scorer
    memory = build_memory(list(papers_before_t), cutoff_t=cutoff_t, **(memory_kwargs or {}))

    z_list = sampler(memory, n_candidates, temperature)
    if not z_list:
        return []

    proposals: list[ScoredForecast] = []
    for z in z_list:
        try:
            real = realizer(memory, z, papers_before_t)
        except Exception as exc:  # pragma: no cover - protective
            logger.warning("realizer failed for z=%s: %s", z, exc, exc_info=True)
            continue
        prior_score, real_score = scorer(z, real, memory)
        joint = _joint_score(
            prior_score, real_score,
            prior_weight=prior_weight,
            realization_weight=realization_weight,
        )
        proposals.append(ScoredForecast(
            rank=0,                       # set after dedup + sort
            innovation=z,
            proposal_text=real.proposal_text,
            prior_score=prior_score,
            realization_score=real_score,
            joint_score=joint,
            evidence_paper_ids=list(real.evidence_paper_ids),
        ))

    proposals.sort(key=lambda p: -p.joint_score)
    kept = _dedup_inplace(proposals, threshold=dedup_threshold)
    kept = kept[:top_k]
    for i, p in enumerate(kept, start=1):
        p.rank = i
    return kept


__all__ = [
    "RealizedIdea",
    "ScoredForecast",
    "SamplerFn",
    "RealizerFn",
    "ScorerFn",
    "forecast",
]
