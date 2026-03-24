"""Joint inference algorithm (Algorithm 1 from the paper)."""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable

from live_idea_bench.models import PaperRecord

from forecaster.models import Innovation, JointCandidate, ScoredProposal
from forecaster.config import InferenceConfig, RealizationConfig
from forecaster.inference.scoring import (
    compute_prior_score,
    compute_realization_score,
    compute_joint_score,
)
from forecaster.inference.deduplication import deduplicate_proposals
from forecaster.prior.memory import MemoryStore
from forecaster.realization.evidence import retrieve_evidence
from forecaster.realization.proposal_generator import generate_proposal

logger = logging.getLogger(__name__)


def run_joint_inference(
    innovations: list[Innovation],
    papers: list[PaperRecord],
    memory_store: MemoryStore,
    llm_client: Any,
    model: str,
    inference_config: InferenceConfig,
    realization_config: RealizationConfig,
    *,
    popularity_scorer: Callable[[Innovation, list[PaperRecord]], float] | None = None,
) -> list[ScoredProposal]:
    """Run Algorithm 1: joint inference for idea forecasting.

    Takes pre-sampled innovations (from prior sampler or mock) and produces
    a ranked ScoredProposal list.

    Steps:
    1. For each innovation z_i:
       a. Compute prior score
       b. Retrieve evidence from papers
       c. Generate proposal
       d. Compute realization score
       e. Optionally compute popularity_bonus via popularity_scorer
       f. Compute joint score (includes popularity when configured)
    2. Sort by joint score (descending)
    3. Deduplicate
    4. Return top-K as ScoredProposal list with ranks

    Args:
        popularity_scorer: Optional callable(innovation, papers) → float in [0,1].
            When provided AND inference_config.popularity_weight > 0, the returned
            value is used as a popularity_bonus in compute_joint_score().

    Errors in individual proposals (LLM failures, etc.) are logged as warnings
    and skipped rather than failing the entire inference.

    Returns:
        Top-K ScoredProposal objects, ranked and deduplicated.
    """
    candidates: list[JointCandidate] = []

    for i, innovation in enumerate(innovations):
        try:
            prior_score = compute_prior_score(innovation, memory_store)

            evidence = retrieve_evidence(
                innovation,
                papers,
                top_k=realization_config.evidence_top_k,
                similarity_threshold=realization_config.evidence_similarity_threshold,
            )

            proposal_text = generate_proposal(
                innovation=innovation,
                evidence=evidence,
                llm_client=llm_client,
                model=model,
                config=realization_config,
            )

            realization_score = compute_realization_score(
                proposal_text,
                innovation,
                evidence,
                realization_config,
            )

            popularity_bonus = 0.0
            if popularity_scorer is not None and inference_config.popularity_weight > 0:
                try:
                    popularity_bonus = float(popularity_scorer(innovation, papers))
                except Exception as exc:
                    logger.warning("popularity_scorer failed for innovation %d: %s", i, exc)

            candidate = JointCandidate(
                innovation=innovation,
                prior_score=prior_score,
                evidence_paper_ids=tuple(p.paper_id for p in evidence),
                proposal_text=proposal_text,
                realization_score=realization_score,
                popularity_bonus=popularity_bonus,
            )
            candidates.append(candidate)

        except Exception as exc:
            logger.warning(
                "Skipping innovation %d (%s) due to error: %s",
                i,
                innovation,
                exc,
            )

    # Sort candidates by joint score descending
    def _joint_score_of(c: JointCandidate) -> float:
        return compute_joint_score(
            c.prior_score, c.realization_score, inference_config,
            popularity_bonus=c.popularity_bonus,
        )

    sorted_candidates = sorted(candidates, key=_joint_score_of, reverse=True)

    # Deduplicate
    deduped = deduplicate_proposals(sorted_candidates, threshold=inference_config.dedup_threshold)

    # Take top-K
    top_k = deduped[: inference_config.top_k]

    # Build ScoredProposal list with 1-indexed ranks
    proposals: list[ScoredProposal] = []
    for rank, candidate in enumerate(top_k, start=1):
        joint_score = _joint_score_of(candidate)
        proposal = ScoredProposal(
            innovation=candidate.innovation,
            proposal_text=candidate.proposal_text,
            prior_score=candidate.prior_score,
            realization_score=candidate.realization_score,
            joint_score=joint_score,
            evidence_paper_ids=candidate.evidence_paper_ids,
            rank=rank,
            popularity_bonus=candidate.popularity_bonus,
        )
        proposals.append(proposal)

    return proposals
