"""Joint inference algorithm (Algorithm 1 from the paper)."""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable

from live_idea_bench.models import PaperRecord

from forecaster.models import Innovation, JointCandidate, ScoredProposal
from forecaster.config import InferenceConfig, RealizationConfig
from forecaster.inference.scoring import (
    build_realization_scorer,
    compute_prior_score,
    compute_realization_score,
    compute_joint_score,
)
from forecaster.inference.deduplication import deduplicate_proposals
from forecaster.prior.memory import MemoryStore
from forecaster.prior.sampler import build_prior_scorer
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
    prior_model_path: str | None = None,
    realization_model_path: str | None = None,
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

    Args:
        realization_model_path: Optional path to the GRPO-trained realization checkpoint.
            When provided, proposal generation uses the trained local model (p_ψ)
            instead of the generic LLM client. Falls back to llm_client on failure.

    Returns:
        Top-K ScoredProposal objects, ranked and deduplicated.
    """
    candidates: list[JointCandidate] = []
    strict_runtime = str(inference_config.runtime_mode).strip().lower() == "strict_eval"
    prior_scorer: Callable[[Innovation], float] | None = None
    realization_scorer: Callable[[str, Innovation, list[PaperRecord]], float] | None = None
    if strict_runtime and inference_config.prior_score_method == "conditional_logprob" and not prior_model_path:
        raise ValueError("Strict joint inference requires a prior_model_path for conditional prior scoring.")
    if strict_runtime and inference_config.realization_score_method == "conditional_logprob" and not realization_model_path:
        raise ValueError(
            "Strict joint inference requires a realization_model_path for conditional realization scoring."
        )
    if (
        prior_model_path
        and inference_config.prior_score_method == "conditional_logprob"
    ):
        try:
            prior_scorer = build_prior_scorer(
                prior_model_path,
                memory_store,
                inference_config,
            )
        except Exception as exc:
            if strict_runtime:
                raise RuntimeError(
                    f"Strict prior scorer initialization failed: {exc}"
                ) from exc
            logger.warning(
                "Prior scorer unavailable (%s); falling back to heuristic memory scores.",
                exc,
            )
    if (
        realization_model_path
        and inference_config.realization_score_method == "conditional_logprob"
    ):
        try:
            realization_scorer = build_realization_scorer(
                realization_model_path,
                papers,
                realization_config,
                inference_config,
            )
        except Exception as exc:
            if strict_runtime:
                raise RuntimeError(
                    f"Strict realization scorer initialization failed: {exc}"
                ) from exc
            logger.warning(
                "Realization scorer unavailable (%s); falling back to paper reward scores.",
                exc,
            )

    for i, innovation in enumerate(innovations):
        try:
            prior_score_source = "heuristic_memory"
            if prior_scorer is not None:
                try:
                    prior_score = float(prior_scorer(innovation))
                    prior_score_source = "model_conditional_logprob"
                except Exception as exc:
                    if strict_runtime:
                        raise RuntimeError(
                            f"Strict prior scoring failed for innovation {i}: {exc}"
                        ) from exc
                    logger.warning(
                        "Prior scorer failed for innovation %d (%s); using heuristic memory score.",
                        i,
                        exc,
                    )
                    prior_score = compute_prior_score(innovation, memory_store)
                    prior_score_source = "heuristic_memory_fallback"
            else:
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
                context_papers=papers,
                llm_client=llm_client,
                model=model,
                config=realization_config,
                realization_model_path=realization_model_path,
            )

            realization_score_source = "paper_reward_log"
            if realization_scorer is not None:
                try:
                    realization_score = float(
                        realization_scorer(proposal_text, innovation, evidence)
                    )
                    realization_score_source = "model_conditional_logprob"
                except Exception as exc:
                    if strict_runtime:
                        raise RuntimeError(
                            f"Strict realization scoring failed for innovation {i}: {exc}"
                        ) from exc
                    logger.warning(
                        "Realization scorer failed for innovation %d (%s); using paper reward score.",
                        i,
                        exc,
                    )
                    realization_score = compute_realization_score(
                        proposal_text,
                        innovation,
                        evidence,
                        realization_config,
                    )
                    realization_score_source = "paper_reward_log_fallback"
            else:
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
                metadata={
                    "innovation": dataclasses.asdict(innovation),
                    "prior_score_source": prior_score_source,
                    "prior_score_method": inference_config.prior_score_method,
                    "realization_score_source": realization_score_source,
                    "realization_score_method": inference_config.realization_score_method,
                    "score_normalization": inference_config.score_normalization,
                    "proposal_title": proposal_text.splitlines()[0].strip() if proposal_text.strip() else "",
                    "evidence_paper_ids": list(tuple(p.paper_id for p in evidence)),
                },
            )
            candidates.append(candidate)

        except Exception as exc:
            if strict_runtime:
                raise RuntimeError(
                    f"Strict joint inference failed for innovation {i}: {exc}"
                ) from exc
            logger.warning(
                "Skipping innovation %d (%s) due to error: %s",
                i,
                innovation,
                exc,
            )

    # Compute joint scores once, sort descending
    scored: list[tuple[float, JointCandidate]] = [
        (
            compute_joint_score(
                c.prior_score, c.realization_score, inference_config,
                popularity_bonus=c.popularity_bonus,
            ),
            c,
        )
        for c in candidates
    ]
    sorted_scored = sorted(scored, key=lambda x: x[0], reverse=True)
    sorted_candidates = [c for _, c in sorted_scored]

    # Deduplicate
    deduped = deduplicate_proposals(sorted_candidates, threshold=inference_config.dedup_threshold)

    # Take top-K
    top_k_candidates = deduped[: inference_config.top_k]

    # Build ScoredProposal list with 1-indexed ranks (reuse precomputed scores)
    score_by_candidate = {id(c): s for s, c in sorted_scored}
    proposals: list[ScoredProposal] = []
    for rank, candidate in enumerate(top_k_candidates, start=1):
        joint_score = score_by_candidate.get(id(candidate), 0.0)
        proposal = ScoredProposal(
            innovation=candidate.innovation,
            proposal_text=candidate.proposal_text,
            prior_score=candidate.prior_score,
            realization_score=candidate.realization_score,
            joint_score=joint_score,
            evidence_paper_ids=candidate.evidence_paper_ids,
            rank=rank,
            popularity_bonus=candidate.popularity_bonus,
            metadata={
                **candidate.metadata,
                "joint_score_mode": inference_config.joint_score_mode,
            },
        )
        proposals.append(proposal)

    return proposals
