"""Joint inference algorithm (Algorithm 1 from the paper)."""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Callable
from typing import Any, cast

from forecaster.config import (
    InferenceConfig,
    RealizationConfig,
    strict_inference_score_contract,
    validate_inference_config,
)
from forecaster.inference.deduplication import deduplicate_proposals
from forecaster.inference.scoring import (
    build_realization_scorer,
    build_strict_realization_scorer,
    compute_joint_score,
    compute_prior_score,
    compute_realization_score,
    compute_strict_joint_score,
)
from forecaster.models import (
    Innovation,
    JointCandidate,
    RealizationTrajectory,
    ScoredProposal,
    realization_trajectory_to_dict,
)
from forecaster.prior.memory import MemoryStore
from forecaster.prior.sampler import build_prior_scorer
from forecaster.realization.evidence import retrieve_evidence
from forecaster.realization.proposal_generator import generate_proposal
from forecaster.realization.realization_reward import evaluate_strict_trajectory_reward
from forecaster.realization.strict_runtime import (
    run_strict_realization_rollout,
    serialize_strict_rollout_completion,
)
from idea_forecast_bench.models import PaperRecord

logger = logging.getLogger(__name__)
_LOG_EPSILON = 1e-6


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
    if strict_runtime:
        validate_inference_config(inference_config)
        if popularity_scorer is not None:
            raise ValueError("Strict joint inference does not allow popularity_scorer.")
    prior_scorer: Callable[[Innovation], float] | None = None
    realization_scorer: Callable[[str, Innovation, list[PaperRecord]], float] | None = (
        None
    )
    strict_realization_scorer: Callable[[RealizationTrajectory], float] | None = None
    if (
        strict_runtime
        and inference_config.prior_score_method == "conditional_logprob"
        and not prior_model_path
    ):
        raise ValueError(
            "Strict joint inference requires a prior_model_path for conditional prior scoring."
        )
    if (
        strict_runtime
        and inference_config.realization_score_method == "conditional_logprob"
        and not realization_model_path
    ):
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
            if strict_runtime:
                strict_realization_scorer = build_strict_realization_scorer(
                    realization_model_path,
                    inference_config,
                )
            else:
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

    # --- Batched fast path for non-strict mode with local realization model ---
    use_batch = (
        not strict_runtime
        and realization_model_path
        and not realization_scorer  # batch replaces per-item scoring
    )

    if use_batch:
        from forecaster.realization.proposal_generator import generate_proposals_batch

        # Step 1: Prior scores
        prior_scores: list[float] = []
        prior_sources: list[str] = []
        for innovation in innovations:
            if prior_scorer is not None:
                try:
                    prior_scores.append(float(prior_scorer(innovation)))
                    prior_sources.append("model_conditional_logprob")
                except Exception:
                    prior_scores.append(compute_prior_score(innovation, memory_store))
                    prior_sources.append("heuristic_memory_fallback")
            else:
                prior_scores.append(compute_prior_score(innovation, memory_store))
                prior_sources.append("heuristic_memory")

        # Step 2: Evidence retrieval (fast, CPU-bound)
        all_evidence: list[list[PaperRecord]] = []
        for innovation in innovations:
            evidence = retrieve_evidence(
                innovation,
                papers,
                top_k=realization_config.evidence_top_k,
                similarity_threshold=realization_config.evidence_similarity_threshold,
            )
            all_evidence.append(evidence)

        # Step 3: Batched proposal generation (single model.generate() call)
        innovations_and_evidence = list(zip(innovations, all_evidence, strict=False))
        try:
            proposal_texts = generate_proposals_batch(
                innovations_and_evidence,
                # `use_batch` is only truthy when `realization_model_path` is a
                # non-empty str, but mypy cannot narrow through that variable.
                cast(str, realization_model_path),
                realization_config,
                context_papers=papers,
            )
        except Exception as exc:
            logger.warning(
                "Batch proposal generation failed (%s); falling back to sequential.",
                exc,
            )
            proposal_texts = None

        if proposal_texts is None:
            # Fallback: sequential generation
            proposal_texts = []
            for innovation, evidence in innovations_and_evidence:
                try:
                    text = generate_proposal(
                        innovation=innovation,
                        evidence=evidence,
                        context_papers=papers,
                        llm_client=llm_client,
                        model=model,
                        config=realization_config,
                        realization_model_path=realization_model_path,
                    )
                    proposal_texts.append(text)
                except Exception as exc:
                    logger.warning("Proposal generation failed for innovation: %s", exc)
                    proposal_texts.append("")

        # Step 4: Score and build candidates
        for _i, (
            innovation,
            evidence,
            proposal_text,
            prior_score,
            prior_source,
        ) in enumerate(
            zip(
                innovations,
                all_evidence,
                proposal_texts,
                prior_scores,
                prior_sources,
                strict=False,
            )
        ):
            if not proposal_text.strip():
                continue
            realization_score = compute_realization_score(
                proposal_text,
                innovation,
                evidence,
                realization_config,
            )
            candidates.append(
                JointCandidate(
                    innovation=innovation,
                    prior_score=prior_score,
                    evidence_paper_ids=tuple(p.paper_id for p in evidence),
                    proposal_text=proposal_text,
                    realization_score=realization_score,
                    popularity_bonus=0.0,
                    metadata={
                        "innovation": dataclasses.asdict(innovation),
                        "prior_score_source": prior_source,
                        "prior_score_method": inference_config.prior_score_method,
                        "realization_score_source": "paper_reward_log",
                        "realization_score_method": inference_config.realization_score_method,
                        "score_normalization": inference_config.score_normalization,
                        "proposal_title": proposal_text.splitlines()[0].strip()
                        if proposal_text.strip()
                        else "",
                        "evidence_paper_ids": [p.paper_id for p in evidence],
                    },
                )
            )

    else:
        # --- Original sequential path (strict mode or LLM API fallback) ---
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

                search_queries: list[str] = []
                surfaced_paper_ids_by_step: list[list[str]] = []
                selected_evidence_ids: list[str] = []
                strict_rollout_payload = ""
                strict_trajectory_payload: dict[str, Any] = {}
                if strict_runtime:
                    trajectory, evidence = run_strict_realization_rollout(
                        innovation,
                        papers,
                        llm_client=llm_client,
                        model=model,
                        realization_config=realization_config,
                        realization_model_path=realization_model_path,
                    )
                    if trajectory.invalid_reason:
                        raise RuntimeError(
                            f"Strict realization rollout produced invalid trajectory: {trajectory.invalid_reason}"
                        )
                    if trajectory.result is None:
                        raise RuntimeError(
                            "Strict realization rollout did not finish with a proposal."
                        )
                    strict_rollout_payload = serialize_strict_rollout_completion(
                        trajectory
                    )
                    strict_trajectory_payload = realization_trajectory_to_dict(
                        trajectory
                    )
                    proposal_text = trajectory.result.proposal_text
                    search_queries = list(trajectory.result.search_queries)
                    surfaced_paper_ids_by_step = [
                        [obs.paper_id for obs in step.observation]
                        for step in trajectory.steps
                        if step.action.action_type == "search"
                    ]
                    selected_evidence_ids = list(
                        trajectory.result.selected_evidence_ids
                    )
                else:
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
                if strict_runtime and strict_realization_scorer is not None:
                    try:
                        realization_score = float(strict_realization_scorer(trajectory))
                        realization_score_source = "model_conditional_logprob"
                    except Exception as exc:
                        raise RuntimeError(
                            f"Strict realization scoring failed for innovation {i}: {exc}"
                        ) from exc
                elif realization_scorer is not None:
                    try:
                        realization_score = float(
                            realization_scorer(proposal_text, innovation, evidence)
                        )
                        realization_score_source = "model_conditional_logprob"
                    except Exception as exc:
                        logger.warning(
                            "Realization scorer failed for innovation %d (%s); using paper reward.",
                            i,
                            exc,
                        )
                        realization_score = compute_realization_score(
                            proposal_text, innovation, evidence, realization_config
                        )
                        realization_score_source = "paper_reward_log_fallback"
                else:
                    if strict_runtime:
                        strict_reward = evaluate_strict_trajectory_reward(
                            trajectory, papers, realization_config
                        )
                        if strict_reward.invalid_completion:
                            raise RuntimeError(
                                f"Strict realization reward rejected: {strict_reward.invalid_reason}"
                            )
                        realization_score = math.log(
                            strict_reward.total_reward + _LOG_EPSILON
                        )
                        realization_score_source = "strict_trajectory_reward_log"
                    else:
                        realization_score = compute_realization_score(
                            proposal_text, innovation, evidence, realization_config
                        )

                candidates.append(
                    JointCandidate(
                        innovation=innovation,
                        prior_score=prior_score,
                        evidence_paper_ids=tuple(p.paper_id for p in evidence),
                        proposal_text=proposal_text,
                        realization_score=realization_score,
                        popularity_bonus=0.0,
                        metadata={
                            "innovation": dataclasses.asdict(innovation),
                            "prior_score_source": prior_score_source,
                            "prior_score_method": inference_config.prior_score_method,
                            "realization_score_source": realization_score_source,
                            "realization_score_method": inference_config.realization_score_method,
                            "score_normalization": inference_config.score_normalization,
                            "proposal_title": proposal_text.splitlines()[0].strip()
                            if proposal_text.strip()
                            else "",
                            "evidence_paper_ids": [p.paper_id for p in evidence],
                            "search_queries": search_queries,
                            "surfaced_paper_ids_by_step": surfaced_paper_ids_by_step,
                            "selected_evidence_ids": selected_evidence_ids,
                            "policy_rollout": strict_rollout_payload,
                            "strict_trajectory": strict_trajectory_payload,
                        },
                    )
                )

            except Exception as exc:
                if strict_runtime:
                    raise RuntimeError(
                        f"Strict joint inference failed for innovation {i}: {exc}"
                    ) from exc
                logger.warning(
                    "Skipping innovation %d (%s) due to error: %s", i, innovation, exc
                )

    # Compute joint scores once, sort descending
    scored: list[tuple[float, JointCandidate]] = []
    for candidate in candidates:
        joint_score = (
            compute_strict_joint_score(
                candidate.prior_score,
                candidate.realization_score,
                inference_config,
            )
            if strict_runtime
            else compute_joint_score(
                candidate.prior_score,
                candidate.realization_score,
                inference_config,
                popularity_bonus=candidate.popularity_bonus,
            )
        )
        scored.append((joint_score, candidate))
    sorted_scored = sorted(scored, key=lambda x: x[0], reverse=True)
    sorted_candidates = [c for _, c in sorted_scored]

    # Deduplicate
    deduped = deduplicate_proposals(
        sorted_candidates, threshold=inference_config.dedup_threshold
    )

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
                "strict_score_contract": strict_inference_score_contract(
                    score_normalization=inference_config.score_normalization,
                    score_temperature=inference_config.score_temperature,
                )
                if strict_runtime
                else {},
            },
        )
        proposals.append(proposal)

    return proposals
