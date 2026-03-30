"""Strict interactive search environment shared by training and inference."""
from __future__ import annotations

import logging
from dataclasses import replace

from live_idea_bench.config import SimilarityConfig
from live_idea_bench.models import PaperRecord
from live_idea_bench.similarity import compute_similarity, paper_text

from forecaster.models import (
    Innovation,
    RealizationTrajectory,
    RealizationTrajectoryStep,
    STRICT_SEARCH_ENV_DEFAULTS,
    SearchAction,
    SearchObservation,
    SearchState,
    StrictRealizationResult,
)

logger = logging.getLogger(__name__)


def build_search_observation(paper: PaperRecord) -> SearchObservation:
    """Project a paper into the strict observation contract."""
    return SearchObservation(
        paper_id=paper.paper_id,
        title=paper.title,
        month=paper.month,
        summary=paper.summary,
    )


def initialize_search_state(innovation: Innovation) -> SearchState:
    """Start a new strict search state."""
    return SearchState(innovation=innovation)


def _extend_unique(existing: tuple[str, ...], values: tuple[str, ...]) -> tuple[str, ...]:
    ordered = list(existing)
    seen = set(existing)
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return tuple(ordered)


def _invalid_state(state: SearchState, reason: str, *, action: SearchAction | None = None) -> SearchState:
    history = state.action_history + ((action,) if action is not None else ())
    return replace(
        state,
        action_history=history,
        last_observation=(),
        done=True,
        invalid_reason=reason,
    )


def search_corpus(
    query: str,
    papers: list[PaperRecord],
    *,
    top_k: int = STRICT_SEARCH_ENV_DEFAULTS["top_k"],
    similarity_config: SimilarityConfig | None = None,
) -> tuple[SearchObservation, ...]:
    """Search the historical corpus and return strict observations."""
    normalized_query = str(query).strip()
    if not normalized_query or not papers:
        return ()
    config = similarity_config or SimilarityConfig(engine="hybrid")
    scored: list[tuple[float, str, str, PaperRecord]] = []
    for paper in papers:
        try:
            result = compute_similarity(normalized_query, paper_text(paper), config)
        except Exception as exc:
            logger.warning(
                "Strict search failed for paper %s and query %r: %s",
                paper.paper_id,
                normalized_query,
                exc,
            )
            continue
        scored.append((result.score, paper.month, paper.paper_id, paper))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    return tuple(build_search_observation(paper) for _, _, _, paper in scored[:top_k])


def apply_search_action(
    state: SearchState,
    action: SearchAction,
    papers: list[PaperRecord],
    *,
    top_k: int = STRICT_SEARCH_ENV_DEFAULTS["top_k"],
    max_search_steps: int = STRICT_SEARCH_ENV_DEFAULTS["max_search_steps"],
    max_selected_evidence: int = STRICT_SEARCH_ENV_DEFAULTS["max_selected_evidence"],
    similarity_config: SimilarityConfig | None = None,
) -> tuple[SearchState, tuple[SearchObservation, ...]]:
    """Advance the strict search state by one action."""
    if state.invalid_reason:
        return state, ()
    if state.done:
        return _invalid_state(state, "trajectory_already_finished", action=action), ()

    if action.action_type == "search":
        if state.step_index >= max_search_steps:
            return _invalid_state(state, "max_search_steps_exceeded", action=action), ()
        observation = search_corpus(
            action.query,
            papers,
            top_k=top_k,
            similarity_config=similarity_config,
        )
        surfaced_ids = _extend_unique(
            state.surfaced_paper_ids,
            tuple(item.paper_id for item in observation),
        )
        next_state = replace(
            state,
            step_index=state.step_index + 1,
            action_history=state.action_history + (action,),
            last_observation=observation,
            observation_history=state.observation_history + (observation,),
            surfaced_paper_ids=surfaced_ids,
            search_queries=state.search_queries + (action.query.strip(),),
        )
        return next_state, observation

    if action.action_type == "select":
        if action.paper_id not in state.surfaced_paper_ids:
            return _invalid_state(state, "paper_id_not_surfaced", action=action), ()
        if action.paper_id in state.selected_evidence_ids:
            return replace(
                state,
                action_history=state.action_history + (action,),
                last_observation=(),
            ), ()
        if len(state.selected_evidence_ids) >= max_selected_evidence:
            return _invalid_state(state, "max_selected_evidence_exceeded", action=action), ()
        return replace(
            state,
            action_history=state.action_history + (action,),
            last_observation=(),
            selected_evidence_ids=state.selected_evidence_ids + (action.paper_id,),
        ), ()

    if action.action_type == "finish":
        return replace(
            state,
            action_history=state.action_history + (action,),
            last_observation=(),
            proposal_text=action.proposal_text.strip(),
            done=True,
        ), ()

    return _invalid_state(state, f"unsupported_action_type:{action.action_type}", action=action), ()


def strict_result_from_state(state: SearchState) -> StrictRealizationResult | None:
    """Extract the strict completion object from a finished state."""
    if not state.done or state.invalid_reason or not state.proposal_text.strip():
        return None
    return StrictRealizationResult(
        selected_evidence_ids=state.selected_evidence_ids,
        proposal_text=state.proposal_text,
        search_queries=state.search_queries,
    )


def rollout_search_trajectory(
    innovation: Innovation,
    actions: list[SearchAction],
    papers: list[PaperRecord],
    *,
    top_k: int = STRICT_SEARCH_ENV_DEFAULTS["top_k"],
    max_search_steps: int = STRICT_SEARCH_ENV_DEFAULTS["max_search_steps"],
    max_selected_evidence: int = STRICT_SEARCH_ENV_DEFAULTS["max_selected_evidence"],
    similarity_config: SimilarityConfig | None = None,
) -> RealizationTrajectory:
    """Replay a sequence of policy actions inside the strict search environment."""
    state = initialize_search_state(innovation)
    steps: list[RealizationTrajectoryStep] = []
    for action in actions:
        prior_observation = state.last_observation
        state, observation = apply_search_action(
            state,
            action,
            papers,
            top_k=top_k,
            max_search_steps=max_search_steps,
            max_selected_evidence=max_selected_evidence,
            similarity_config=similarity_config,
        )
        steps.append(
            RealizationTrajectoryStep(
                action=action,
                step_index=len(steps),
                prior_observation=prior_observation,
                observation=observation,
                surfaced_paper_ids=state.surfaced_paper_ids,
                selected_evidence_ids=state.selected_evidence_ids,
            )
        )
        if state.done:
            break
    return RealizationTrajectory(
        innovation=innovation,
        steps=tuple(steps),
        result=strict_result_from_state(state),
        invalid_reason=state.invalid_reason,
    )
