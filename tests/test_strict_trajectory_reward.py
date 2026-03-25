"""Tests for strict trajectory-based realization reward."""
from __future__ import annotations

from live_idea_bench.models import PaperRecord

from forecaster.config import RealizationConfig
from forecaster.models import (
    Innovation,
    RealizationTrajectory,
    RealizationTrajectoryStep,
    SearchAction,
    SearchObservation,
    StrictRealizationResult,
)
from forecaster.realization.realization_reward import evaluate_strict_trajectory_reward


def _make_innovation() -> Innovation:
    return Innovation(
        base_direction="retrieval planning",
        operator="compose",
        gap="ground long-horizon agents",
    )


def _make_paper(paper_id: str, title: str, summary: str) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        month="2024-01",
        summary=summary,
        keywords=[],
        source_path="",
    )


def _make_observation(paper: PaperRecord) -> SearchObservation:
    return SearchObservation(
        paper_id=paper.paper_id,
        title=paper.title,
        month=paper.month,
        summary=paper.summary,
    )


def _make_trajectory(
    *,
    selected_evidence_ids: tuple[str, ...],
    surfaced_papers: tuple[PaperRecord, ...],
    proposal_text: str,
) -> RealizationTrajectory:
    innovation = _make_innovation()
    search_query = "retrieval planning grounded agents"
    return RealizationTrajectory(
        innovation=innovation,
        steps=(
            RealizationTrajectoryStep(
                action=SearchAction(action_type="search", query=search_query),
                observation=tuple(_make_observation(paper) for paper in surfaced_papers),
                selected_evidence_ids=(),
            ),
            RealizationTrajectoryStep(
                action=SearchAction(action_type="finish", proposal_text=proposal_text),
                observation=(),
                selected_evidence_ids=selected_evidence_ids,
            ),
        ),
        result=StrictRealizationResult(
            selected_evidence_ids=selected_evidence_ids,
            proposal_text=proposal_text,
            search_queries=(search_query,),
        ),
    )


def test_strict_trajectory_reward_changes_with_selected_evidence() -> None:
    good = _make_paper(
        "paper-good",
        "Grounded Retrieval Planning",
        "retrieval planning grounded long-horizon agents compose memory",
    )
    weak = _make_paper(
        "paper-weak",
        "Cooking Travel Notes",
        "cooking recipes and travel guides for weekend trips",
    )
    proposal_text = (
        "Grounded Retrieval Planning\n"
        "We compose retrieval and planning to ground long-horizon agents with memory and evaluation."
    )

    good_reward = evaluate_strict_trajectory_reward(
        _make_trajectory(
            selected_evidence_ids=("paper-good",),
            surfaced_papers=(good, weak),
            proposal_text=proposal_text,
        ),
        [good, weak],
        RealizationConfig(),
    )
    weak_reward = evaluate_strict_trajectory_reward(
        _make_trajectory(
            selected_evidence_ids=("paper-weak",),
            surfaced_papers=(good, weak),
            proposal_text=proposal_text,
        ),
        [good, weak],
        RealizationConfig(),
    )

    assert good_reward.evidence_quality > weak_reward.evidence_quality


def test_strict_trajectory_reward_has_zero_evidence_quality_without_selection() -> None:
    paper = _make_paper(
        "paper-good",
        "Grounded Retrieval Planning",
        "retrieval planning grounded long-horizon agents compose memory",
    )
    reward = evaluate_strict_trajectory_reward(
        _make_trajectory(
            selected_evidence_ids=(),
            surfaced_papers=(paper,),
            proposal_text="Grounded Retrieval Planning\nWe compose retrieval and planning with memory.",
        ),
        [paper],
        RealizationConfig(),
    )

    assert reward.evidence_quality == 0.0


def test_strict_trajectory_reward_operator_and_coherence_depend_only_on_proposal_text() -> None:
    good = _make_paper(
        "paper-good",
        "Grounded Retrieval Planning",
        "retrieval planning grounded long-horizon agents compose memory",
    )
    weak = _make_paper(
        "paper-weak",
        "Cooking Travel Notes",
        "cooking recipes and travel guides for weekend trips",
    )
    proposal_text = (
        "Grounded Retrieval Planning\n"
        "We compose retrieval and planning to ground long-horizon agents with memory and evaluation."
    )

    first = evaluate_strict_trajectory_reward(
        _make_trajectory(
            selected_evidence_ids=("paper-good",),
            surfaced_papers=(good, weak),
            proposal_text=proposal_text,
        ),
        [good, weak],
        RealizationConfig(),
    )
    second = evaluate_strict_trajectory_reward(
        _make_trajectory(
            selected_evidence_ids=("paper-weak",),
            surfaced_papers=(good, weak),
            proposal_text=proposal_text,
        ),
        [good, weak],
        RealizationConfig(),
    )

    assert first.operator_adherence == second.operator_adherence
    assert first.proposal_coherence == second.proposal_coherence


def test_strict_trajectory_reward_invalid_when_selected_id_not_surfaced() -> None:
    paper = _make_paper(
        "paper-good",
        "Grounded Retrieval Planning",
        "retrieval planning grounded long-horizon agents compose memory",
    )
    reward = evaluate_strict_trajectory_reward(
        _make_trajectory(
            selected_evidence_ids=("paper-missing",),
            surfaced_papers=(paper,),
            proposal_text="Grounded Retrieval Planning\nWe compose retrieval and planning with memory.",
        ),
        [paper],
        RealizationConfig(),
    )

    assert reward.invalid_completion is True
    assert reward.invalid_reason == "selected_paper_id_not_surfaced"
