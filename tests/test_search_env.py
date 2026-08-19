"""Tests for the strict interactive search environment."""
from __future__ import annotations

from live_idea_bench.models import PaperRecord

from forecaster.models import (
    Innovation,
    STRICT_TRAJECTORY_SCHEMA_VERSION,
    SearchAction,
    realization_trajectory_to_dict,
)
from forecaster.realization.search_env import (
    apply_search_action,
    initialize_search_state,
    rollout_search_trajectory,
)


def _make_innovation() -> Innovation:
    return Innovation(
        base_direction="graph reasoning",
        operator="extend",
        gap="improve long-horizon planning",
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


def _make_papers() -> list[PaperRecord]:
    return [
        _make_paper(
            "paper-1",
            "Graph Reasoning for Long-Horizon Planning",
            "Graph reasoning improves planning with explicit search and structured evidence.",
        ),
        _make_paper(
            "paper-2",
            "Benchmarking Multimodal Models",
            "A benchmark dataset for multimodal evaluation and leaderboard analysis.",
        ),
    ]


def test_search_action_returns_strict_observation_fields() -> None:
    state = initialize_search_state(_make_innovation())
    next_state, observation = apply_search_action(
        state,
        SearchAction(action_type="search", query="graph reasoning planning"),
        _make_papers(),
    )

    assert next_state.step_index == 1
    assert len(observation) == 2
    assert observation[0].paper_id == "paper-1"
    assert set(observation[0].__dict__) == {"paper_id", "title", "month", "summary"}


def test_select_requires_previously_surfaced_paper() -> None:
    state = initialize_search_state(_make_innovation())
    next_state, observation = apply_search_action(
        state,
        SearchAction(action_type="select", paper_id="paper-1"),
        _make_papers(),
    )

    assert observation == ()
    assert next_state.invalid_reason == "paper_id_not_surfaced"


def test_search_step_limit_marks_state_invalid() -> None:
    state = initialize_search_state(_make_innovation())
    papers = _make_papers()
    for idx in range(4):
        state, _ = apply_search_action(
            state,
            SearchAction(action_type="search", query=f"graph reasoning {idx}"),
            papers,
            max_search_steps=3,
        )

    assert state.invalid_reason == "max_search_steps_exceeded"


def test_rollout_search_trajectory_serializes_queries_and_result() -> None:
    innovation = _make_innovation()
    trajectory = rollout_search_trajectory(
        innovation,
        [
            SearchAction(action_type="search", query="graph reasoning planning"),
            SearchAction(action_type="select", paper_id="paper-1"),
            SearchAction(action_type="finish", proposal_text="Proposal title\nProposal body"),
        ],
        _make_papers(),
    )

    assert trajectory.invalid_reason is None
    assert trajectory.result is not None
    assert trajectory.result.search_queries == ("graph reasoning planning",)
    assert trajectory.result.selected_evidence_ids == ("paper-1",)

    payload = realization_trajectory_to_dict(trajectory)
    assert payload["schema_version"] == STRICT_TRAJECTORY_SCHEMA_VERSION
    assert payload["result"]["selected_evidence_ids"] == ["paper-1"]
    assert payload["steps"][0]["observation"][0]["paper_id"] == "paper-1"
