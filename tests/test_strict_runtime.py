"""Tests for the strict interactive realization runtime."""
from __future__ import annotations

from unittest.mock import patch

from forecaster.config import RealizationConfig
from forecaster.models import (
    Innovation,
    RealizationTrajectory,
    RealizationTrajectoryStep,
    SearchAction,
    SearchObservation,
    StrictRealizationResult,
)
from forecaster.realization.strict_runtime import (
    build_default_search_queries,
    run_strict_realization_rollout,
    score_strict_realization_trajectory,
)
from live_idea_bench.models import PaperRecord


def _make_innovation() -> Innovation:
    return Innovation(
        base_direction="graph reasoning",
        operator="extend",
        gap="long-horizon planning",
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


def test_build_default_search_queries_is_bounded_and_unique() -> None:
    queries = build_default_search_queries(_make_innovation(), max_search_steps=3)

    assert len(queries) == 3
    assert len(set(queries)) == len(queries)


def test_run_strict_realization_rollout_executes_search_select_and_finish_stepwise() -> None:
    papers = [
        _make_paper(
            "paper-1",
            "Graph Reasoning for Planning",
            "graph reasoning improves long-horizon planning with explicit search",
        ),
        _make_paper(
            "paper-2",
            "Benchmarking Agents",
            "benchmark dataset for agent evaluation",
        ),
    ]

    with patch(
        "forecaster.realization.strict_runtime.generate_strict_policy_completion",
        side_effect=[
            '{"action_type":"search","query":"graph reasoning long-horizon planning"}',
            '{"action_type":"select","paper_id":"paper-1"}',
            '{"action_type":"finish","proposal_text":"Strict Proposal\\nProposal body"}',
        ],
    ):
        trajectory, selected_evidence = run_strict_realization_rollout(
            _make_innovation(),
            papers,
            llm_client=object(),
            model="gpt-4o",
            realization_config=RealizationConfig(),
        )

    assert trajectory.invalid_reason is None
    assert trajectory.result is not None
    assert trajectory.result.search_queries == ("graph reasoning long-horizon planning",)
    assert trajectory.result.proposal_text == "Strict Proposal\nProposal body"
    assert [step.action.action_type for step in trajectory.steps] == ["search", "select", "finish"]
    assert [paper.paper_id for paper in selected_evidence] == ["paper-1"]


def test_run_strict_realization_rollout_feeds_previous_observation_only_after_first_step() -> None:
    papers = [
        _make_paper(
            "paper-1",
            "Graph Reasoning for Planning",
            "graph reasoning improves long-horizon planning with explicit search",
        ),
    ]

    with patch(
        "forecaster.realization.strict_runtime.generate_strict_policy_completion",
        side_effect=[
            '{"action_type":"search","query":"graph reasoning long-horizon planning"}',
            '{"action_type":"finish","proposal_text":"Strict Proposal\\nProposal body"}',
        ],
    ):
        trajectory, _ = run_strict_realization_rollout(
            _make_innovation(),
            papers,
            llm_client=object(),
            model="gpt-4o",
            realization_config=RealizationConfig(),
        )

    assert trajectory.invalid_reason is None
    assert "- previous_observation: []" in trajectory.steps[0].prompt_user
    assert "paper-1" in trajectory.steps[1].prompt_user
    assert "graph reasoning long-horizon planning" in trajectory.steps[1].prompt_user


def test_run_strict_realization_rollout_rejects_full_action_list_completion() -> None:
    papers = [
        _make_paper(
            "paper-1",
            "Graph Reasoning for Planning",
            "graph reasoning improves long-horizon planning with explicit search",
        ),
    ]

    with patch(
        "forecaster.realization.strict_runtime.generate_strict_policy_completion",
        return_value='{"actions":[{"action_type":"search","query":"graph reasoning long-horizon planning"}]}',
    ):
        trajectory, selected_evidence = run_strict_realization_rollout(
            _make_innovation(),
            papers,
            llm_client=object(),
            model="gpt-4o",
            realization_config=RealizationConfig(),
        )

    assert trajectory.invalid_reason == "full_action_list_invalid_in_strict_mode"
    assert trajectory.result is None
    assert selected_evidence == []


def test_score_strict_realization_trajectory_aggregates_per_step_conditionals() -> None:
    trajectory = RealizationTrajectory(
        innovation=_make_innovation(),
        steps=(
            RealizationTrajectoryStep(
                action=SearchAction(action_type="search", query="graph reasoning planning"),
                prompt_system="system",
                prompt_user="step-0",
                observation=(
                    SearchObservation(
                        paper_id="paper-1",
                        title="Graph Reasoning for Planning",
                        month="2024-01",
                        summary="summary one",
                    ),
                ),
                surfaced_paper_ids=("paper-1",),
                selected_evidence_ids=(),
            ),
            RealizationTrajectoryStep(
                action=SearchAction(action_type="finish", proposal_text="Strict Proposal\nProposal body"),
                prompt_system="system",
                prompt_user="step-1",
                observation=(),
                surfaced_paper_ids=("paper-1",),
                selected_evidence_ids=("paper-1",),
            ),
        ),
        result=StrictRealizationResult(
            selected_evidence_ids=("paper-1",),
            proposal_text="Strict Proposal\nProposal body",
            search_queries=("graph reasoning planning",),
        ),
    )

    with patch(
        "forecaster.realization.strict_runtime._load_local_model",
        return_value=(object(), object()),
    ), patch(
        "forecaster.realization.strict_runtime._score_conditioned_completion_with_model",
        side_effect=[-0.2, -0.4],
    ):
        score = score_strict_realization_trajectory(
            trajectory,
            model_name_or_path="/fake/model",
            base_model_name="base",
            score_normalization="per_token",
        )

    assert score == (-0.2 + -0.4) / 2


def test_score_strict_realization_trajectory_depends_on_observation_conditioning() -> None:
    base_trajectory = RealizationTrajectory(
        innovation=_make_innovation(),
        steps=(
            RealizationTrajectoryStep(
                action=SearchAction(action_type="finish", proposal_text="Strict Proposal\nProposal body"),
                prompt_system="system",
                prompt_user="step with paper-1",
                observation=(),
                surfaced_paper_ids=("paper-1",),
                selected_evidence_ids=("paper-1",),
            ),
        ),
        result=StrictRealizationResult(
            selected_evidence_ids=("paper-1",),
            proposal_text="Strict Proposal\nProposal body",
            search_queries=(),
        ),
    )
    other_trajectory = RealizationTrajectory(
        innovation=_make_innovation(),
        steps=(
            RealizationTrajectoryStep(
                action=SearchAction(action_type="finish", proposal_text="Strict Proposal\nProposal body"),
                prompt_system="system",
                prompt_user="step with paper-2",
                observation=(),
                surfaced_paper_ids=("paper-2",),
                selected_evidence_ids=("paper-2",),
            ),
        ),
        result=StrictRealizationResult(
            selected_evidence_ids=("paper-2",),
            proposal_text="Strict Proposal\nProposal body",
            search_queries=(),
        ),
    )

    def _score_side_effect(*, user_prompt, **kwargs):  # type: ignore[no-untyped-def]
        return -0.2 if "paper-1" in user_prompt else -0.8

    with patch(
        "forecaster.realization.strict_runtime._load_local_model",
        return_value=(object(), object()),
    ), patch(
        "forecaster.realization.strict_runtime._score_conditioned_completion_with_model",
        side_effect=_score_side_effect,
    ):
        first = score_strict_realization_trajectory(
            base_trajectory,
            model_name_or_path="/fake/model",
            base_model_name="base",
        )
        second = score_strict_realization_trajectory(
            other_trajectory,
            model_name_or_path="/fake/model",
            base_model_name="base",
        )

    assert first != second
