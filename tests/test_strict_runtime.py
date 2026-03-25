"""Tests for the strict interactive realization runtime."""
from __future__ import annotations

from unittest.mock import patch

from live_idea_bench.models import PaperRecord

from forecaster.config import RealizationConfig
from forecaster.models import Innovation
from forecaster.realization.strict_runtime import (
    build_default_search_queries,
    run_strict_realization_rollout,
)


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


def test_run_strict_realization_rollout_executes_search_and_finish() -> None:
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
        return_value='{"actions":[{"action_type":"search","query":"graph reasoning long-horizon planning"},{"action_type":"search","query":"graph reasoning extend"},{"action_type":"search","query":"extend long-horizon planning"},{"action_type":"select","paper_id":"paper-1"},{"action_type":"finish","proposal_text":"Strict Proposal\\nProposal body"}]}',
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
    assert len(trajectory.result.search_queries) == 3
    assert trajectory.result.proposal_text == "Strict Proposal\nProposal body"
    assert trajectory.steps[-1].action.action_type == "finish"
    assert [paper.paper_id for paper in selected_evidence] == list(trajectory.result.selected_evidence_ids)
