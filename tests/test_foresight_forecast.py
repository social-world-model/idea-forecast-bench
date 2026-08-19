"""Tests for the Phase-7 forecast() composer."""

from __future__ import annotations

import pytest

from forecaster.foresight.forecast import (
    RealizedIdea,
    forecast,
)
from forecaster.models import Innovation
from live_idea_bench.models import PaperRecord


def _paper(pid: str, date: str, topic: str) -> PaperRecord:
    return PaperRecord(
        paper_id=pid,
        title=f"Title {pid}",
        month=date[:7],
        summary=f"Summary about {topic} work.",
        keywords=[topic],
        source_path="",
        published_date=date,
        metadata={"topic_id": topic},
    )


def test_forecast_returns_top_k_ranked():
    papers = [
        _paper("p1", "2024-04-15", "rag"),
        _paper("p2", "2024-05-15", "rag"),
        _paper("p3", "2024-05-20", "agents"),
    ]

    def sampler(memory: str, n: int, t: float):
        # 4 distinct innovations.
        return [
            Innovation("rag", "extend", "x1"),
            Innovation("rag", "compose", "x2"),
            Innovation("agents", "transfer", "x3"),
            Innovation("agents", "benchmark", "x4"),
        ][:n]

    def realizer(memory: str, z: Innovation, papers):
        # Distinct proposal text per innovation.
        return RealizedIdea(
            proposal_text=f"Proposal for {z.operator} on {z.base_direction}",
            evidence_paper_ids=[p.paper_id for p in papers],
        )

    def scorer(z: Innovation, real: RealizedIdea, memory: str):
        # Prior: 1.0 always; realization: based on the operator letter for variance.
        bonus = {"extend": 0.4, "compose": 0.5, "transfer": 0.3, "benchmark": 0.2}
        return 1.0, bonus.get(z.operator, 0.1)

    out = forecast(
        papers,
        cutoff_t="2024-06-30",
        n_candidates=4,
        top_k=2,
        sampler=sampler,
        realizer=realizer,
        scorer=scorer,
        prior_weight=0.0,
        realization_weight=1.0,
    )
    assert [s.rank for s in out] == [1, 2]
    assert out[0].innovation.operator == "compose"  # highest real_score
    assert out[1].innovation.operator == "extend"
    assert out[0].joint_score >= out[1].joint_score
    assert out[0].evidence_paper_ids == ["p1", "p2", "p3"]


def test_forecast_deduplicates_near_identical_proposals():
    papers = [_paper("p1", "2024-04-15", "rag")]

    def sampler(memory: str, n: int, t: float):
        return [Innovation("rag", "extend", f"variant_{i}") for i in range(4)]

    def realizer(memory: str, z: Innovation, papers):
        # Almost identical proposal text — should dedupe down to one.
        return RealizedIdea(
            proposal_text="We extend retrieval with a new mechanism.",
            evidence_paper_ids=[p.paper_id for p in papers],
        )

    out = forecast(
        papers,
        cutoff_t="2024-06-30",
        n_candidates=4,
        top_k=4,
        sampler=sampler,
        realizer=realizer,
        dedup_threshold=0.5,
    )
    assert len(out) == 1
    assert out[0].rank == 1


def test_forecast_empty_when_sampler_returns_nothing():
    papers = [_paper("p1", "2024-04-15", "rag")]

    out = forecast(
        papers,
        cutoff_t="2024-06-30",
        n_candidates=4,
        top_k=4,
        sampler=lambda *_a: [],
        realizer=lambda *_a: RealizedIdea(proposal_text=""),
    )
    assert out == []


def test_forecast_requires_realizer():
    papers = [_paper("p1", "2024-04-15", "rag")]
    with pytest.raises(ValueError, match="realizer"):
        forecast(
            papers,
            cutoff_t="2024-06-30",
            sampler=lambda *_a: [Innovation("rag", "extend", "x")],
        )
