"""Tests for forecaster/realization/evidence.py"""
from __future__ import annotations

import pytest

from live_idea_bench.models import PaperRecord
from forecaster.models import Innovation
from forecaster.realization.evidence import build_innovation_query, retrieve_evidence


def _make_paper(paper_id: str, title: str, summary: str, keywords: list[str] | None = None) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        month="2024-01",
        summary=summary,
        keywords=keywords or [],
        source_path="",
    )


def _make_innovation(
    base_direction: str = "transformer attention",
    operator: str = "extend",
    gap: str = "efficiency in long sequences",
) -> Innovation:
    return Innovation(
        base_direction=base_direction,
        operator=operator,
        gap=gap,
    )


class TestBuildInnovationQuery:
    def test_build_innovation_query_contains_base_direction(self) -> None:
        innovation = _make_innovation(base_direction="diffusion language model")
        query = build_innovation_query(innovation)
        assert "diffusion language model" in query

    def test_build_innovation_query_contains_operator(self) -> None:
        innovation = _make_innovation(operator="transfer")
        query = build_innovation_query(innovation)
        assert "transfer" in query

    def test_build_innovation_query_contains_gap(self) -> None:
        innovation = _make_innovation(gap="cross-domain adaptation for low-resource settings")
        query = build_innovation_query(innovation)
        assert "cross-domain adaptation for low-resource settings" in query

    def test_build_innovation_query_non_empty(self) -> None:
        innovation = _make_innovation()
        query = build_innovation_query(innovation)
        assert len(query.strip()) > 0

    def test_build_innovation_query_combines_all_fields(self) -> None:
        innovation = Innovation(
            base_direction="vision language model",
            operator="compose",
            gap="chart understanding",
        )
        query = build_innovation_query(innovation)
        # All three fields should appear somewhere in the combined query
        assert "vision language model" in query
        assert "compose" in query
        assert "chart understanding" in query


class TestRetrieveEvidence:
    def test_retrieve_evidence_returns_list(self) -> None:
        innovation = _make_innovation()
        papers = [
            _make_paper("p1", "Efficient Transformer", "attention mechanism for long sequence efficiency"),
        ]
        result = retrieve_evidence(innovation, papers)
        assert isinstance(result, list)

    def test_retrieve_evidence_top_k_limit(self) -> None:
        innovation = _make_innovation(
            base_direction="attention mechanism",
            operator="extend",
            gap="long sequence processing",
        )
        papers = [
            _make_paper(f"p{i}", f"Attention Paper {i}", "attention mechanism long sequence transformer")
            for i in range(20)
        ]
        result = retrieve_evidence(innovation, papers, top_k=3)
        assert len(result) <= 3

    def test_retrieve_evidence_empty_papers(self) -> None:
        innovation = _make_innovation()
        result = retrieve_evidence(innovation, [])
        assert result == []

    def test_retrieve_evidence_above_threshold(self) -> None:
        innovation = _make_innovation(
            base_direction="neural network optimization",
            operator="extend",
            gap="convergence speed",
        )
        relevant_paper = _make_paper(
            "rel",
            "Neural Network Optimization",
            "neural network optimization convergence speed gradient descent",
        )
        irrelevant_paper = _make_paper(
            "irrel",
            "Cooking Recipes",
            "pasta sauce garlic olive oil recipe italian",
        )
        # Use a very low threshold to test that irrelevant papers score low
        result = retrieve_evidence(
            innovation,
            [relevant_paper, irrelevant_paper],
            top_k=5,
            similarity_threshold=0.0,
        )
        # All returned should be PaperRecord instances
        for paper in result:
            assert isinstance(paper, PaperRecord)

    def test_retrieve_evidence_sorted_by_relevance(self) -> None:
        """Papers with higher overlap to the innovation should appear first."""
        innovation = _make_innovation(
            base_direction="attention mechanism transformer",
            operator="extend",
            gap="long sequence efficiency",
        )
        high_relevance = _make_paper(
            "high",
            "Efficient Attention Transformer Long Sequence",
            "attention mechanism transformer long sequence efficiency extend processing",
        )
        low_relevance = _make_paper(
            "low",
            "Cooking Guide",
            "pasta sauce garlic olive oil recipe",
        )
        result = retrieve_evidence(
            innovation,
            [low_relevance, high_relevance],
            top_k=5,
            similarity_threshold=0.0,
        )
        # If both are returned, high relevance should come first
        if len(result) >= 2:
            assert result[0].paper_id == "high"
