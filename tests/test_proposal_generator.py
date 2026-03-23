"""Tests for forecaster/realization/proposal_generator.py"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from live_idea_bench.models import IdeaPrediction, PaperRecord
from forecaster.models import Innovation
from forecaster.config import RealizationConfig
from forecaster.realization.proposal_generator import (
    generate_proposal,
    proposal_to_idea_prediction,
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


def _make_innovation(
    base_direction: str = "transformer attention",
    operator: str = "extend",
    gap: str = "long sequence efficiency",
) -> Innovation:
    return Innovation(base_direction=base_direction, operator=operator, gap=gap)


class TestGenerateProposal:
    def test_generate_proposal_returns_string(self) -> None:
        innovation = _make_innovation()
        evidence = [_make_paper("p1", "Efficient Attention", "attention mechanism for long sequences")]
        config = RealizationConfig()

        mock_client = MagicMock()
        with patch(
            "forecaster.realization.proposal_generator.get_response_from_llm",
            return_value=("Proposed Title\nThis is the proposal body.", []),
        ):
            result = generate_proposal(
                innovation=innovation,
                evidence=evidence,
                llm_client=mock_client,
                model="gpt-4o",
                config=config,
            )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_proposal_uses_innovation_fields(self) -> None:
        """The innovation fields should appear in the prompt sent to the LLM."""
        innovation = Innovation(
            base_direction="sparse attention",
            operator="extend",
            gap="memory efficiency in long documents",
        )
        evidence: list[PaperRecord] = []
        config = RealizationConfig()

        captured_calls: list[dict] = []

        def fake_llm(msg, client, model, system_message, **kwargs):
            captured_calls.append({"msg": msg, "system_message": system_message})
            return ("Title\nBody text here.", [])

        mock_client = MagicMock()
        with patch("forecaster.realization.proposal_generator.get_response_from_llm", side_effect=fake_llm):
            generate_proposal(
                innovation=innovation,
                evidence=evidence,
                llm_client=mock_client,
                model="gpt-4o",
                config=config,
            )

        assert len(captured_calls) == 1
        prompt_text = captured_calls[0]["msg"]
        assert "sparse attention" in prompt_text
        assert "extend" in prompt_text
        assert "memory efficiency in long documents" in prompt_text

    def test_generate_proposal_no_evidence(self) -> None:
        """Should still work when evidence list is empty."""
        innovation = _make_innovation()
        config = RealizationConfig()
        mock_client = MagicMock()

        with patch(
            "forecaster.realization.proposal_generator.get_response_from_llm",
            return_value=("My Title\nBody.", []),
        ):
            result = generate_proposal(
                innovation=innovation,
                evidence=[],
                llm_client=mock_client,
                model="gpt-4o",
                config=config,
            )

        assert isinstance(result, str)


class TestProposalToIdeaPrediction:
    def test_proposal_to_idea_prediction_extracts_title(self) -> None:
        proposal_text = "Efficient Sparse Attention for Long Documents\nThe body of the proposal follows here."
        innovation = _make_innovation()
        result = proposal_to_idea_prediction(proposal_text, innovation, evidence=[])
        assert result.title == "Efficient Sparse Attention for Long Documents"

    def test_proposal_to_idea_prediction_rank(self) -> None:
        proposal_text = "Title\nBody."
        innovation = _make_innovation()
        result = proposal_to_idea_prediction(proposal_text, innovation, evidence=[], rank=3)
        assert result.rank == 3

    def test_proposal_to_idea_prediction_default_rank(self) -> None:
        proposal_text = "Title\nBody."
        innovation = _make_innovation()
        result = proposal_to_idea_prediction(proposal_text, innovation, evidence=[])
        assert result.rank == 1

    def test_proposal_to_idea_prediction_returns_idea_prediction(self) -> None:
        proposal_text = "My Proposal Title\nDetailed proposal body goes here."
        innovation = _make_innovation()
        result = proposal_to_idea_prediction(proposal_text, innovation, evidence=[])
        assert isinstance(result, IdeaPrediction)

    def test_proposal_to_idea_prediction_gap_in_rationale(self) -> None:
        """The gap from the innovation should appear in the rationale."""
        proposal_text = "Title\nBody text."
        innovation = Innovation(
            base_direction="transformer",
            operator="extend",
            gap="handling very long documents",
        )
        result = proposal_to_idea_prediction(proposal_text, innovation, evidence=[])
        assert "handling very long documents" in result.rationale

    def test_proposal_to_idea_prediction_operator_in_approach(self) -> None:
        """The operator from the innovation should appear in the approach."""
        proposal_text = "Title\nBody text."
        innovation = Innovation(
            base_direction="transformer",
            operator="compose",
            gap="some gap",
        )
        result = proposal_to_idea_prediction(proposal_text, innovation, evidence=[])
        assert "compose" in result.approach

    def test_proposal_to_idea_prediction_empty_proposal(self) -> None:
        """Empty proposal should still return a valid IdeaPrediction."""
        innovation = _make_innovation()
        result = proposal_to_idea_prediction("", innovation, evidence=[])
        assert isinstance(result, IdeaPrediction)

    def test_proposal_to_idea_prediction_single_line(self) -> None:
        """Single-line proposal (no body) should still work."""
        proposal_text = "Only A Title Line"
        innovation = _make_innovation()
        result = proposal_to_idea_prediction(proposal_text, innovation, evidence=[])
        assert result.title == "Only A Title Line"
        assert isinstance(result, IdeaPrediction)
