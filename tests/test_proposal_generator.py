"""Tests for forecaster/realization/proposal_generator.py"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forecaster.config import RealizationConfig
from forecaster.models import Innovation
from forecaster.realization.proposal_generator import (
    build_realization_messages,
    generate_proposal,
    proposal_to_idea_prediction,
)
from live_idea_bench.models import IdeaPrediction, PaperRecord


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
    def test_build_realization_messages_include_context_and_evidence(self) -> None:
        innovation = _make_innovation()
        context = [_make_paper("ctx1", "Historical Paper", "historical context for the idea")]
        evidence = [_make_paper("p1", "Efficient Attention", "attention mechanism for long sequences")]

        system_prompt, user_prompt = build_realization_messages(
            innovation,
            evidence,
            context_papers=context,
            config=RealizationConfig(),
        )

        assert "Historical context available before the cutoff" in user_prompt
        assert "Historical Paper" in user_prompt
        assert "Efficient Attention" in user_prompt
        assert "transformer attention" in user_prompt
        assert len(system_prompt) > 20

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
        result = proposal_to_idea_prediction(proposal_text, innovation)
        assert result.title == "Efficient Sparse Attention for Long Documents"

    def test_proposal_to_idea_prediction_rank(self) -> None:
        proposal_text = "Title\nBody."
        innovation = _make_innovation()
        result = proposal_to_idea_prediction(proposal_text, innovation, rank=3)
        assert result.rank == 3

    def test_proposal_to_idea_prediction_default_rank(self) -> None:
        proposal_text = "Title\nBody."
        innovation = _make_innovation()
        result = proposal_to_idea_prediction(proposal_text, innovation)
        assert result.rank == 1

    def test_proposal_to_idea_prediction_returns_idea_prediction(self) -> None:
        proposal_text = "My Proposal Title\nDetailed proposal body goes here."
        innovation = _make_innovation()
        result = proposal_to_idea_prediction(proposal_text, innovation)
        assert isinstance(result, IdeaPrediction)

    def test_proposal_to_idea_prediction_gap_in_rationale(self) -> None:
        """The gap from the innovation should appear in the rationale."""
        proposal_text = "Title\nBody text."
        innovation = Innovation(
            base_direction="transformer",
            operator="extend",
            gap="handling very long documents",
        )
        result = proposal_to_idea_prediction(proposal_text, innovation)
        assert "handling very long documents" in result.rationale

    def test_proposal_to_idea_prediction_operator_in_approach(self) -> None:
        """The operator from the innovation should appear in the approach."""
        proposal_text = "Title\nBody text."
        innovation = Innovation(
            base_direction="transformer",
            operator="compose",
            gap="some gap",
        )
        result = proposal_to_idea_prediction(proposal_text, innovation)
        assert "compose" in result.approach

    def test_proposal_to_idea_prediction_empty_proposal(self) -> None:
        """Empty proposal should still return a valid IdeaPrediction."""
        innovation = _make_innovation()
        result = proposal_to_idea_prediction("", innovation)
        assert isinstance(result, IdeaPrediction)

    def test_proposal_to_idea_prediction_single_line(self) -> None:
        """Single-line proposal (no body) should still work."""
        proposal_text = "Only A Title Line"
        innovation = _make_innovation()
        result = proposal_to_idea_prediction(proposal_text, innovation)
        assert result.title == "Only A Title Line"
        assert isinstance(result, IdeaPrediction)


class TestGenerateProposalLocalModel:
    """Phase 1: generate_proposal dispatches to local model when realization_model_path given."""

    def test_generate_proposal_uses_local_model_when_path_provided(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """When realization_model_path exists, uses _generate_proposal_local, not LLM client."""
        from forecaster.config import RealizationConfig

        ckpt_dir = tmp_path / "realization_ckpt"
        ckpt_dir.mkdir()

        innovation = _make_innovation()
        config = RealizationConfig()
        mock_client = MagicMock()

        with patch(
            "forecaster.realization.proposal_generator._generate_proposal_local",
            return_value="Local Model Title\nLocal body text.",
        ) as mock_local, \
        patch(
            "forecaster.realization.proposal_generator.get_response_from_llm",
            return_value=("LLM response", []),
        ) as mock_llm:
            result = generate_proposal(
                innovation=innovation,
                evidence=[],
                llm_client=mock_client,
                model="gpt-4o",
                config=config,
                realization_model_path=str(ckpt_dir),
            )

        mock_local.assert_called_once()
        mock_llm.assert_not_called()
        assert result == "Local Model Title\nLocal body text."

    def test_generate_proposal_raises_when_local_artifact_fails(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """When an artifact is supplied, serving should fail explicitly instead of silently falling back."""
        from forecaster.config import RealizationConfig

        ckpt_dir = tmp_path / "realization_ckpt"
        ckpt_dir.mkdir()

        innovation = _make_innovation()
        config = RealizationConfig()
        mock_client = MagicMock()

        with patch(
            "forecaster.realization.proposal_generator._generate_proposal_local",
            side_effect=RuntimeError("GPU OOM"),
        ), \
        patch(
            "forecaster.realization.proposal_generator.get_response_from_llm",
            return_value=("LLM fallback response", []),
        ) as mock_llm, pytest.raises(RuntimeError, match="artifact generation failed"):
            generate_proposal(
                innovation=innovation,
                evidence=[],
                llm_client=mock_client,
                model="gpt-4o",
                config=config,
                realization_model_path=str(ckpt_dir),
            )

        mock_llm.assert_not_called()

    def test_generate_proposal_can_fallback_when_config_explicitly_allows_it(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Legacy fallback remains opt-in only."""
        ckpt_dir = tmp_path / "realization_ckpt"
        ckpt_dir.mkdir()

        innovation = _make_innovation()
        config = RealizationConfig(allow_artifact_fallback_to_llm=True)
        mock_client = MagicMock()

        with patch(
            "forecaster.realization.proposal_generator._generate_proposal_local",
            side_effect=RuntimeError("GPU OOM"),
        ), \
        patch(
            "forecaster.realization.proposal_generator.get_response_from_llm",
            return_value=("LLM fallback response", []),
        ) as mock_llm:
            result = generate_proposal(
                innovation=innovation,
                evidence=[],
                llm_client=mock_client,
                model="gpt-4o",
                config=config,
                realization_model_path=str(ckpt_dir),
            )

        mock_llm.assert_called_once()
        assert result == "LLM fallback response"

    def test_generate_proposal_uses_llm_when_no_path(self) -> None:
        """Without realization_model_path, always uses the generic LLM client."""
        from forecaster.config import RealizationConfig

        innovation = _make_innovation()
        config = RealizationConfig()
        mock_client = MagicMock()

        with patch(
            "forecaster.realization.proposal_generator._generate_proposal_local",
        ) as mock_local, \
        patch(
            "forecaster.realization.proposal_generator.get_response_from_llm",
            return_value=("LLM response", []),
        ):
            generate_proposal(
                innovation=innovation,
                evidence=[],
                llm_client=mock_client,
                model="gpt-4o",
                config=config,
                realization_model_path=None,
            )

        mock_local.assert_not_called()

    def test_generate_proposal_output_changes_with_artifact(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Changing the realization artifact should change the served proposal under a deterministic stub."""
        artifact_a = tmp_path / "artifact_a"
        artifact_b = tmp_path / "artifact_b"
        artifact_a.mkdir()
        artifact_b.mkdir()

        def _local_side_effect(*, realization_model_path, **kwargs):  # type: ignore[no-untyped-def]
            return f"{Path(realization_model_path).name}\nBody"

        with patch(
            "forecaster.realization.proposal_generator._generate_proposal_local",
            side_effect=_local_side_effect,
        ):
            proposal_a = generate_proposal(
                innovation=_make_innovation(),
                evidence=[],
                llm_client=MagicMock(),
                model="gpt-4o",
                config=RealizationConfig(),
                realization_model_path=str(artifact_a),
            )
            proposal_b = generate_proposal(
                innovation=_make_innovation(),
                evidence=[],
                llm_client=MagicMock(),
                model="gpt-4o",
                config=RealizationConfig(),
                realization_model_path=str(artifact_b),
            )

        assert proposal_a != proposal_b
