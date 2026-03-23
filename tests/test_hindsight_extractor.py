"""Tests for hindsight extraction — extractor and prompt."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from live_idea_bench.models import PaperRecord
from forecaster.models import Innovation
from forecaster.config import HindsightConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_paper(
    paper_id: str,
    title: str,
    month: str = "2024-01",
    summary: str = "Test abstract.",
) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        month=month,
        summary=summary,
        keywords=["ml"],
        source_path="",
        published_date=f"{month}-15",
    )


FUTURE_PAPER = _make_paper(
    "future-001",
    "Scaling Diffusion Language Models with Chain-of-Thought",
    month="2024-03",
    summary="We extend diffusion language models to support chain-of-thought reasoning.",
)

CONTEXT_PAPERS = [
    _make_paper(f"ctx-{i:03d}", f"Context Paper {i}", month="2024-01")
    for i in range(20)
]

VALID_JSON = '{"base_direction": "diffusion language models", "operator": "extend", "gap": "Lack of reasoning support in diffusion LMs"}'
VALID_JSON_WITH_FENCES = "```json\n" + VALID_JSON + "\n```"


# ---------------------------------------------------------------------------
# Extractor tests
# ---------------------------------------------------------------------------


class TestExtractInnovation:
    """Tests for forecaster.hindsight.extractor.extract_innovation."""

    def test_extract_innovation_returns_innovation(self):
        """Mock LLM returning valid JSON → Innovation with correct fields."""
        from forecaster.hindsight.extractor import extract_innovation

        client = MagicMock()
        config = HindsightConfig(max_retries=2, temperature=0.2)

        with patch(
            "forecaster.hindsight.extractor.get_response_from_llm",
            return_value=(VALID_JSON, []),
        ):
            result = extract_innovation(
                future_paper=FUTURE_PAPER,
                context_papers=CONTEXT_PAPERS[:5],
                llm_client=client,
                model="gpt-4o",
                config=config,
            )

        assert isinstance(result, Innovation)
        assert result.base_direction == "diffusion language models"
        assert result.operator == "extend"
        assert result.gap == "Lack of reasoning support in diffusion LMs"

    def test_extract_innovation_handles_code_block(self):
        """Mock LLM returning ```json...``` → still parses correctly."""
        from forecaster.hindsight.extractor import extract_innovation

        client = MagicMock()
        config = HindsightConfig(max_retries=2, temperature=0.2)

        with patch(
            "forecaster.hindsight.extractor.get_response_from_llm",
            return_value=(VALID_JSON_WITH_FENCES, []),
        ):
            result = extract_innovation(
                future_paper=FUTURE_PAPER,
                context_papers=CONTEXT_PAPERS[:5],
                llm_client=client,
                model="gpt-4o",
                config=config,
            )

        assert isinstance(result, Innovation)
        assert result.base_direction == "diffusion language models"

    def test_extract_innovation_retries_on_parse_failure(self):
        """First call returns invalid JSON, second returns valid → retry works."""
        from forecaster.hindsight.extractor import extract_innovation

        client = MagicMock()
        config = HindsightConfig(max_retries=2, temperature=0.2)

        call_results = [
            ("not valid json at all", []),
            (VALID_JSON, []),
        ]

        with patch(
            "forecaster.hindsight.extractor.get_response_from_llm",
            side_effect=call_results,
        ):
            result = extract_innovation(
                future_paper=FUTURE_PAPER,
                context_papers=CONTEXT_PAPERS[:5],
                llm_client=client,
                model="gpt-4o",
                config=config,
            )

        assert isinstance(result, Innovation)
        assert result.operator == "extend"

    def test_extract_innovation_raises_after_max_retries(self):
        """Always invalid JSON → ValueError after max_retries exhausted."""
        from forecaster.hindsight.extractor import extract_innovation

        client = MagicMock()
        config = HindsightConfig(max_retries=2, temperature=0.2)

        with patch(
            "forecaster.hindsight.extractor.get_response_from_llm",
            return_value=("this is not json", []),
        ):
            with pytest.raises(ValueError, match="[Ff]ailed|[Ee]xtraction"):
                extract_innovation(
                    future_paper=FUTURE_PAPER,
                    context_papers=CONTEXT_PAPERS[:5],
                    llm_client=client,
                    model="gpt-4o",
                    config=config,
                )


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------


class TestBuildHindsightPrompt:
    """Tests for forecaster.hindsight.prompt.build_hindsight_prompt."""

    def test_build_hindsight_prompt_format(self):
        """System prompt and user message contain expected sections."""
        from forecaster.hindsight.prompt import build_hindsight_prompt

        system_prompt, user_message = build_hindsight_prompt(
            future_paper=FUTURE_PAPER,
            context_papers=CONTEXT_PAPERS[:5],
        )

        # System prompt should reference innovation triple or structured triple
        assert len(system_prompt) > 50

        # User message should contain future paper title
        assert FUTURE_PAPER.title in user_message

        # User message should include at least one context paper title
        assert "Context Paper 0" in user_message

        # User message should contain future abstract
        assert FUTURE_PAPER.summary in user_message

    def test_build_hindsight_prompt_truncates_context(self):
        """Context is limited to max_context_papers."""
        from forecaster.hindsight.prompt import build_hindsight_prompt

        max_papers = 3
        _system_prompt, user_message = build_hindsight_prompt(
            future_paper=FUTURE_PAPER,
            context_papers=CONTEXT_PAPERS,  # 20 papers
            max_context_papers=max_papers,
        )

        # Only the first max_papers context paper titles should appear
        for i in range(max_papers):
            assert f"Context Paper {i}" in user_message

        # Papers beyond the limit should NOT appear
        for i in range(max_papers, len(CONTEXT_PAPERS)):
            assert f"Context Paper {i}" not in user_message

    def test_build_hindsight_prompt_returns_tuple(self):
        """Return value is a (str, str) tuple."""
        from forecaster.hindsight.prompt import build_hindsight_prompt

        result = build_hindsight_prompt(
            future_paper=FUTURE_PAPER,
            context_papers=CONTEXT_PAPERS[:3],
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(part, str) for part in result)

    def test_build_hindsight_prompt_empty_context(self):
        """Empty context papers list does not crash."""
        from forecaster.hindsight.prompt import build_hindsight_prompt

        system_prompt, user_message = build_hindsight_prompt(
            future_paper=FUTURE_PAPER,
            context_papers=[],
        )
        assert FUTURE_PAPER.title in user_message
