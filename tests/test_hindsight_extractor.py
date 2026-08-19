"""Tests for hindsight extraction — extractor and prompt."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forecaster.config import HindsightConfig
from forecaster.models import Innovation
from live_idea_bench.models import PaperRecord
from live_idea_bench.papers import parse_markdown_paper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_paper(
    paper_id: str,
    title: str,
    month: str = "2024-01",
    summary: str = "Test abstract.",
    references: list[dict[str, object]] | None = None,
    citations: list[dict[str, object]] | None = None,
) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        month=month,
        summary=summary,
        keywords=["ml"],
        source_path="",
        published_date=f"{month}-15",
        references=references or [],
        citations=citations or [],
    )


FUTURE_PAPER = _make_paper(
    "future-001",
    "Scaling Diffusion Language Models with Chain-of-Thought",
    month="2024-03",
    summary="We extend diffusion language models to support chain-of-thought reasoning.",
    references=[
        {
            "title": "Diffusion Language Modeling",
            "authors": ["Ada Lovelace", "Alan Turing"],
            "year": 2023,
        }
    ],
    citations=[
        {
            "title": "Reasoning with External Memory",
            "context": "Compared against prior chain-of-thought approaches.",
        }
    ],
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

        with (
            patch(
                "forecaster.hindsight.extractor.get_response_from_llm",
                return_value=("this is not json", []),
            ),
            pytest.raises(ValueError, match="[Ff]ailed|[Ee]xtraction"),
        ):
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

        # User message should include citation/reference grounding when available
        assert "Diffusion Language Modeling" in user_message
        assert "Reasoning with External Memory" in user_message

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

    def test_build_hindsight_prompt_uses_body_style_bibliography(self, tmp_path: Path):
        """Body-style REFERENCES should become non-empty reference grounding."""
        from forecaster.hindsight.prompt import build_hindsight_prompt

        paper_path = tmp_path / "2024-03" / "2403.12345.md"
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        paper_path.write_text(
            (
                "# Retrieval-Augmented Forecasting\n\n"
                "Abstract— We ground forecasting models with bibliography evidence.\n\n"
                "# INTRODUCTION\n\n"
                "Intro.\n\n"
                "# REFERENCES\n\n"
                "[1] Foundational Work on Retrieval.\n"
                "[2] Evidence Selection for Forecasting.\n"
            ),
            encoding="utf-8",
        )

        future_paper = parse_markdown_paper(paper_path)
        assert future_paper is not None
        assert future_paper.references

        _system_prompt, user_message = build_hindsight_prompt(
            future_paper=future_paper,
            context_papers=CONTEXT_PAPERS[:2],
        )

        assert "Reference grounding:" in user_message
        assert "Foundational Work on Retrieval." in user_message
        assert "Evidence Selection for Forecasting." in user_message
