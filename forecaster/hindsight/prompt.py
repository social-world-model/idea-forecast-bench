"""Prompt construction for hindsight innovation extraction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from live_idea_bench.models import PaperRecord

PROMPT_FILE = Path(__file__).parent.parent / "prompt" / "hindsight.yaml"

_PROMPT_CONFIG_CACHE: dict[str, Any] | None = None


def _load_prompt_config() -> dict[str, Any]:
    """Load hindsight prompt YAML (cached at module level)."""
    global _PROMPT_CONFIG_CACHE  # noqa: PLW0603
    if _PROMPT_CONFIG_CACHE is None:
        raw = PROMPT_FILE.read_text(encoding="utf-8")
        payload = yaml.safe_load(raw)
        if not isinstance(payload, dict):
            raise ValueError(
                f"hindsight.yaml must decode to a mapping, got {type(payload)}"
            )
        _PROMPT_CONFIG_CACHE = payload
    return _PROMPT_CONFIG_CACHE


def _build_context_summary(
    context_papers: list[PaperRecord],
    max_context_papers: int,
) -> str:
    """Return a numbered list of paper titles, limited to max_context_papers."""
    truncated = context_papers[:max_context_papers]
    if not truncated:
        return "(no historical context available)"
    lines = [f"{i + 1}. {paper.title}" for i, paper in enumerate(truncated)]
    return "\n".join(lines)


def build_hindsight_prompt(
    future_paper: PaperRecord,
    context_papers: list[PaperRecord],
    *,
    max_context_papers: int = 15,
) -> tuple[str, str]:
    """Build (system_prompt, user_message) for hindsight extraction.

    Args:
        future_paper: The paper whose innovation to extract.
        context_papers: Historical papers available before cutoff
            (truncated to max_context_papers).
        max_context_papers: Max number of context papers to include.

    Returns:
        (system_prompt, user_message) tuple for LLM call.
    """
    config = _load_prompt_config()
    system_prompt: str = config["system_prompt"].strip()
    user_template: str = config["user_template"]

    context_summary = _build_context_summary(context_papers, max_context_papers)

    user_message = user_template.format(
        context_summary=context_summary,
        future_title=future_paper.title,
        future_abstract=future_paper.summary,
    )

    return system_prompt, user_message
