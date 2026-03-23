"""Generate research proposals from innovation triples and evidence."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.llm import get_response_from_llm

from forecaster.models import Innovation
from forecaster.config import RealizationConfig

logger = logging.getLogger(__name__)

PROMPT_FILE = Path(__file__).parent.parent / "prompt" / "realization.yaml"


def _load_prompt() -> dict[str, str]:
    """Load the realization prompt templates from YAML."""
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Realization prompt file not found: {PROMPT_FILE}")
    payload = yaml.safe_load(PROMPT_FILE.read_text(encoding="utf-8")) or {}
    return {
        "system_prompt": str(payload.get("system_prompt", "")).strip(),
        "user_template": str(payload.get("user_template", "")).strip(),
    }


def _format_evidence_summary(evidence: list[PaperRecord]) -> str:
    """Format evidence papers into a readable summary string."""
    if not evidence:
        return "(No supporting evidence retrieved.)"
    lines = []
    for i, paper in enumerate(evidence, start=1):
        lines.append(f"{i}. {paper.title}: {paper.summary[:200]}")
    return "\n".join(lines)


def generate_proposal(
    innovation: Innovation,
    evidence: list[PaperRecord],
    llm_client: Any,
    model: str,
    config: RealizationConfig,
) -> str:
    """Generate a research proposal text given innovation triple and evidence.

    Args:
        innovation: The innovation triple (z).
        evidence: Retrieved supporting papers.
        llm_client: Initialized LLM client.
        model: LLM model name.
        config: RealizationConfig with max_tokens.

    Returns:
        Proposal text string (title on first line, body following).
    """
    prompt_data = _load_prompt()
    evidence_summary = _format_evidence_summary(evidence)

    user_msg = prompt_data["user_template"].format(
        base_direction=innovation.base_direction,
        operator=innovation.operator,
        gap=innovation.gap,
        evidence_summary=evidence_summary,
    )

    response_text, _ = get_response_from_llm(
        msg=user_msg,
        client=llm_client,
        model=model,
        system_message=prompt_data["system_prompt"],
    )

    return response_text.strip()


def proposal_to_idea_prediction(
    proposal_text: str,
    innovation: Innovation,
    rank: int = 1,
) -> IdeaPrediction:
    """Convert a proposal text to an IdeaPrediction for benchmark evaluation.

    Extracts title from first line, uses gap as rationale, operator as approach.
    """
    lines = proposal_text.strip().splitlines()
    title = lines[0].strip() if lines else ""
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    return IdeaPrediction(
        rank=rank,
        title=title,
        rationale=f"{innovation.gap}. {body}"[:500] if body else innovation.gap,
        approach=f"{innovation.operator}: {innovation.base_direction}",
        score=0.0,
    )
