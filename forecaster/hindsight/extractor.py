"""LLM-based hindsight innovation extraction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from forecaster.config import HindsightConfig
from forecaster.hindsight.prompt import build_hindsight_prompt
from forecaster.models import ALLOWED_INNOVATION_OPERATORS, Innovation
from idea_forecast_bench.llm import get_response_from_llm
from idea_forecast_bench.models import PaperRecord

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def parse_innovation(text: str) -> Innovation:
    """Parse LLM output into Innovation.

    Handles JSON wrapped in ```json...``` code fences.

    Raises:
        ValueError: If parsing fails or required fields are missing.
    """
    stripped = text.strip()

    # Strip code fences if present
    fence_match = _CODE_FENCE_RE.search(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM output is not valid JSON: {exc!r}\nRaw text: {text!r}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}: {text!r}")

    required_fields = ("base_direction", "operator", "gap")
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(
            f"Innovation JSON is missing required fields {missing}. Got keys: {list(data.keys())}"
        )

    operator = str(data["operator"]).strip().lower()
    if operator not in ALLOWED_INNOVATION_OPERATORS:
        raise ValueError(
            f"Unsupported operator {operator!r}. "
            f"Allowed values: {', '.join(ALLOWED_INNOVATION_OPERATORS)}"
        )

    return Innovation(
        base_direction=str(data["base_direction"]),
        operator=operator,
        gap=str(data["gap"]),
    )


# Backward-compatible alias
_parse_innovation = parse_innovation


def extract_innovation(
    future_paper: PaperRecord,
    context_papers: list[PaperRecord],
    llm_client: Any,
    model: str,
    config: HindsightConfig,
) -> Innovation:
    """Extract the latent innovation z from a future paper given historical context.

    Uses LLM to extract structured triple {base_direction, operator, gap}.
    Retries up to config.max_retries on parse failure.

    Args:
        future_paper: The paper to extract an innovation from.
        context_papers: Historical papers available before the cutoff.
        llm_client: Initialized LLM client.
        model: LLM model identifier.
        config: HindsightConfig controlling retries, temperature, etc.

    Returns:
        Innovation with base_direction, operator, and gap fields.

    Raises:
        ValueError: If extraction fails after all retries.
    """
    system_prompt, user_message = build_hindsight_prompt(
        future_paper=future_paper,
        context_papers=context_papers,
        max_context_papers=config.max_context_papers,
    )

    last_error: Exception | None = None
    attempts = config.max_retries + 1

    for attempt in range(1, attempts + 1):
        try:
            raw_text, _ = get_response_from_llm(
                msg=user_message,
                client=llm_client,
                model=model,
                system_message=system_prompt,
                temperature=config.temperature,
            )
            innovation = parse_innovation(raw_text)
            logger.info(
                "Extracted innovation for paper %r (attempt %d/%d): operator=%r",
                future_paper.paper_id,
                attempt,
                attempts,
                innovation.operator,
            )
            return innovation
        except ValueError as exc:
            last_error = exc
            logger.warning(
                "Failed to parse innovation for paper %r on attempt %d/%d: %s",
                future_paper.paper_id,
                attempt,
                attempts,
                exc,
            )

    raise ValueError(
        f"Extraction failed for paper {future_paper.paper_id!r} after {attempts} attempts. "
        f"Last error: {last_error}"
    )
