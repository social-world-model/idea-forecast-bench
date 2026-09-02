"""Memory-Augmented Prompting baseline strategy.

Maintains a compressed "memory" of long-term historical papers and combines it
with recent full abstracts when prompting the LLM to forecast future ideas.

Architecture
------------
Given training papers up to cutoff_month:

  long-term papers  (older than `memory_months` before cutoff)
        │
        ▼
  [LLM compress]  → memory: bullet-point summary of recurring themes
        │
        ├── + recent papers (last `memory_months` months, full abstracts)
        │
        ▼
  [LLM forecast]  → top_k research predictions

This tests whether lightweight historical abstraction (memory) adds value over
using only recent context (Direct Prompting).
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

from idea_forecast_bench.llm import create_client, get_response_from_llm
from idea_forecast_bench.models import IdeaPrediction, PaperRecord
from idea_forecast_bench.papers import add_months, month_to_index
from idea_forecast_bench.similarity import _sanitize
from idea_forecast_bench.strategy.base import IdeaStrategy

_DEFAULT_MODEL = "gpt-4o"
_MAX_LONG_TERM_PAPERS = 60  # cap to avoid huge prompts
_MAX_RECENT_PAPERS = 20
_MAX_MEMORY_BULLETS = 8


def _split_papers(
    train_papers: list[PaperRecord],
    cutoff_month: str,
    memory_months: int,
) -> tuple[list[PaperRecord], list[PaperRecord]]:
    """Split into long-term (older) and recent (last memory_months) papers."""
    recent_start_idx = month_to_index(add_months(cutoff_month, -(memory_months - 1)))
    long_term, recent = [], []
    for p in train_papers:
        if month_to_index(p.month) >= recent_start_idx:
            recent.append(p)
        else:
            long_term.append(p)
    return long_term, recent


def _compress_to_memory(
    long_term_papers: list[PaperRecord],
    cutoff_month: str,
    client: object,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None = None,
) -> str:
    """Use LLM to compress long-term papers into memory bullets."""
    if not long_term_papers:
        return ""

    # Build a condensed representation (title + summary snippet only)
    snippets = []
    for p in long_term_papers[-_MAX_LONG_TERM_PAPERS:]:
        summary_short = _sanitize(p.summary[:300]).replace("\n", " ")
        snippets.append(f"- [{p.month}] {_sanitize(p.title)}: {summary_short}")
    snippet_block = "\n".join(snippets)

    prompt = (
        f"The following papers were published before {cutoff_month}.\n\n"
        f"{snippet_block}\n\n"
        f"Summarize the {_MAX_MEMORY_BULLETS} most important recurring research themes "
        f"from these papers as concise bullet points (one line each). "
        f"Focus on methodological trends and open problems, not individual papers."
    )
    system = "You are a research analyst. Output only a bullet-point list, no preamble."
    memory_text, _ = get_response_from_llm(
        msg=prompt,
        client=client,
        model=model,
        system_message=system,
        temperature=temperature if temperature is not None else 0.3,
        reasoning_effort=reasoning_effort,
    )
    return memory_text.strip()


def _forecast_with_memory(
    memory: str,
    recent_papers: list[PaperRecord],
    cutoff_month: str,
    top_k: int,
    client: object,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None = None,
) -> list[IdeaPrediction]:
    """Generate predictions using memory + recent papers."""
    recent_block = "\n\n---\n\n".join(
        f"{i}. Title: {_sanitize(p.title)}\nMonth: {p.month}\nSummary: {_sanitize(p.summary)}"
        for i, p in enumerate(recent_papers[-_MAX_RECENT_PAPERS:], start=1)
    )

    memory_section = (
        f"Historical Memory (recurring themes from earlier literature):\n{memory}\n\n"
        if memory
        else ""
    )

    prompt = (
        f"{memory_section}"
        f"Recent abstracts (up to {cutoff_month}):\n\n{recent_block}\n\n"
        f"Based on the historical memory and recent literature, forecast {top_k} concrete research "
        f"ideas likely to appear in the months after {cutoff_month}. "
        f"Do NOT use knowledge from after {cutoff_month}.\n\n"
        f"Return JSON only:\n"
        f'{{"ideas": [{{"title": "...", "rationale": "...", "approach": "...", "confidence": 0.0, "key_terms": []}}]}}'
    )
    system = (
        "You are a research forecasting assistant with memory of long-term trends. "
        "Return only valid JSON matching the schema."
    )
    raw, _ = get_response_from_llm(
        msg=prompt,
        client=client,
        model=model,
        system_message=system,
        temperature=temperature if temperature is not None else 0.4,
        reasoning_effort=reasoning_effort,
    )

    items: list[dict[str, Any]] = []
    try:
        payload = json.loads(raw.strip())
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            ideas = payload.get("ideas") or payload.get("predictions") or []
            items = ideas if isinstance(ideas, list) else []
    except json.JSONDecodeError:
        import re

        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            with contextlib.suppress(json.JSONDecodeError):
                items = json.loads(m.group())

    predictions: list[IdeaPrediction] = []
    for item in items[:top_k]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        predictions.append(
            IdeaPrediction(
                rank=len(predictions) + 1,
                title=title,
                rationale=str(item.get("rationale") or "").strip(),
                approach=str(item.get("approach") or "").strip(),
                score=0.5,
                confidence=float(item.get("confidence") or 0.5),
                key_terms=list(item.get("key_terms") or []),
            )
        )
    return predictions


class MemoryPromptingStrategy(IdeaStrategy):
    """Memory-Augmented Prompting baseline.

    Compresses long-term papers into a memory summary, then forecasts using
    memory + recent full abstracts. Tests whether generic memory augmentation
    (without structured prototyping) improves over direct prompting.
    """

    name = "memory_prompting"

    def __init__(
        self,
        model_name: str | None = None,
        memory_months: int = 6,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model_name = model_name or _DEFAULT_MODEL
        self.memory_months = memory_months
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def generate(
        self,
        train_papers: list[PaperRecord],
        cutoff_month: str,
        top_k: int,
    ) -> list[IdeaPrediction]:
        if not train_papers:
            return []

        long_term, recent = _split_papers(
            train_papers, cutoff_month, self.memory_months
        )
        client, resolved_model = create_client(self.model_name)

        # If no long-term papers, skip memory compression step
        memory = (
            _compress_to_memory(
                long_term,
                cutoff_month,
                client,
                resolved_model,
                self.temperature,
                reasoning_effort=self.reasoning_effort,
            )
            if long_term
            else ""
        )

        predictions = _forecast_with_memory(
            memory=memory,
            recent_papers=recent if recent else train_papers,
            cutoff_month=cutoff_month,
            top_k=top_k,
            client=client,
            model=resolved_model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )

        # Retry if short
        if len(predictions) < top_k:
            import sys

            print(
                f"[memory_prompting WARNING] cutoff={cutoff_month}: got {len(predictions)}/{top_k} predictions — retrying",
                file=sys.stderr,
                flush=True,
            )
            extra = _forecast_with_memory(
                memory=memory,
                recent_papers=recent if recent else train_papers,
                cutoff_month=cutoff_month,
                top_k=top_k,
                client=client,
                model=resolved_model,
                temperature=(self.temperature or 0.4) + 0.1,
                reasoning_effort=self.reasoning_effort,
            )
            seen = {p.title.lower() for p in predictions}
            for p in extra:
                if p.title.lower() not in seen and len(predictions) < top_k:
                    predictions.append(p)
                    seen.add(p.title.lower())

        for idx, pred in enumerate(predictions[:top_k], start=1):
            pred.rank = idx
        return predictions[:top_k]
