"""Summary Prompting baseline strategy.

Implements baseline (4) from the LiveIdeaBench paper:

    Summary Prompting first compresses the historical literature into a short
    summary and then forecasts future ideas from that summary, testing whether
    lightweight historical abstraction is sufficient.

This differs from Memory-Augmented Prompting in that the forecast step
conditions ONLY on the compressed summary — raw recent abstracts are not
re-introduced. This isolates the contribution of "lightweight abstraction"
alone, as the paper specifies.

Two LLM calls per window:
  Call 1 (compress) — system prefix "You are a research summarizer." —
    compress all train_papers into a short single-paragraph summary.
  Call 2 (forecast) — system prefix "You are a research forecasting assistant" —
    forecast top_k ideas conditioned on the summary only.

The two system prefixes are intentionally distinct from MemoryPromptingStrategy
so the batch runner can route the two rounds independently.
"""

from __future__ import annotations

import json
import re
from typing import Any

from live_idea_bench.llm import create_client, get_response_from_llm
from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.similarity import _sanitize
from live_idea_bench.strategy.base import IdeaStrategy

_DEFAULT_MODEL = "gpt-4o"
_MAX_COMPRESS_PAPERS = 60
_SUMMARY_TARGET_SENTENCES = 8

_COMPRESS_SYSTEM = "You are a research summarizer. Output a single short paragraph; no preamble, no bullets."
_FORECAST_SYSTEM = (
    "You are a research forecasting assistant working from a compressed historical summary. "
    "Return only valid JSON matching the requested schema."
)


def _build_compress_prompt(train_papers: list[PaperRecord], cutoff_month: str) -> str:
    snippets = []
    for paper in train_papers[-_MAX_COMPRESS_PAPERS:]:
        snippet = _sanitize(paper.summary[:300]).replace("\n", " ")
        snippets.append(f"- [{paper.month}] {_sanitize(paper.title)}: {snippet}")
    block = "\n".join(snippets)
    return (
        f"The following papers were published before {cutoff_month}.\n\n"
        f"{block}\n\n"
        f"Compress this historical literature into a SHORT single-paragraph summary of "
        f"about {_SUMMARY_TARGET_SENTENCES} sentences. Capture dominant research themes, "
        f"methodological trajectories, and recurring open problems. "
        f"Do not list individual papers; produce a coherent prose summary."
    )


def _build_forecast_prompt(summary: str, cutoff_month: str, top_k: int) -> str:
    return (
        f"Historical literature summary (cutoff {cutoff_month}):\n\n{summary}\n\n"
        f"Working ONLY from this summary (do not invoke knowledge from after {cutoff_month}), "
        f"forecast {top_k} concrete research ideas likely to appear in the months following "
        f"{cutoff_month}.\n\n"
        f"Return JSON only:\n"
        f'{{"ideas": [{{"title": "...", "rationale": "...", "approach": "...", '
        f'"confidence": 0.0, "key_terms": []}}]}}'
    )


def _parse_predictions(raw: str, top_k: int) -> list[IdeaPrediction]:
    items: list[dict[str, Any]] = []
    try:
        payload = json.loads(raw.strip())
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            ideas = payload.get("ideas") or payload.get("predictions") or []
            items = ideas if isinstance(ideas, list) else []
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group())
            except json.JSONDecodeError:
                items = []

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


class SummaryPromptingStrategy(IdeaStrategy):
    """Summary Prompting baseline — compress, then forecast from summary only."""

    name = "summary_prompting"

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model_name = model_name or _DEFAULT_MODEL
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

        client, resolved_model = create_client(self.model_name)

        compress_prompt = _build_compress_prompt(train_papers, cutoff_month)
        summary, _ = get_response_from_llm(
            msg=compress_prompt,
            client=client,
            model=resolved_model,
            system_message=_COMPRESS_SYSTEM,
            temperature=self.temperature if self.temperature is not None else 0.3,
            reasoning_effort=self.reasoning_effort,
        )
        summary = summary.strip()

        forecast_prompt = _build_forecast_prompt(summary, cutoff_month, top_k)
        raw, _ = get_response_from_llm(
            msg=forecast_prompt,
            client=client,
            model=resolved_model,
            system_message=_FORECAST_SYSTEM,
            temperature=self.temperature if self.temperature is not None else 0.4,
            reasoning_effort=self.reasoning_effort,
        )
        predictions = _parse_predictions(raw, top_k)

        if len(predictions) < top_k:
            import sys

            print(
                f"[summary_prompting WARNING] cutoff={cutoff_month}: got "
                f"{len(predictions)}/{top_k} predictions — retrying",
                file=sys.stderr,
                flush=True,
            )
            extra_raw, _ = get_response_from_llm(
                msg=forecast_prompt,
                client=client,
                model=resolved_model,
                system_message=_FORECAST_SYSTEM,
                temperature=(self.temperature or 0.4) + 0.1,
                reasoning_effort=self.reasoning_effort,
            )
            extra = _parse_predictions(extra_raw, top_k)
            seen = {p.title.lower() for p in predictions}
            for pred in extra:
                if pred.title.lower() not in seen and len(predictions) < top_k:
                    predictions.append(pred)
                    seen.add(pred.title.lower())

        for idx, pred in enumerate(predictions[:top_k], start=1):
            pred.rank = idx
        return predictions[:top_k]
