"""Retrieval-Augmented Prompting baseline strategy.

Implements baseline (5) from the LiveIdeaBench paper:

    Retrieval-Augmented Prompting augments the historical context with
    retrieved representative papers or summaries before generation, testing
    whether stronger grounding in the historical literature improves
    forecasting beyond direct prompting.

This differs from Direct Prompting in that the historical context is selected
by similarity-based retrieval against a query derived from the most recent
literature, rather than a fixed chronological trailing window.

Single LLM call per window. Retrieval is deterministic (hybrid
semantic+keyword similarity from `live_idea_bench.similarity`), keeping
the baseline batch-friendly and reproducible across re-runs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from live_idea_bench.llm import create_client, get_response_from_llm
from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.papers import add_months, month_to_index
from live_idea_bench.similarity import (
    _hybrid_similarity,
    _keyword_overlap,
    _sanitize,
    paper_text,
)
from live_idea_bench.strategy.base import IdeaStrategy

_DEFAULT_MODEL = "gpt-4o"
_RETRIEVAL_TOP_N = 20  # representative papers fed to the forecaster
_QUERY_RECENT_MONTHS = 3
_QUERY_TOPIC_TERMS = 12

_FORECAST_SYSTEM = (
    "You are a research forecasting assistant grounded in retrieved representative literature. "
    "Return only valid JSON matching the requested schema."
)


def _build_query(train_papers: list[PaperRecord], cutoff_month: str) -> str:
    """Build a query from recent papers — the dominant topics the field is moving toward."""
    cutoff_idx = month_to_index(cutoff_month)
    recent_start_idx = month_to_index(
        add_months(cutoff_month, -(_QUERY_RECENT_MONTHS - 1))
    )
    recent = [
        paper
        for paper in train_papers
        if recent_start_idx <= month_to_index(paper.month) <= cutoff_idx
    ]
    if not recent:
        recent = train_papers[-10:]

    kw_counter: Counter[str] = Counter()
    for paper in recent:
        for kw in paper.keywords:
            cleaned = kw.lower().strip()
            if len(cleaned) > 2:
                kw_counter[cleaned] += 1
    top_terms = [term for term, _ in kw_counter.most_common(_QUERY_TOPIC_TERMS)]

    titles = " | ".join(_sanitize(paper.title) for paper in recent[-10:])
    if top_terms:
        return f"Active topics: {', '.join(top_terms)}. Recent titles: {titles}"
    return f"Recent titles: {titles}"


def _retrieve(
    query: str,
    train_papers: list[PaperRecord],
    top_n: int,
) -> list[PaperRecord]:
    """Hybrid semantic+keyword retrieval — deterministic, no LLM."""
    scored: list[tuple[float, PaperRecord]] = []
    for paper in train_papers:
        ctx = paper_text(paper)
        semantic = _hybrid_similarity(query, ctx)
        keyword = _keyword_overlap(query, ctx)
        scored.append((max(semantic, keyword), paper))
    scored.sort(key=lambda item: -item[0])
    return [paper for _, paper in scored[:top_n]]


def _build_forecast_prompt(
    retrieved: list[PaperRecord],
    cutoff_month: str,
    top_k: int,
) -> str:
    blocks = []
    for idx, paper in enumerate(retrieved, start=1):
        blocks.append(
            f"{idx}. Title: {_sanitize(paper.title)}\n"
            f"Month: {paper.month}\n"
            f"Summary: {_sanitize(paper.summary)}"
        )
    context = "\n\n---\n\n".join(blocks)
    return (
        f"You are forecasting research ideas with a literature cutoff of {cutoff_month}.\n\n"
        f"The following representative papers were retrieved from the historical literature "
        f"as the most relevant grounding for your forecast:\n\n{context}\n\n"
        f"Based on these retrieved papers, forecast {top_k} concrete research ideas likely to "
        f"appear in the months following {cutoff_month}. Do NOT use knowledge from after "
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


class RetrievalPromptingStrategy(IdeaStrategy):
    """Retrieval-Augmented Prompting baseline.

    Retrieves top-N representative papers from the historical literature by
    hybrid similarity to a query derived from the most recent literature,
    then forecasts in a single LLM call grounded on the retrieved set.
    """

    name = "retrieval_prompting"

    def __init__(
        self,
        model_name: str | None = None,
        retrieval_top_n: int = _RETRIEVAL_TOP_N,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model_name = model_name or _DEFAULT_MODEL
        self.retrieval_top_n = retrieval_top_n
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

        query = _build_query(train_papers, cutoff_month)
        retrieved = _retrieve(query, train_papers, self.retrieval_top_n)
        if not retrieved:
            retrieved = train_papers[-self.retrieval_top_n :]

        client, resolved_model = create_client(self.model_name)
        prompt = _build_forecast_prompt(retrieved, cutoff_month, top_k)
        raw, _ = get_response_from_llm(
            msg=prompt,
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
                f"[retrieval_prompting WARNING] cutoff={cutoff_month}: got "
                f"{len(predictions)}/{top_k} predictions — retrying",
                file=sys.stderr,
                flush=True,
            )
            extra_raw, _ = get_response_from_llm(
                msg=prompt,
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
