from __future__ import annotations

import json
from typing import Any, List

from src.backtest.models import IdeaPrediction, PaperRecord
from src.strategy.base import IdeaStrategy


def _normalize_key_terms(raw: Any) -> List[str]:
    if isinstance(raw, list):
        terms = [str(item).strip() for item in raw if str(item).strip()]
    elif isinstance(raw, str):
        terms = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        terms = []

    deduped: List[str] = []
    seen: set[str] = set()
    for term in terms:
        lowered = term.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(term)
    return deduped


def _normalize_confidence(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.5

    if value > 1.0:
        value = value / 10.0
    return round(min(1.0, max(0.0, value)), 4)


def _extract_json_payload(raw_text: str) -> Any:
    text = raw_text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for left, right in (("[", "]"), ("{", "}")):
        start = text.find(left)
        end = text.rfind(right)
        if start == -1 or end == -1 or end < start:
            continue
        fragment = text[start : end + 1]
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            continue
    return None


def _extract_idea_items(payload: Any) -> List[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for key in ("ideas", "predictions"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return [item for item in candidate if isinstance(item, dict)]

        if payload.get("title") or payload.get("Title"):
            return [payload]

    return []


def _build_user_message(
    train_papers: List[PaperRecord],
    cutoff_month: str,
    top_k: int,
) -> str:
    paper_lines = []
    for idx, paper in enumerate(train_papers[-20:], start=1):
        keywords = ", ".join(paper.keywords[:6])
        paper_lines.append(
            f"{idx}. title={paper.title}; month={paper.month}; keywords={keywords}"
        )

    papers_block = "\n".join(paper_lines)
    return (
        "Use the following paper history to propose forward-looking ideas.\n"
        f"Cutoff month: {cutoff_month}.\n"
        f"Return at most {top_k} items.\n"
        "Respond as JSON only with either an array or an object containing an "
        "'ideas' array.\n"
        "Each item must include: title, rationale, key_terms (list of strings), "
        "confidence (0..1).\n\n"
        f"Papers:\n{papers_block}"
    )


class PromptLLMStrategy(IdeaStrategy):
    name = "prompt_llm"

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        prompt_id: str = "llm_baseline",
        prompt_version: str = "v1",
        temperature: float | None = None,
    ) -> None:
        self.model_id = model_id
        self.prompt_id = prompt_id
        self.prompt_version = prompt_version
        self.temperature = temperature

    def generate(
        self,
        train_papers: List[PaperRecord],
        cutoff_month: str,
        top_k: int,
    ) -> List[IdeaPrediction]:
        if not train_papers or top_k <= 0:
            return []

        from backend.llm_utils import create_client, get_response_from_llm
        from backend.prompt_registry import get_prompt_policy

        policy = get_prompt_policy(self.prompt_id, self.prompt_version)
        model_id = self.model_id or str(policy.get("model_id", "gpt-4o-mini"))
        client, resolved_model = create_client(model_id)
        temperature = (
            self.temperature
            if self.temperature is not None
            else float(policy.get("temperature", 0.7))
        )

        system_message = str(policy.get("template") or "").strip()
        if not system_message:
            raise ValueError("Prompt policy template is empty")

        message = _build_user_message(
            train_papers=train_papers,
            cutoff_month=cutoff_month,
            top_k=top_k,
        )

        raw_text, _ = get_response_from_llm(
            msg=message,
            client=client,
            model=resolved_model,
            system_message=system_message,
            temperature=temperature,
        )

        payload = _extract_json_payload(raw_text)
        items = _extract_idea_items(payload)

        predictions: List[IdeaPrediction] = []
        for item in items:
            if len(predictions) >= top_k:
                break

            title = str(item.get("title") or item.get("Title") or "").strip()
            if not title:
                continue

            rationale = str(item.get("rationale") or "").strip()
            if not rationale:
                problem = str(item.get("Problem") or item.get("problem") or "").strip()
                approach = str(item.get("Approach") or item.get("approach") or "").strip()
                importance = str(
                    item.get("Importance") or item.get("importance") or ""
                ).strip()
                parts = [part for part in (problem, approach, importance) if part]
                rationale = " ".join(parts)

            key_terms = _normalize_key_terms(
                item.get("key_terms")
                or item.get("keyTerms")
                or item.get("keywords")
                or item.get("Keywords")
            )
            confidence = _normalize_confidence(
                item.get("confidence", item.get("Score", 0.5))
            )

            predictions.append(
                IdeaPrediction(
                    rank=len(predictions) + 1,
                    title=title,
                    rationale=rationale,
                    key_terms=key_terms,
                    confidence=confidence,
                )
            )

        return predictions
