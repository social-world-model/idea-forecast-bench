from __future__ import annotations

import re
from typing import Any

from idea_forecast_bench.combinatorial.config import PromptPair
from idea_forecast_bench.combinatorial.extraction import extract_json_object
from idea_forecast_bench.combinatorial.llm_caller import TextCaller
from idea_forecast_bench.models import IdeaPrediction

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]+")
_GENERIC = frozenset(
    [
        "novel", "framework", "approach", "method", "improve", "improving",
        "robust", "robustness", "efficient", "efficiency", "general",
        "generalization", "learning", "model", "models", "large", "language",
        "new", "better", "performance", "towards", "toward", "study", "analysis",
    ]
)  # fmt: skip


def _score_0_3(raw: object) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float) and raw.is_integer():
        value = int(raw)
    elif isinstance(raw, str) and raw.strip().isdigit():
        value = int(raw.strip())
    else:
        return None
    return value if 0 <= value <= 3 else None


def lexical_stats(pred: IdeaPrediction) -> dict[str, float]:
    """Cheap outcome-blind proxies for breadth, reported next to the LLM
    rating so the two can be checked against each other."""
    title_tokens = _TOKEN_RE.findall(pred.title.lower())
    body_tokens = _TOKEN_RE.findall(f"{pred.rationale} {pred.approach}".lower())
    content = [t for t in title_tokens if t not in _GENERIC]
    return {
        "title_tokens": float(len(title_tokens)),
        "title_content_tokens": float(len(content)),
        "body_tokens": float(len(body_tokens)),
        "n_key_terms": float(len(pred.key_terms)),
    }


def rate_specificity(
    pred: IdeaPrediction,
    caller: TextCaller,
    prompt: PromptPair,
    temperature: float,
) -> dict[str, Any]:
    user = prompt.user_template.format(
        title=pred.title,
        rationale=pred.rationale[:800],
        approach=pred.approach[:600],
        key_terms=", ".join(pred.key_terms),
    )
    raw = caller.complete(prompt.system_prompt, user, temperature=temperature, seed=0)
    payload = extract_json_object(raw) or {}
    specificity = _score_0_3(payload.get("specificity"))
    breadth = _score_0_3(payload.get("breadth"))
    testable_raw = payload.get("testable")
    testable = testable_raw if isinstance(testable_raw, bool) else None
    return {
        "specificity": specificity,
        "breadth": breadth,
        "testable": testable,
        "parse_failed": specificity is None or breadth is None,
        "raw": raw[:300],
        **lexical_stats(pred),
    }
