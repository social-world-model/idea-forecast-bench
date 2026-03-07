from __future__ import annotations

import json
import random
import re
from typing import Any, Iterable, List

from live_idea_bench.config import load_predictor_config, load_runtime_config
from live_idea_bench.llm import create_client, get_response_from_llm
from live_idea_bench.models import IdeaPrediction, PaperRecord

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "via",
    "with",
    "we",
    "our",
    "using",
    "based",
    "towards",
    "toward",
    "across",
    "new",
    "study",
    "paper",
    "method",
    "methods",
    "results",
}

_JSON_DECODER = json.JSONDecoder()


def extract_abstract(text: str) -> str:
    abstract_pattern = re.compile(r"^(#+)\s*Abstract", re.IGNORECASE | re.MULTILINE)
    match = abstract_pattern.search(text)
    if not match:
        fallback_pattern = re.compile(r"^Abstract\s*\n", re.IGNORECASE | re.MULTILINE)
        match = fallback_pattern.search(text)
        if not match:
            return text[:3000]

    start_pos = match.start()
    if match.group(0).startswith("#"):
        header_level = len(match.group(1))
        remaining_text = text[match.end() :]
        next_header_pattern = re.compile(rf"^#{{1,{header_level}}}\s+", re.MULTILINE)
        next_match = next_header_pattern.search(remaining_text)
        if next_match:
            return text[start_pos : match.end() + next_match.start()].strip()

    return text[start_pos : start_pos + 5000].strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _top_terms(texts: Iterable[str], limit: int = 20) -> list[str]:
    counts: dict[str, int] = {}
    for text in texts:
        for tok in _tokenize(text):
            if len(tok) < 4 or tok in STOPWORDS:
                continue
            counts[tok] = counts.get(tok, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [term for term, _ in ordered[:limit]]


def _jaccard(a: str, b: str) -> float:
    sa = set(_tokenize(a))
    sb = set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _prediction_text(pred: IdeaPrediction) -> str:
    return f"{pred.title} {pred.rationale} {pred.approach}".strip()


def _base_score(pred: IdeaPrediction, signal_terms: list[str]) -> float:
    pred_tokens = set(_tokenize(_prediction_text(pred)))
    if not signal_terms:
        return 0.0
    coverage = sum(1 for term in signal_terms if term in pred_tokens) / len(signal_terms)
    specificity = min(len(pred_tokens) / 30.0, 1.0)
    return (0.75 * coverage) + (0.25 * specificity)


def _dedup_predictions(candidates: list[IdeaPrediction], threshold: float = 0.8) -> list[IdeaPrediction]:
    deduped: list[IdeaPrediction] = []
    title_keys: set[str] = set()

    for candidate in candidates:
        key = re.sub(r"\s+", " ", candidate.title.lower()).strip()
        if key in title_keys:
            continue
        if any(_jaccard(_prediction_text(candidate), _prediction_text(kept)) >= threshold for kept in deduped):
            continue
        title_keys.add(key)
        deduped.append(candidate)
    return deduped


def _rank_predictions(candidates: list[IdeaPrediction], signal_terms: list[str], top_k: int) -> list[IdeaPrediction]:
    scored = [(candidate, _base_score(candidate, signal_terms)) for candidate in candidates]
    pool = sorted(scored, key=lambda item: (-item[1], item[0].title.lower()))
    selected: list[IdeaPrediction] = []

    while pool and len(selected) < top_k:
        if not selected:
            pred, score = pool.pop(0)
            pred.score = round(score, 4)
            if pred.confidence is None:
                pred.confidence = round(score, 4)
            pred.rank = len(selected) + 1
            selected.append(pred)
            continue

        best_idx = 0
        best_mmr = float("-inf")
        for idx, (candidate, base_score) in enumerate(pool):
            similarity = max(_jaccard(_prediction_text(candidate), _prediction_text(chosen)) for chosen in selected)
            novelty = 1.0 - similarity
            mmr = (0.65 * base_score) + (0.35 * novelty)
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        pred, _ = pool.pop(best_idx)
        pred.score = round(best_mmr, 4)
        if pred.confidence is None:
            pred.confidence = round(best_mmr, 4)
        pred.rank = len(selected) + 1
        selected.append(pred)

    return selected


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

    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = _JSON_DECODER.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        return payload
    return None


def _coerce_score(raw: Any, default: float = 0.5) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    if value > 1.0:
        value = value / 10.0
    return round(min(1.0, max(0.0, value)), 4)


def _coerce_key_terms(raw: Any) -> list[str]:
    if isinstance(raw, list):
        items = [str(item).strip() for item in raw if str(item).strip()]
    elif isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        items = []
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(item)
    return deduped


def _parse_prediction_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        ideas = payload.get("ideas")
        if isinstance(ideas, list):
            return [item for item in ideas if isinstance(item, dict)]
        if payload.get("title") or payload.get("Title"):
            return [payload]
    return []


def _infer_domain(train_papers: list[PaperRecord]) -> str:
    terms = _top_terms(
        list(paper.summary for paper in train_papers[-20:]) + [kw for p in train_papers[-20:] for kw in p.keywords],
        limit=3,
    )
    return ", ".join(terms) if terms else "recent AI research"


def _build_abstract_block(train_papers: list[PaperRecord], max_context_papers: int) -> str:
    blocks = []
    for idx, paper in enumerate(train_papers[-max_context_papers:], start=1):
        blocks.append(
            f"{idx}. Title: {paper.title}\nMonth: {paper.month}\nSummary: {paper.summary}"
        )
    return "\n\n---\n\n".join(blocks)


def _llm_predictions(
    train_papers: list[PaperRecord],
    cutoff_month: str,
    top_k: int,
    *,
    model_name: str,
    predictor_config_path: str,
    temperature: float | None,
) -> list[IdeaPrediction]:
    predictor_config = load_predictor_config(predictor_config_path)
    runtime_config = load_runtime_config()
    client, resolved_model = create_client(model_name)
    resolved_temperature = (
        temperature
        if temperature is not None
        else predictor_config.default_temperature
        if predictor_config.default_temperature is not None
        else runtime_config.temperature
    )

    message = predictor_config.user_template.format(
        domain=_infer_domain(train_papers),
        horizon=f"the months after {cutoff_month}",
        n_ideas=top_k,
        abstracts=_build_abstract_block(train_papers, predictor_config.max_context_papers),
        cutoff_month=cutoff_month,
    )
    raw_text, _ = get_response_from_llm(
        msg=message,
        client=client,
        model=resolved_model,
        system_message=predictor_config.system_prompt,
        temperature=resolved_temperature,
    )
    payload = _extract_json_payload(raw_text)
    items = _parse_prediction_items(payload)

    predictions: list[IdeaPrediction] = []
    for item in items:
        title = str(item.get("title") or item.get("Title") or "").strip()
        if not title:
            continue
        rationale = str(item.get("rationale") or item.get("Rationale") or "").strip()
        approach = str(item.get("approach") or item.get("Approach") or "").strip()
        predictions.append(
            IdeaPrediction(
                rank=len(predictions) + 1,
                title=title,
                rationale=rationale,
                approach=approach,
                score=_coerce_score(item.get("score", item.get("Score", 0.5))),
                confidence=_coerce_score(
                    item.get("confidence", item.get("Confidence", item.get("score", item.get("Score", 0.5))))
                ),
                key_terms=_coerce_key_terms(item.get("key_terms") or item.get("keywords")),
            )
        )
        if len(predictions) >= top_k:
            break

    if not predictions:
        raise RuntimeError("predictor LLM output could not be parsed into structured ideas")
    return predictions


def _heuristic_predictions(
    train_papers: list[PaperRecord],
    cutoff_month: str,
    top_k: int,
) -> list[IdeaPrediction]:
    texts = [paper.summary for paper in train_papers[-40:]]
    terms = _top_terms(texts + [kw for paper in train_papers[-20:] for kw in paper.keywords], limit=max(10, top_k * 3))
    if len(terms) < 3:
        terms.extend(["retrieval", "alignment", "reasoning"])

    templates = [
        "{a}-{b} co-training for robust research forecasting",
        "Data-centric {a} pipelines with uncertainty-aware {b}",
        "Continual {a} adaptation for long-horizon {b}",
        "Resource-efficient {a} with sparse {b} routing",
        "Cross-modal {a} planning with verifier-guided {b}",
        "Benchmark-driven {a} generalization beyond static {b}",
    ]

    seed = hash((cutoff_month, tuple(terms[:10]), top_k)) & 0xFFFFFFFF
    rng = random.Random(seed)
    candidates: list[IdeaPrediction] = []
    for idx in range(max(top_k * 2, 8)):
        a = terms[idx % len(terms)]
        b = terms[(idx * 3 + 1) % len(terms)]
        if a == b:
            continue
        candidates.append(
            IdeaPrediction(
                rank=idx + 1,
                title=templates[idx % len(templates)].format(a=a, b=b),
                rationale=(
                    f"Recent papers before {cutoff_month} repeatedly emphasize {a} and {b}, "
                    "suggesting a plausible next-stage direction."
                ),
                approach=(
                    f"Build a controlled benchmark around {a} and {b}, then compare transfer, "
                    "robustness, and compute trade-offs on a rolling future-paper stream."
                ),
                score=0.0,
                confidence=None,
                key_terms=[a, b],
            )
        )

    rng.shuffle(candidates)
    candidates = _dedup_predictions(candidates)
    signal_terms = _top_terms(texts + [paper.title for paper in train_papers[-20:]], limit=12)
    return _rank_predictions(candidates, signal_terms, top_k)


def generate_predictions(
    train_papers: list[PaperRecord],
    cutoff_month: str,
    top_k: int,
    *,
    model_name: str | None = None,
    predictor_config_path: str = "predictor.yaml",
    temperature: float | None = None,
) -> list[IdeaPrediction]:
    if not train_papers or top_k <= 0:
        return []

    runtime_config = load_runtime_config()
    predictor_config = load_predictor_config(predictor_config_path)
    resolved_model = model_name or predictor_config.default_model or runtime_config.model_name

    try:
        predictions = _llm_predictions(
            train_papers=train_papers,
            cutoff_month=cutoff_month,
            top_k=top_k,
            model_name=resolved_model,
            predictor_config_path=predictor_config_path,
            temperature=temperature,
        )
    except Exception:
        predictions = _heuristic_predictions(
            train_papers=train_papers,
            cutoff_month=cutoff_month,
            top_k=top_k,
        )

    for idx, prediction in enumerate(predictions, start=1):
        prediction.rank = idx
    return predictions[:top_k]
