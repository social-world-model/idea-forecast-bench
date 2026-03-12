from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set
from zoneinfo import ZoneInfo

from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.papers import date_to_ordinal, get_paper_published_date, month_start_date, normalize_date, normalize_month
from live_idea_bench.similarity import evaluate_predictions


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def coerce_prediction(raw: Dict[str, Any], rank_fallback: int) -> IdeaPrediction:
    rank_raw = raw.get("rank", rank_fallback)
    try:
        rank = int(rank_raw)
    except (TypeError, ValueError):
        rank = rank_fallback

    def _maybe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return None
        if value_float > 1.0:
            value_float = value_float / 10.0
        return round(min(1.0, max(0.0, value_float)), 4)

    key_terms_raw = raw.get("key_terms") or raw.get("keywords") or []
    if not isinstance(key_terms_raw, list):
        key_terms_raw = []

    return IdeaPrediction(
        rank=rank,
        title=str(raw.get("title", "")),
        rationale=str(raw.get("rationale", "")),
        approach=str(raw.get("approach", "")),
        score=_maybe_float(raw.get("score", raw.get("Score", raw.get("confidence", 0.0)))) or 0.0,
        confidence=_maybe_float(raw.get("confidence", raw.get("Confidence", raw.get("score")))),
        key_terms=[str(term).strip() for term in key_terms_raw if str(term).strip()],
        metadata=(raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}),
    )


def compute_leaderboard_score(daily_eval: Dict[str, Any]) -> float:
    hit = float(daily_eval.get("hit_at_k", 0.0))
    mrr = float(daily_eval.get("mrr", 0.0))
    return round((0.7 * hit) + (0.3 * mrr), 4)


def daily_cutoff_date(now_utc: datetime) -> str:
    return now_utc.astimezone(ZoneInfo("America/New_York")).date().isoformat()


def evaluate_previous_generation(
    strategy: Dict[str, Any],
    *,
    papers: list[PaperRecord],
    new_paper_ids: Set[str],
    evaluated_at: datetime,
) -> Optional[Dict[str, Any]]:
    generation = strategy.get("generation") or {}
    cutoff_date_raw = str(generation.get("cutoff_date") or "").strip()
    cutoff_month_raw = str(generation.get("cutoff_month") or "").strip()
    predictions_raw = generation.get("predictions")
    if not isinstance(predictions_raw, list) or not predictions_raw:
        return None

    if cutoff_date_raw:
        try:
            cutoff_date = normalize_date(cutoff_date_raw)
        except ValueError:
            if not cutoff_month_raw:
                return None
            cutoff_date = month_start_date(cutoff_month_raw)
    elif cutoff_month_raw:
        cutoff_date = month_start_date(cutoff_month_raw)
    else:
        return None

    cutoff_month = normalize_month(cutoff_date)
    cutoff_ord = date_to_ordinal(cutoff_date)
    train = [
        paper
        for paper in papers
        if date_to_ordinal(get_paper_published_date(paper)) <= cutoff_ord
    ]
    future = [
        paper
        for paper in papers
        if paper.paper_id in new_paper_ids and date_to_ordinal(get_paper_published_date(paper)) > cutoff_ord
    ]

    predictions = [
        coerce_prediction(raw, idx + 1)
        for idx, raw in enumerate(predictions_raw)
        if isinstance(raw, dict)
    ]
    if not predictions:
        return None

    params = strategy.get("params") or {}
    top_k_raw = (strategy.get("config") or {}).get("top_k", len(predictions))
    try:
        top_k = max(1, int(top_k_raw))
    except (TypeError, ValueError):
        top_k = max(1, len(predictions))

    evaluation = evaluate_predictions(
        predictions=predictions,
        train_papers=train,
        future_papers=future,
        k=top_k,
        similarity_config_path=str(params.get("similarity_config", "similarity.yaml")),
        model_name=(
            str(params.get("model_name"))
            if params.get("model_name") not in {None, ""}
            else None
        ),
    )
    return {
        "evaluated_at": _iso(evaluated_at),
        "prediction_cutoff_date": cutoff_date,
        "prediction_cutoff_month": cutoff_month,
        "new_papers_count": len(future),
        "prediction_count": len(predictions),
        **asdict(evaluation),
    }
