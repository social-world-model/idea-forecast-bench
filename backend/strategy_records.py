"""Normalisation of stored strategy records.

Every field that reaches these functions came out of a JSON file on disk, so
they coerce defensively rather than trusting the shape. Split out of
strategy_store.py, which had grown past the repo's 800-line cap; the store
itself is now about persistence and running jobs.
"""

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from typing import SupportsFloat, SupportsInt

DEFAULT_MODEL_NAME = "gpt-4o"


def _coerce_float(value: object) -> float | None:
    """Best-effort float, mirroring _coerce_int's tolerance of raw JSON.

    Returns None rather than raising: a stored strategy carrying a junk
    temperature used to make _read() swallow the exception and drop the
    record entirely, so one bad field silently deleted a whole strategy.
    """
    if value is None:
        return None
    try:
        return float(cast("SupportsFloat | str", value))
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object, default: int) -> int:
    try:
        # Deliberate duck-typing of an untyped JSON value; the cast keeps the
        # runtime call unchanged and the failure modes are caught below.
        return int(cast("SupportsInt | str", value))
    except (TypeError, ValueError):
        return default


def _normalize_params(
    strategy_name: str,
    params: object,
    *,
    strict_keyword_coercion: bool = False,
) -> dict[str, Any]:
    raw_params = params if isinstance(params, dict) else {}
    normalized_params = dict(raw_params)

    if strategy_name == "topic_trend":
        if strict_keyword_coercion:
            normalized_params["recent_months"] = int(raw_params.get("recent_months", 3))
            normalized_params["min_keyword_freq"] = int(
                raw_params.get("min_keyword_freq", 2)
            )
        else:
            normalized_params["recent_months"] = _coerce_int(
                raw_params.get("recent_months", 3), 3
            )
            normalized_params["min_keyword_freq"] = _coerce_int(
                raw_params.get("min_keyword_freq", 2), 2
            )
    else:
        if raw_params.get("model_name") in {None, ""} and raw_params.get(
            "model_id"
        ) not in {None, ""}:
            normalized_params["model_name"] = raw_params.get("model_id")
        normalized_params.setdefault("model_name", DEFAULT_MODEL_NAME)
        normalized_params.setdefault("predictor_config", "predictor.yaml")
        normalized_params.setdefault("similarity_config", "similarity.yaml")
        temperature = _coerce_float(raw_params.get("temperature"))
        if temperature is not None:
            normalized_params["temperature"] = temperature

    normalized_params.pop("model_id", None)
    normalized_params.pop("prompt_id", None)
    normalized_params.pop("prompt_version", None)

    return normalized_params


def _normalize_prediction(raw: object, rank_fallback: int) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    key_terms_raw = raw.get("key_terms") or raw.get("keywords") or []
    if not isinstance(key_terms_raw, list):
        key_terms_raw = []

    def _coerce_score(value: object, default: float = 0.0) -> float:
        try:
            # Same duck-typing as _coerce_int: JSON scores may arrive as str.
            resolved = float(cast("SupportsFloat | str", value))
        except (TypeError, ValueError):
            resolved = default
        if resolved > 1.0:
            resolved = resolved / 10.0
        return round(min(1.0, max(0.0, resolved)), 4)

    confidence = _coerce_score(
        raw.get(
            "confidence", raw.get("Confidence", raw.get("score", raw.get("Score", 0.0)))
        ),
        default=0.0,
    )
    score = _coerce_score(
        raw.get("score", raw.get("Score", confidence)), default=confidence
    )

    try:
        rank = int(raw.get("rank", rank_fallback))
    except (TypeError, ValueError):
        rank = rank_fallback

    return {
        "rank": rank,
        "title": str(raw.get("title") or raw.get("Title") or ""),
        "rationale": str(raw.get("rationale") or raw.get("Rationale") or ""),
        "approach": str(raw.get("approach") or raw.get("Approach") or ""),
        "score": score,
        "confidence": confidence,
        "key_terms": [str(term).strip() for term in key_terms_raw if str(term).strip()],
        "metadata": raw.get("metadata")
        if isinstance(raw.get("metadata"), dict)
        else {},
    }


def _normalize_generation(generation: object) -> object:
    if not isinstance(generation, dict):
        return generation

    predictions_raw = generation.get("predictions")
    predictions = []
    if isinstance(predictions_raw, list):
        for idx, raw in enumerate(predictions_raw, start=1):
            normalized = _normalize_prediction(raw, idx)
            if normalized is not None:
                predictions.append(normalized)

    normalized = dict(generation)
    if predictions:
        normalized["predictions"] = predictions
    cutoff_month = str(normalized.get("cutoff_month") or "").strip()
    cutoff_date = str(normalized.get("cutoff_date") or "").strip()
    if not cutoff_date and cutoff_month:
        normalized["cutoff_date"] = f"{cutoff_month}-01"
    return normalized


def _normalize_evaluation(evaluation: object) -> object:
    if not isinstance(evaluation, dict):
        return evaluation
    normalized = dict(evaluation)
    normalized.setdefault("matched_paper_ids", [])
    normalized.setdefault("matched_prediction_ranks", [])
    normalized.setdefault("lead_time", 0.0)
    normalized.setdefault("duplicate_rate", 0.0)
    normalized.pop("matched_terms", None)
    return normalized


def _normalize_backtest_result(backtest_result: object) -> object:
    if not isinstance(backtest_result, dict):
        return backtest_result
    normalized = dict(backtest_result)
    windows = normalized.get("windows")
    if isinstance(windows, list):
        normalized_windows = []
        for window in windows:
            if not isinstance(window, dict):
                continue
            updated = dict(window)
            predictions_raw = updated.get("predictions")
            if isinstance(predictions_raw, list):
                predictions = []
                for idx, raw in enumerate(predictions_raw, start=1):
                    pred = _normalize_prediction(raw, idx)
                    if pred is not None:
                        predictions.append(pred)
                updated["predictions"] = predictions
            updated["evaluation"] = _normalize_evaluation(updated.get("evaluation"))
            normalized_windows.append(updated)
        normalized["windows"] = normalized_windows
    return normalized


def _normalize_status(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"pending", "running", "done", "failed"}:
        return raw
    return "pending"


def _normalize_topic_run(topic_run: object) -> dict[str, Any] | None:
    if not isinstance(topic_run, dict):
        return None

    normalized = dict(topic_run)
    normalized["topic_id"] = str(normalized.get("topic_id") or "").strip()
    normalized["topic_name"] = str(normalized.get("topic_name") or "").strip()
    normalized["matched_paper_count"] = _coerce_int(
        normalized.get("matched_paper_count", 0), 0
    )
    normalized["generation_status"] = _normalize_status(
        normalized.get("generation_status")
    )
    normalized["backtest_status"] = _normalize_status(normalized.get("backtest_status"))
    normalized["generation"] = _normalize_generation(normalized.get("generation"))
    normalized["backtest_result"] = _normalize_backtest_result(
        normalized.get("backtest_result")
    )
    normalized["generation_error"] = (
        str(normalized.get("generation_error")).strip()
        if normalized.get("generation_error") not in {None, ""}
        else None
    )
    normalized["backtest_error"] = (
        str(normalized.get("backtest_error")).strip()
        if normalized.get("backtest_error") not in {None, ""}
        else None
    )
    if not normalized["topic_id"] or not normalized["topic_name"]:
        return None
    return normalized


def _normalize_topic_runs(topic_runs: object) -> list[dict[str, Any]]:
    if not isinstance(topic_runs, list):
        return []

    normalized: list[dict[str, Any]] = []
    for raw in topic_runs:
        topic_run = _normalize_topic_run(raw)
        if topic_run is not None:
            normalized.append(topic_run)
    return normalized


def _normalize_strategy(strategy: dict[str, Any]) -> dict[str, Any]:
    strategy_name = str(strategy.get("strategy_name") or "topic_trend")
    if strategy_name == "prompt_llm":
        strategy_name = "predictor_llm"
    normalized = dict(strategy)
    normalized["strategy_name"] = strategy_name
    normalized["params"] = _normalize_params(strategy_name, strategy.get("params"))
    normalized["generation"] = _normalize_generation(strategy.get("generation"))
    normalized["backtest_result"] = _normalize_backtest_result(
        strategy.get("backtest_result")
    )
    normalized["topic_runs"] = _normalize_topic_runs(strategy.get("topic_runs"))
    normalized["daily_evaluation"] = _normalize_evaluation(
        strategy.get("daily_evaluation")
    )
    normalized.setdefault("leaderboard_score", None)
    normalized.setdefault("last_daily_run_at", None)
    normalized.setdefault("last_generation_cutoff_month", None)
    return normalized
