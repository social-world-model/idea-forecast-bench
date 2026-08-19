from __future__ import annotations

import dataclasses
from typing import Any

from live_idea_bench.backtest import BacktestConfig, backtest
from live_idea_bench.models import PaperRecord
from live_idea_bench.papers import date_to_ordinal, get_paper_published_date, normalize_date, normalize_month
from live_idea_bench.strategy.registry import create_strategy


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _legacy_to_current_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    if "model_name" not in normalized and normalized.get("model_id"):
        normalized["model_name"] = normalized.get("model_id")
    if "predictor_config" not in normalized and (
        normalized.get("prompt_id") or normalized.get("prompt_version")
    ):
        normalized["predictor_config"] = "predictor.yaml"
    normalized.setdefault("similarity_config", "similarity.yaml")
    normalized.setdefault("predictor_config", "predictor.yaml")
    return normalized


def build_strategy(strategy_record: dict[str, Any]):
    strategy_name = str(strategy_record.get("strategy_name") or "keyword_trend")
    params = _legacy_to_current_params(strategy_record.get("params") or {})
    if strategy_name == "prompt_llm":
        strategy_name = "predictor_llm"

    return create_strategy(
        strategy_name,
        recent_months=_coerce_int(params.get("recent_months", 3), 3),
        min_keyword_freq=_coerce_int(params.get("min_keyword_freq", 2), 2),
        model_name=(
            str(params.get("model_name"))
            if params.get("model_name") not in {None, ""}
            else None
        ),
        predictor_config=str(params.get("predictor_config", "predictor.yaml")),
        similarity_config=str(params.get("similarity_config", "similarity.yaml")),
        temperature=(
            float(params.get("temperature"))
            if params.get("temperature") is not None
            else None
        ),
        policy_manifest_path=(
            str(params.get("policy_manifest_path"))
            if params.get("policy_manifest_path") not in {None, ""}
            else None
        ),
    )


def run_strategy_backtest(
    strategy_record: dict[str, Any],
    papers: list[PaperRecord],
) -> dict[str, Any]:
    config = strategy_record.get("config") or {}
    params = _legacy_to_current_params(strategy_record.get("params") or {})
    strategy = build_strategy(strategy_record)
    backtest_config = BacktestConfig(
        top_k=int(config.get("top_k", 5)),
        horizon_months=int(config.get("horizon_months", 3)),
        min_train_papers=int(config.get("min_train_papers", 6)),
        start_month=config.get("start_month"),
        end_month=config.get("end_month"),
        similarity_config=str(params.get("similarity_config", "similarity.yaml")),
    )
    return backtest(
        papers,
        strategy,
        backtest_config,
        model_name=(
            str(params.get("model_name"))
            if params.get("model_name") not in {None, ""}
            else None
        ),
    )


def run_strategy_generation(
    strategy_record: dict[str, Any],
    papers: list[PaperRecord],
    *,
    cutoff_date: str,
) -> dict[str, Any]:
    config = strategy_record.get("config") or {}
    top_k = int(config.get("top_k", 5))
    resolved_cutoff_date = normalize_date(cutoff_date)
    cutoff_month = normalize_month(resolved_cutoff_date)
    cutoff_ord = date_to_ordinal(resolved_cutoff_date)

    train_papers = [
        paper
        for paper in papers
        if date_to_ordinal(get_paper_published_date(paper)) <= cutoff_ord
    ]

    strategy = build_strategy(strategy_record)
    predictions_raw = strategy.generate(
        train_papers=train_papers,
        cutoff_month=cutoff_month,
        top_k=top_k,
    )
    predictions = [
        dataclasses.asdict(prediction)
        if dataclasses.is_dataclass(prediction)
        else (
            prediction._asdict()
            if hasattr(prediction, "_asdict")
            else (dict(prediction) if hasattr(prediction, "keys") else str(prediction))
        )
        for prediction in predictions_raw
    ]

    return {
        "cutoff_date": resolved_cutoff_date,
        "cutoff_month": cutoff_month,
        "predictions": predictions,
    }
