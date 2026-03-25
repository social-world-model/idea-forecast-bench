from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from live_idea_bench.models import PaperRecord
from forecaster.config import RealizationConfig, load_realization_config
from forecaster.models import innovation_from_dict
from forecaster.realization.config import load_reward_config
from forecaster.realization.reward import (
    build_invalid_reward_evaluation,
    coerce_reward_prediction,
    evaluate_rl_reward,
)


@lru_cache(maxsize=8)
def _cached_reward_config(path: str) -> Any:
    return load_reward_config(path)


def _coerce_extra_info(extra_info: Any) -> dict[str, Any]:
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str):
        try:
            payload = json.loads(extra_info)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, dict):
            return payload
    return {}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | str | None = None,
    reward_config_path: str = "reward.yaml",
    realization_config_path: str = "realization.yaml",
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    model_name: str | None = None,
) -> float:
    del data_source, ground_truth
    reward_config = _cached_reward_config(reward_config_path)
    payload = _coerce_extra_info(extra_info)
    try:
        train_papers = [PaperRecord(**row) for row in payload.get("train_papers", [])]
        future_papers = [PaperRecord(**row) for row in payload.get("future_papers", [])]
        evidence_papers = [PaperRecord(**row) for row in payload.get("evidence_papers", [])]
    except (TypeError, KeyError):
        return build_invalid_reward_evaluation(reward_config).list_reward
    try:
        innovation = (
            innovation_from_dict(payload.get("innovation", {}))
            if isinstance(payload.get("innovation", {}), dict) and payload.get("innovation")
            else None
        )
    except (TypeError, KeyError, ValueError):
        innovation = None
    try:
        realization_config_payload = payload.get("realization_config", {})
        realization_config = (
            RealizationConfig(**realization_config_payload)
            if isinstance(realization_config_payload, dict) and realization_config_payload
            else load_realization_config(realization_config_path)
        )
    except (TypeError, ValueError):
        realization_config = load_realization_config(realization_config_path)
    prediction, proposal_text = coerce_reward_prediction(
        solution_str,
        prompt_mode=str(payload.get("prompt_mode", "") or ""),
        innovation=innovation,
    )
    if prediction is None:
        return build_invalid_reward_evaluation(reward_config).list_reward
    evaluation = evaluate_rl_reward(
        predictions=[prediction],
        train_papers=train_papers,
        future_papers=future_papers,
        reward_config=reward_config,
        innovation=innovation,
        evidence_papers=evidence_papers,
        proposal_text=proposal_text,
        realization_config=realization_config,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
        model_name=model_name,
        cutoff_date=str(payload.get("cutoff_date", "") or "") or None,
        future_end_date=str(payload.get("future_end_date", "") or "") or None,
    )
    return evaluation.list_reward
