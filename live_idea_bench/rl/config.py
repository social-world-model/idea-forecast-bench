from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RL_CONFIG_DIR = PROJECT_ROOT / "config" / "rl"


@dataclass
class RewardWeights:
    future_match: float = 0.6
    novelty: float = 0.15
    specificity: float = 0.15
    duplicate_penalty: float = 0.1


@dataclass
class RewardConfig:
    top_k: int = 5
    candidate_limit: int = 25
    duplicate_similarity_threshold: float = 0.8
    specificity_title_weight: float = 0.2
    specificity_rationale_weight: float = 0.4
    specificity_approach_weight: float = 0.4
    rank_decay: float = 0.15
    benchmark_score_weight: float = 0.4
    weights: RewardWeights = field(default_factory=RewardWeights)


@dataclass
class EpisodeBuildConfig:
    top_k: int = 5
    horizon_months: int = 3
    min_train_papers: int = 6
    start_month: str | None = None
    end_month: str | None = None
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    similarity_config: str = "similarity.yaml"


@dataclass
class DPOTrainConfig:
    num_candidate_lists: int = 8
    quantile_fraction: float = 0.25
    beta: float = 0.1
    per_device_batch_size: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 5e-5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    dry_run: bool = False


@dataclass
class GRPOTrainConfig:
    num_candidate_lists: int = 8
    kl_coef: float = 0.05
    clip_range: float = 0.2
    per_device_batch_size: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 3e-5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    reward_alignment_threshold: float = 0.5
    dry_run: bool = False


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML at {path} must decode to a mapping")
    return payload


def _resolve_rl_config_path(name_or_path: str) -> Path:
    raw = str(name_or_path or "").strip()
    if not raw:
        raise ValueError("RL config path is required")
    path = Path(raw)
    if path.is_absolute():
        return path
    return (DEFAULT_RL_CONFIG_DIR / path.name).resolve()


def _load_model_config(name_or_path: str, model_class: type[Any]) -> Any:
    payload = _read_yaml(_resolve_rl_config_path(name_or_path))
    if model_class is RewardConfig:
        weights_payload = payload.pop("weights", {}) or {}
        if not isinstance(weights_payload, dict):
            raise ValueError("reward weights must be a mapping")
        return RewardConfig(weights=RewardWeights(**weights_payload), **payload)
    return model_class(**payload)


def load_reward_config(name_or_path: str = "reward.yaml") -> RewardConfig:
    return _load_model_config(name_or_path, RewardConfig)


def load_episode_build_config(name_or_path: str = "episode_build.yaml") -> EpisodeBuildConfig:
    return _load_model_config(name_or_path, EpisodeBuildConfig)


def load_dpo_train_config(name_or_path: str = "dpo_train.yaml") -> DPOTrainConfig:
    return _load_model_config(name_or_path, DPOTrainConfig)


def load_grpo_train_config(name_or_path: str = "grpo_train.yaml") -> GRPOTrainConfig:
    return _load_model_config(name_or_path, GRPOTrainConfig)
