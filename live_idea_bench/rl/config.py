from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RL_CONFIG_DIR = PROJECT_ROOT / "config" / "rl"


@dataclass
class RewardWeights:
    future_match: float = 0.5
    novelty: float = 0.15
    specificity: float = 0.15
    lead_time: float = 0.1
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
    past_window_months: int | None = 24
    horizon_months: int = 6
    step_months: int = 3
    min_train_papers: int = 6
    start_month: str | None = None
    end_month: str | None = None
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    similarity_config: str = "similarity.yaml"


@dataclass
class CandidateGenerationConfig:
    num_candidate_lists: int = 8
    ideas_per_list: int = 20
    backend: str = "auto"
    predictor_config: str = "predictor.yaml"
    min_temperature: float = 0.55
    max_temperature: float = 1.15
    top_p: float = 0.9
    top_k: int = 40
    max_new_tokens: int = 1536
    repetition_penalty: float = 1.05
    seed: int = 7
    enable_thinking: bool | None = False


@dataclass
class DPOTrainConfig:
    quantile_fraction: float = 0.25
    beta: float = 0.1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 5e-5
    max_length: int = 4096
    logging_steps: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    dry_run: bool = False


@dataclass
class GRPOTrainConfig:
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 2e-6
    num_generations: int = 8
    max_completion_length: int = 1024
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    reward_alignment_threshold: float = 0.5
    logging_steps: int = 1
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
    if path.parts[:-1]:  # has parent dirs
        raise ValueError(
            f"Relative RL config path must be a plain filename, not a path with directories: {raw!r}. "
            "Use an absolute path to reference files outside the default config directory."
        )
    return (DEFAULT_RL_CONFIG_DIR / path.name).resolve()


def _load_model_config(name_or_path: str, model_class: type[Any]) -> Any:
    payload = _read_yaml(_resolve_rl_config_path(name_or_path))
    if model_class is RewardConfig:
        weights_payload = payload.pop("weights", {}) or {}
        if not isinstance(weights_payload, dict):
            raise ValueError("reward weights must be a mapping")
        try:
            return RewardConfig(weights=RewardWeights(**weights_payload), **payload)
        except TypeError as exc:
            raise ValueError(
                f"Invalid config for {RewardConfig.__name__}: {exc}. "
                f"Check the YAML file at {name_or_path!r} for unknown or missing keys."
            ) from exc
    try:
        return model_class(**payload)
    except TypeError as exc:
        raise ValueError(
            f"Invalid config for {model_class.__name__}: {exc}. "
            f"Check the YAML file at {name_or_path!r} for unknown or missing keys."
        ) from exc


def load_reward_config(name_or_path: str = "reward.yaml") -> RewardConfig:
    return _load_model_config(name_or_path, RewardConfig)


def load_episode_build_config(name_or_path: str = "episode_build.yaml") -> EpisodeBuildConfig:
    return _load_model_config(name_or_path, EpisodeBuildConfig)


def load_candidate_generation_config(name_or_path: str = "candidate_generation.yaml") -> CandidateGenerationConfig:
    return _load_model_config(name_or_path, CandidateGenerationConfig)


def load_dpo_train_config(name_or_path: str = "dpo_train.yaml") -> DPOTrainConfig:
    return _load_model_config(name_or_path, DPOTrainConfig)


def load_grpo_train_config(name_or_path: str = "grpo_train.yaml") -> GRPOTrainConfig:
    return _load_model_config(name_or_path, GRPOTrainConfig)
