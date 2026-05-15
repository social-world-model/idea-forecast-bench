"""Configuration dataclasses and YAML loaders for the forecaster package.

Config dataclasses are mutable (not frozen), following the same pattern as
live_idea_bench/config.py and forecaster/realization/config.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORECASTER_CONFIG_DIR = PROJECT_ROOT / "config" / "forecaster"
STRICT_RUNTIME_MODE = "strict_eval"
STRICT_PRIOR_SCORE_METHOD = "conditional_logprob"
STRICT_REALIZATION_SCORE_METHOD = "conditional_logprob"
STRICT_JOINT_SCORE_MODE = "linear_blend"
STRICT_POPULARITY_WEIGHT = 0.0
STRICT_JOINT_SCORE_COMPONENTS: tuple[str, str] = ("prior_score", "realization_score")


# ---------------------------------------------------------------------------
# Config dataclasses (mutable, NOT frozen)
# ---------------------------------------------------------------------------


@dataclass
class HindsightConfig:
    llm_model: str = "gpt-4o"
    temperature: float = 0.2
    max_context_papers: int = 15
    max_retries: int = 2


@dataclass
class PriorConfig:
    memory_path: str = "data/memory_inventory.json"
    recency_decay: float = 0.9
    frequency_cap: int = 10
    utility_ema_alpha: float = 0.3


@dataclass
class SFTTrainConfig:
    model_alias: str = "qwen2.5-3b-instruct"
    num_epochs: int = 3
    learning_rate: float = 2e-5
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 2048
    max_memory_entries: int = 20
    output_dir: str = "output/prior_sft"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01


@dataclass
class RealizationConfig:
    context_top_k: int = 5
    evidence_top_k: int = 5
    evidence_similarity_threshold: float = 0.3
    proposal_max_tokens: int = 256
    allow_artifact_fallback_to_llm: bool = False
    evidence_accuracy_weight: float = 0.2
    operator_adherence_weight: float = 0.3
    coherence_weight: float = 0.5


@dataclass
class InferenceConfig:
    runtime_mode: str = "strict_eval"
    num_candidates: int = 8
    prior_weight: float = 0.4
    realization_weight: float = 0.6
    top_k: int = 5
    dedup_threshold: float = 0.8
    prior_temperature: float = 0.8
    prior_score_method: str = "conditional_logprob"
    realization_score_method: str = "conditional_logprob"
    score_normalization: str = "per_token"
    score_temperature: float = 1.0
    joint_score_mode: str = "linear_blend"
    popularity_weight: float = 0.0  # Opt-in: weight for popularity bonus in joint score

    def __post_init__(self) -> None:
        validate_inference_config(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {path}")

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML at {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"YAML at {path} must decode to a mapping")
    return payload


def _resolve_forecaster_config_path(name_or_path: str) -> Path:
    """Resolve a config name or path to an absolute Path.

    Plain filenames are resolved relative to DEFAULT_FORECASTER_CONFIG_DIR.
    Absolute paths are used as-is.
    Relative paths with parent directories are rejected.
    """
    raw = str(name_or_path or "").strip()
    if not raw:
        raise ValueError("Forecaster config path is required")
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.parts[:-1]:  # has parent dirs
        raise ValueError(
            f"Relative forecaster config path must be a plain filename, "
            f"not a path with directories: {raw!r}. "
            "Use an absolute path to reference files outside the default config directory."
        )
    return (DEFAULT_FORECASTER_CONFIG_DIR / path.name).resolve()


def _load_config(name_or_path: str, model_class: type[Any]) -> Any:
    payload = _read_yaml(_resolve_forecaster_config_path(name_or_path))
    try:
        return model_class(**payload)
    except TypeError as exc:
        raise ValueError(
            f"Invalid config for {model_class.__name__}: {exc}. "
            f"Check the YAML file at {name_or_path!r} for unknown or missing keys."
        ) from exc


def strict_inference_score_contract(
    *,
    score_normalization: str = "per_token",
    score_temperature: float = 1.0,
) -> dict[str, Any]:
    """Return the frozen strict inference score contract."""
    return {
        "prior_score_method": STRICT_PRIOR_SCORE_METHOD,
        "realization_score_method": STRICT_REALIZATION_SCORE_METHOD,
        "joint_score_mode": STRICT_JOINT_SCORE_MODE,
        "joint_score_formula": "linear_blend(prior_score, realization_score)",
        "score_normalization": str(score_normalization).strip().lower() or "per_token",
        "score_temperature": float(score_temperature),
        "popularity_weight": STRICT_POPULARITY_WEIGHT,
        "joint_score_components": STRICT_JOINT_SCORE_COMPONENTS,
        "allows_extra_bonus_terms": False,
    }


def validate_inference_config(config: InferenceConfig) -> None:
    """Validate strict runtime invariants for inference scoring."""
    runtime_mode = str(getattr(config, "runtime_mode", "") or "").strip().lower()
    if runtime_mode != STRICT_RUNTIME_MODE:
        return

    errors: list[str] = []
    if str(getattr(config, "prior_score_method", "") or "").strip().lower() != STRICT_PRIOR_SCORE_METHOD:
        errors.append(f"prior_score_method must be {STRICT_PRIOR_SCORE_METHOD!r}")
    if (
        str(getattr(config, "realization_score_method", "") or "").strip().lower()
        != STRICT_REALIZATION_SCORE_METHOD
    ):
        errors.append(f"realization_score_method must be {STRICT_REALIZATION_SCORE_METHOD!r}")
    if str(getattr(config, "joint_score_mode", "") or "").strip().lower() != STRICT_JOINT_SCORE_MODE:
        errors.append(f"joint_score_mode must be {STRICT_JOINT_SCORE_MODE!r}")
    popularity_weight = float(getattr(config, "popularity_weight", STRICT_POPULARITY_WEIGHT) or 0.0)
    if not math.isclose(popularity_weight, STRICT_POPULARITY_WEIGHT, abs_tol=1e-12):
        errors.append("popularity_weight must be 0.0")

    if errors:
        raise ValueError(
            "Strict inference config requires: " + "; ".join(errors) + "."
        )


# ---------------------------------------------------------------------------
# Public loader functions
# ---------------------------------------------------------------------------


def load_hindsight_config(name_or_path: str = "hindsight.yaml") -> HindsightConfig:
    """Load HindsightConfig from a YAML file."""
    return _load_config(name_or_path, HindsightConfig)


def load_prior_config(name_or_path: str = "prior.yaml") -> PriorConfig:
    """Load PriorConfig from a YAML file (uses only PriorConfig fields)."""
    payload = _read_yaml(_resolve_forecaster_config_path(name_or_path))
    prior_fields = {f for f in PriorConfig.__dataclass_fields__}
    filtered = {k: v for k, v in payload.items() if k in prior_fields}
    try:
        return PriorConfig(**filtered)
    except TypeError as exc:
        raise ValueError(
            f"Invalid config for PriorConfig: {exc}. "
            f"Check the YAML file at {name_or_path!r}."
        ) from exc


def load_sft_train_config(name_or_path: str = "prior.yaml") -> SFTTrainConfig:
    """Load SFTTrainConfig from a YAML file (uses only SFTTrainConfig fields)."""
    payload = _read_yaml(_resolve_forecaster_config_path(name_or_path))
    sft_fields = {f for f in SFTTrainConfig.__dataclass_fields__}
    filtered = {k: v for k, v in payload.items() if k in sft_fields}
    try:
        return SFTTrainConfig(**filtered)
    except TypeError as exc:
        raise ValueError(
            f"Invalid config for SFTTrainConfig: {exc}. "
            f"Check the YAML file at {name_or_path!r}."
        ) from exc


def load_realization_config(name_or_path: str = "realization.yaml") -> RealizationConfig:
    """Load RealizationConfig from a YAML file."""
    return _load_config(name_or_path, RealizationConfig)


def load_inference_config(name_or_path: str = "inference.yaml") -> InferenceConfig:
    """Load InferenceConfig from a YAML file."""
    return _load_config(name_or_path, InferenceConfig)
