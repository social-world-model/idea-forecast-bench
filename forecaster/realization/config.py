from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RL_CONFIG_DIR = PROJECT_ROOT / "config" / "forecaster"


@dataclass
class RewardWeights:
    future_match: float = 0.7
    novelty: float = 0.05
    specificity: float = 0.1
    lead_time: float = 0.15
    duplicate_penalty: float = 0.0
    popularity: float = 0.0  # Opt-in: weight for matched paper popularity in RL reward


@dataclass
class RewardConfig:
    top_k: int = 1
    candidate_limit: int | None = None
    duplicate_similarity_threshold: float = 0.8
    invalid_completion_reward: float = -0.05
    specificity_title_weight: float = 0.2
    specificity_rationale_weight: float = 0.4
    specificity_approach_weight: float = 0.4
    rank_decay: float = 0.15
    benchmark_score_weight: float = 0.0
    weights: RewardWeights = field(default_factory=RewardWeights)
    # When set, completely replaces the composite reward with a single
    # eval metric for GRPO training: "soft" | "coverage" | "novelty".
    # Default "composite" keeps the legacy mixed reward.
    mode: str = "composite"
    judge_top_r: int = 10
    cluster_k: int = 5


@dataclass
class EpisodeBuildConfig:
    top_k: int = 5
    past_window_months: int | None = 24
    horizon_months: int = 3
    step_months: int = 3
    min_train_papers: int = 6
    start_month: str | None = None
    end_month: str | None = None
    validation_start_month: str | None = None
    test_start_month: str | None = None
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
class SelectionConfig:
    candidate_pool_size: int = 24
    output_top_k: int = 5
    dedup_similarity_threshold: float = 0.8
    temperature_schedule: list[float] = field(default_factory=lambda: [0.35, 0.55, 0.75, 0.95])
    top_p_schedule: list[float] = field(default_factory=lambda: [0.8, 0.9, 0.95])
    enable_context_shuffle: bool = True
    relevance_frequency_weight: float = 0.5
    relevance_confidence_weight: float = 0.3
    relevance_heuristic_weight: float = 0.2
    mmr_relevance_weight: float = 0.7
    mmr_diversity_weight: float = 0.3


@dataclass
class OnlineRLTrainConfig:
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 2e-6
    num_generations: int = 8
    max_prompt_length: int = 4096
    max_completion_length: int = 1024
    kl_coef: float = 0.001
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.4
    logging_steps: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    reward_alignment_threshold: float = 0.5
    dry_run: bool = False
    # Phase-4 switch: "legacy" preserves the existing composite reward via
    # forecaster.realization.verl.reward_fn.compute_score; "foresight" routes
    # rewards through forecaster.foresight.reward.compute_score_v2.
    reward_mode: str = "legacy"
    # When reward_mode == "foresight", load CutoffIndexBundles + rubrics from this dir.
    foresight_artifact_dir: str = ""
    # When reward_mode == "foresight", which embedder + judge backend to use.
    foresight_embedder: str = "sentence-transformer:all-MiniLM-L6-v2"
    foresight_judge_mode: str = "live"   # "live" | "stub" (stub used only in tests)
    # Phase-5: in-group dedup penalty. Subtracted from each rollout's reward
    # for every near-duplicate sibling within its group (Jaccard >= threshold).
    dedup_penalty: float = 0.0
    dedup_jaccard_threshold: float = 0.85
    # Phase-5: enforce (cutoff_t, z) grouping invariant in the reward callback.
    # Set False only for debug; production training must keep this on.
    grouping_assert: bool = True
    # Phase-6: rubric refresh interval (in trainer steps). 0 = disabled.
    rubric_refresh_every: int = 0
    rubric_refresh_auc_min: float = 0.70


@dataclass
class GRPOTrainConfig(OnlineRLTrainConfig):
    pass


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
        weights_payload = payload.get("weights", {}) or {}
        if not isinstance(weights_payload, dict):
            raise ValueError("reward weights must be a mapping")
        remaining = {k: v for k, v in payload.items() if k != "weights"}
        try:
            return RewardConfig(weights=RewardWeights(**weights_payload), **remaining)
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


def load_selection_config(name_or_path: str = "selection.yaml") -> SelectionConfig:
    return _load_model_config(name_or_path, SelectionConfig)


def load_grpo_train_config(name_or_path: str = "grpo_train.yaml") -> GRPOTrainConfig:
    return _load_model_config(name_or_path, GRPOTrainConfig)
