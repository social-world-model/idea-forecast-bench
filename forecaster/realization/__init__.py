from __future__ import annotations

# NOTE: candidates, episodes, grpo, and pipeline are loaded lazily because
# episodes.py imports live_idea_bench.backtest, which in turn imports
# forecaster.realization.config — creating a circular import if these modules
# were loaded eagerly when the package initializes.
from importlib import import_module
from typing import Any

from forecaster.realization.config import (
    CandidateGenerationConfig,
    EpisodeBuildConfig,
    GRPOTrainConfig,
    RewardConfig,
    RewardWeights,
    SelectionConfig,
    load_candidate_generation_config,
    load_episode_build_config,
    load_grpo_train_config,
    load_reward_config,
    load_selection_config,
)
from forecaster.realization.local_generation import (
    build_prediction_prompt,
    generate_local_predictions,
    parse_completion_predictions,
    parse_single_completion_prediction,
)
from forecaster.realization.model_zoo import (
    SmallModelSpec,
    list_small_model_payloads,
    list_small_model_specs,
    resolve_small_model,
)
from forecaster.realization.reward import (
    RLRewardEvaluation,
    evaluate_rl_reward,
    serialize_reward_evaluation,
    spearman_correlation,
)
from forecaster.realization.selection import select_top_k_predictions
from forecaster.realization.trainers import (
    PreparedRLContext,
    RLTrainerRunner,
    TrainerPreparedArtifacts,
    create_trainer_runner,
)

_LAZY = {
    "CandidateListSample": ("forecaster.realization.candidates", "CandidateListSample"),
    "EpisodeCandidateLists": (
        "forecaster.realization.candidates",
        "EpisodeCandidateLists",
    ),
    "RLEpisode": ("forecaster.realization.episodes", "RLEpisode"),
    "build_rl_episodes": ("forecaster.realization.episodes", "build_rl_episodes"),
    "serialize_episodes": ("forecaster.realization.episodes", "serialize_episodes"),
    "RewardAlignmentReport": ("forecaster.realization.grpo", "RewardAlignmentReport"),
    "build_grpo_advantages": ("forecaster.realization.grpo", "build_grpo_advantages"),
    "compute_reward_alignment": (
        "forecaster.realization.grpo",
        "compute_reward_alignment",
    ),
    "build_grpo_prompt_rows": (
        "forecaster.realization.pipeline",
        "build_grpo_prompt_rows",
    ),
    "build_strict_rl_prompt_rows": (
        "forecaster.realization.pipeline",
        "build_strict_rl_prompt_rows",
    ),
    "generate_episode_candidate_lists": (
        "forecaster.realization.pipeline",
        "generate_episode_candidate_lists",
    ),
    "prepare_common_rl_context": (
        "forecaster.realization.pipeline",
        "prepare_common_rl_context",
    ),
    "run_policy_rl_pipeline": (
        "forecaster.realization.pipeline",
        "run_policy_rl_pipeline",
    ),
    "serialize_episode_candidate_lists": (
        "forecaster.realization.pipeline",
        "serialize_episode_candidate_lists",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    # config
    "CandidateGenerationConfig",
    "EpisodeBuildConfig",
    "GRPOTrainConfig",
    "RewardConfig",
    "RewardWeights",
    "SelectionConfig",
    "load_candidate_generation_config",
    "load_episode_build_config",
    "load_grpo_train_config",
    "load_reward_config",
    "load_selection_config",
    # local_generation
    "build_prediction_prompt",
    "generate_local_predictions",
    "parse_completion_predictions",
    "parse_single_completion_prediction",
    # model_zoo
    "SmallModelSpec",
    "list_small_model_payloads",
    "list_small_model_specs",
    "resolve_small_model",
    # reward
    "RLRewardEvaluation",
    "evaluate_rl_reward",
    "serialize_reward_evaluation",
    "spearman_correlation",
    # selection
    "select_top_k_predictions",
    # trainers
    "PreparedRLContext",
    "RLTrainerRunner",
    "TrainerPreparedArtifacts",
    "create_trainer_runner",
    # lazy (backtest dependency via episodes.py)
    "CandidateListSample",
    "EpisodeCandidateLists",
    "RLEpisode",
    "build_rl_episodes",
    "serialize_episodes",
    "RewardAlignmentReport",
    "build_grpo_advantages",
    "compute_reward_alignment",
    "build_grpo_prompt_rows",
    "build_strict_rl_prompt_rows",
    "generate_episode_candidate_lists",
    "prepare_common_rl_context",
    "run_policy_rl_pipeline",
    "serialize_episode_candidate_lists",
]
