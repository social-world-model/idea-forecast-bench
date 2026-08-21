"""Realization RL pipeline: orchestration.

The episode/prompt, candidate-generation, context-caching and alignment-gate
stages moved to sibling modules; their public names are re-exported here so
existing `from forecaster.realization.pipeline import ...` keeps working.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from forecaster.config import RealizationConfig, load_realization_config
from forecaster.models import (
    HindsightSample,
    strict_runtime_manifest_contract,
)
from forecaster.realization.alignment_gate import (  # noqa: F401  re-export
    run_online_alignment_gate,
)
from forecaster.realization.candidate_generation import (  # noqa: F401  re-export
    generate_episode_candidate_lists,
    serialize_episode_candidate_lists,
)
from forecaster.realization.config import (
    CandidateGenerationConfig,
    EpisodeBuildConfig,
    RewardConfig,
    SelectionConfig,
)
from forecaster.realization.episode_prompts import (  # noqa: F401  re-export
    build_grpo_prompt_rows,
    build_strict_rl_prompt_rows,
)
from forecaster.realization.io import _write_json
from forecaster.realization.model_zoo import list_small_model_payloads
from forecaster.realization.rl_context import (  # noqa: F401  re-export
    prepare_common_rl_context,
)
from forecaster.realization.trainers import (
    create_trainer_runner,
)
from live_idea_bench.models import PaperRecord

logger = logging.getLogger(__name__)


def _require_train_split_for_training(split: str, prepare_only: bool) -> None:
    normalized_split = split.strip().lower()
    if prepare_only:
        return
    if normalized_split != "train":
        raise ValueError(
            "RL training is restricted to the train split. "
            "Use prepare_only if you need to inspect validation/test/all artifacts."
        )


def run_policy_rl_pipeline(
    papers: list[PaperRecord],
    *,
    trainer: str = "grpo",
    model_name: str,
    output_dir: str,
    episode_config: EpisodeBuildConfig,
    candidate_config: CandidateGenerationConfig,
    realization_config: RealizationConfig | None = None,
    reward_config: RewardConfig,
    reward_config_path: str = "reward.yaml",
    selection_config: SelectionConfig,
    trainer_config: Any,
    trainer_config_path: str,
    selection_config_path: str,
    split: str = "train",
    max_episodes: int | None = None,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    strict_mode: bool = False,
    prepare_only: bool = False,
    init_policy_path: str | None = None,
    skip_alignment_check: bool = False,
    hindsight_samples: list[HindsightSample] | None = None,
) -> dict[str, Any]:
    _require_train_split_for_training(split, prepare_only)
    runner = create_trainer_runner(trainer)
    target_dir = Path(output_dir).resolve()
    resolved_realization_config = realization_config or load_realization_config()
    common_context = prepare_common_rl_context(
        papers,
        model_name=model_name,
        output_dir=output_dir,
        episode_config=episode_config,
        candidate_config=candidate_config,
        realization_config=resolved_realization_config,
        reward_config=reward_config,
        selection_config=selection_config,
        split=split,
        max_episodes=max_episodes,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
        strict_mode=strict_mode,
        hindsight_samples=hindsight_samples,
    )
    prepared = runner.prepare(
        common_context,
        model_name=model_name,
        candidate_config=candidate_config,
        reward_config=reward_config,
        trainer_config=trainer_config,
    )

    diagnostics: dict[str, Any] = {}
    if runner.trainer_name == "grpo" and not skip_alignment_check and not prepare_only:
        diagnostics = run_online_alignment_gate(
            common_context,
            model_name=model_name,
            init_policy_path=init_policy_path,
            candidate_config=candidate_config,
            realization_config=resolved_realization_config,
            reward_config=reward_config,
            trainer_config=trainer_config,
            trainer_output_dir=prepared.output_dir,
            strict_mode=strict_mode,
        )
        if not diagnostics.get("alignment_passed", False):
            raise ValueError(
                f"{runner.trainer_name.upper()} reward alignment check failed with rho="
                f"{diagnostics.get('alignment_rho', 0.0)}"
            )

    trainer_manifest: dict[str, Any] | None = None
    if not prepare_only:
        trainer_manifest = runner.train(
            prepared,
            config=trainer_config,
            model_name=model_name,
            predictor_config=candidate_config.predictor_config,
            output_dir=str(prepared.output_dir),
            reward_config=reward_config,
            reward_config_path=reward_config_path,
            similarity_config_path=similarity_config_path,
            runtime_config_path=runtime_config_path,
            trainer_config_path=trainer_config_path,
            selection_config=selection_config,
            selection_config_path=selection_config_path,
            init_policy_path=init_policy_path,
            diagnostics=diagnostics or None,
        )

    manifest = {
        "pipeline_manifest_version": 2,
        "trainer": runner.trainer_name,
        "trainer_backend": runner.backend_name,
        "model_name": model_name,
        "split": split,
        "training_split_policy": "train_only",
        "selected_episode_count": len(common_context.selected_episodes),
        "shared_manifest_path": str(common_context.shared_manifest_path),
        "episodes_path": str(common_context.episodes_path),
        "prompt_rows_path": str(common_context.prompt_rows_path),
        "trainer_dataset_path": str(prepared.dataset_path.resolve()),
        "trainer_output_dir": str(prepared.output_dir.resolve()),
        "trainer_policy_manifest_path": str(
            (prepared.output_dir / "policy_manifest.json").resolve()
        )
        if trainer_manifest
        else "",
        "prepare_only": prepare_only,
        "selection_config_path": selection_config_path,
        "prompt_mode": "strict_interactive_realization"
        if strict_mode
        else "z_conditioned_realization",
        "strict_mode": strict_mode,
        "recommended_small_models": list_small_model_payloads(),
        "shared_fingerprint": common_context.config_fingerprint,
        "trainer_metadata": prepared.metadata,
        "diagnostics": diagnostics,
        "strict_contract": strict_runtime_manifest_contract(),
    }
    _write_json(target_dir / "pipeline_manifest.json", manifest)
    return manifest
