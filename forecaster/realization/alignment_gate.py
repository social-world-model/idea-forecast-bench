from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from forecaster.config import RealizationConfig
from forecaster.models import (
    Innovation,
)
from forecaster.realization.candidate_generation import (
    _generate_realization_candidate_predictions,
    _generate_strict_realization_completion,
    _single_idea_candidate_config,
    generate_episode_candidate_lists,
)
from forecaster.realization.config import (
    CandidateGenerationConfig,
    RewardConfig,
)
from forecaster.realization.episode_prompts import (
    build_grpo_prompt_rows,
    build_strict_rl_prompt_rows,
)
from forecaster.realization.episodes import (
    RLEpisode,
)
from forecaster.realization.grpo import compute_reward_alignment
from forecaster.realization.io import _write_json
from forecaster.realization.reward import (
    build_invalid_reward_evaluation,
    evaluate_rl_reward,
    evaluate_strict_rl_reward,
)
from forecaster.realization.trainers import (
    PreparedRLContext,
)
from idea_forecast_bench.models import PaperRecord

logger = logging.getLogger(__name__)


def _midpoint_temperature(config: CandidateGenerationConfig) -> float:
    return round((config.min_temperature + config.max_temperature) / 2.0, 4)


def _resolve_policy_source(
    *,
    model_name: str,
    init_policy_path: str | None,
) -> tuple[str, str | None]:
    if not init_policy_path:
        return model_name, None
    path = Path(init_policy_path).expanduser()
    if path.exists():
        return str(path.resolve()), model_name
    return init_policy_path, None


def _average_metric(evaluations: list[Any], name: str) -> float:
    if not evaluations:
        return 0.0
    return round(
        sum(float(getattr(evaluation, name)) for evaluation in evaluations)
        / len(evaluations),
        4,
    )


def _alignment_episodes(common_context: PreparedRLContext) -> list[RLEpisode]:
    validation_episodes = [
        episode
        for episode in common_context.all_episodes
        if getattr(episode, "split", "") == "validation"
    ]
    if not validation_episodes:
        raise ValueError("Validation episodes are required for the RL alignment gate.")
    return validation_episodes


def _prompt_baseline_evaluation(
    *,
    prompt_row: dict[str, Any],
    policy_model_name: str,
    base_model_name: str | None,
    candidate_config: CandidateGenerationConfig,
    realization_config: RealizationConfig,
    reward_config: RewardConfig,
    similarity_config_path: str,
    runtime_config_path: str | None,
) -> Any:
    single_config = _single_idea_candidate_config(candidate_config)
    prompt_mode = str(
        prompt_row.get("prompt_mode", "z_conditioned_realization")
        or "z_conditioned_realization"
    )
    train_papers = [
        PaperRecord(**paper) for paper in prompt_row.get("train_papers", [])
    ]
    future_papers = [
        PaperRecord(**paper) for paper in prompt_row.get("future_papers", [])
    ]
    innovation = Innovation(**dict(prompt_row.get("innovation", {})))
    evidence_papers = [
        PaperRecord(**paper) for paper in prompt_row.get("evidence_papers", [])
    ]
    if prompt_mode == "strict_interactive_realization":
        completion_text, _ = _generate_strict_realization_completion(
            prompt_row,
            policy_model_name,
            _midpoint_temperature(single_config),
            single_config,
            realization_config=realization_config,
            top_p=single_config.top_p,
            seed=single_config.seed,
            base_model_name=base_model_name,
        )
        return evaluate_strict_rl_reward(
            completion_text,
            innovation=innovation,
            train_papers=train_papers,
            future_papers=future_papers,
            reward_config=reward_config,
            realization_config=realization_config,
            search_env_payload=prompt_row.get("search_env")
            if isinstance(prompt_row.get("search_env"), dict)
            else None,
            similarity_config_path=similarity_config_path,
            runtime_config_path=runtime_config_path,
            model_name=policy_model_name,
            cutoff_date=str(prompt_row.get("cutoff_date", "") or "") or None,
            future_end_date=str(prompt_row.get("future_end_date", "") or "") or None,
        )
    predictions = _generate_realization_candidate_predictions(
        prompt_row,
        policy_model_name,
        _midpoint_temperature(single_config),
        single_config,
        realization_config=realization_config,
        top_p=single_config.top_p,
        seed=single_config.seed,
        base_model_name=base_model_name,
        fallback_to_heuristic=False,
    )
    if len(predictions) != 1:
        return build_invalid_reward_evaluation(reward_config)
    return evaluate_rl_reward(
        predictions=predictions,
        train_papers=train_papers,
        future_papers=future_papers,
        reward_config=reward_config,
        innovation=innovation,
        evidence_papers=evidence_papers,
        proposal_text=str(predictions[0].metadata.get("proposal_text", "") or ""),
        realization_config=realization_config,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
        model_name=policy_model_name,
        cutoff_date=str(prompt_row.get("cutoff_date", "") or "") or None,
        future_end_date=str(prompt_row.get("future_end_date", "") or "") or None,
    )


def run_online_alignment_gate(
    common_context: PreparedRLContext,
    *,
    model_name: str,
    init_policy_path: str | None,
    candidate_config: CandidateGenerationConfig,
    realization_config: RealizationConfig,
    reward_config: RewardConfig,
    trainer_config: Any,
    trainer_output_dir: Path,
    strict_mode: bool = False,
) -> dict[str, Any]:
    episodes = _alignment_episodes(common_context)
    policy_model_name, base_model_name = _resolve_policy_source(
        model_name=model_name,
        init_policy_path=init_policy_path,
    )
    validation_prompt_rows = (
        build_strict_rl_prompt_rows(
            common_context.papers,
            episodes,
            candidate_config=candidate_config,
            realization_config=realization_config,
            hindsight_samples=common_context.hindsight_samples,
        )
        if strict_mode
        else build_grpo_prompt_rows(
            common_context.papers,
            episodes,
            candidate_config=candidate_config,
            realization_config=realization_config,
            hindsight_samples=common_context.hindsight_samples,
        )
    )
    candidate_lists = generate_episode_candidate_lists(
        common_context.papers,
        episodes,
        model_name=policy_model_name,
        candidate_config=candidate_config,
        reward_config=reward_config,
        realization_config=realization_config,
        prompt_rows=validation_prompt_rows,
        similarity_config_path=common_context.similarity_config_path,
        runtime_config_path=common_context.runtime_config_path,
        base_model_name=base_model_name,
        fallback_to_heuristic=False,
    )
    evaluations = [
        candidate.reward for batch in candidate_lists for candidate in batch.candidates
    ]
    report = compute_reward_alignment(evaluations, trainer_config)
    reward_selected = [
        max(batch.candidates, key=lambda candidate: candidate.reward.list_reward).reward
        for batch in candidate_lists
        if batch.candidates
    ]
    prompt_baseline = []
    for row in validation_prompt_rows:
        prompt_baseline.append(
            _prompt_baseline_evaluation(
                prompt_row=row,
                policy_model_name=policy_model_name,
                base_model_name=base_model_name,
                candidate_config=candidate_config,
                realization_config=realization_config,
                reward_config=reward_config,
                similarity_config_path=common_context.similarity_config_path,
                runtime_config_path=common_context.runtime_config_path,
            )
        )
    invalid_count = sum(
        1 for evaluation in evaluations if evaluation.invalid_completion
    )
    parse_failure_count = sum(
        1
        for evaluation in evaluations
        if float(evaluation.reward_breakdown.get("parse_failure", 0.0)) > 0.0
    )
    zero_or_invalid_fraction = (
        round(
            sum(
                1
                for evaluation in evaluations
                if evaluation.invalid_completion or evaluation.list_reward <= 0.0
            )
            / len(evaluations),
            4,
        )
        if evaluations
        else 0.0
    )
    reward_selected_hit = _average_metric(
        [evaluation.benchmark_evaluation for evaluation in reward_selected],
        "hit_at_k",
    )
    reward_selected_mrr = _average_metric(
        [evaluation.benchmark_evaluation for evaluation in reward_selected],
        "mrr",
    )
    prompt_baseline_hit = _average_metric(
        [evaluation.benchmark_evaluation for evaluation in prompt_baseline],
        "hit_at_k",
    )
    prompt_baseline_mrr = _average_metric(
        [evaluation.benchmark_evaluation for evaluation in prompt_baseline],
        "mrr",
    )
    baseline_passed = (
        reward_selected_hit >= prompt_baseline_hit
        and reward_selected_mrr >= prompt_baseline_mrr
    )
    diagnostics = {
        "alignment_rho": report.correlation,
        "alignment_passed": bool(report.passed and baseline_passed),
        "episodes_used": len(episodes),
        "reward_selected_avg_hit_at_1": reward_selected_hit,
        "reward_selected_avg_mrr": reward_selected_mrr,
        "prompt_baseline_avg_hit_at_1": prompt_baseline_hit,
        "prompt_baseline_avg_mrr": prompt_baseline_mrr,
        "parse_failure_rate": round(parse_failure_count / len(evaluations), 4)
        if evaluations
        else 0.0,
        "invalid_completion_rate": round(invalid_count / len(evaluations), 4)
        if evaluations
        else 0.0,
        "zero_or_invalid_reward_fraction": zero_or_invalid_fraction,
        "sample_count": len(evaluations),
        "alignment_threshold": float(
            getattr(trainer_config, "reward_alignment_threshold", 0.0)
        ),
    }
    _write_json(trainer_output_dir / "alignment_report.json", diagnostics)
    _write_json(trainer_output_dir / "alignment_summary.json", diagnostics)
    return diagnostics
