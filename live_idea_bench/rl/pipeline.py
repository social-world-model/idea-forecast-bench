from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from live_idea_bench.models import PaperRecord
from live_idea_bench.predictor import _heuristic_predictions, generate_predictions
from live_idea_bench.rl.config import CandidateGenerationConfig, EpisodeBuildConfig, RewardConfig, SelectionConfig
from live_idea_bench.rl.dpo import CandidateListSample, EpisodeCandidateLists
from live_idea_bench.rl.episodes import RLEpisode, build_rl_episodes, serialize_episodes
from live_idea_bench.rl.grpo import compute_reward_alignment
from live_idea_bench.rl.io import _read_json, _read_jsonl, _write_json, _write_jsonl
from live_idea_bench.rl.local_generation import build_prediction_prompt, generate_local_predictions
from live_idea_bench.rl.model_zoo import list_small_model_payloads
from live_idea_bench.rl.reward import build_invalid_reward_evaluation, evaluate_rl_reward, serialize_reward_evaluation
from live_idea_bench.rl.trainers import PreparedRLContext, TrainerPreparedArtifacts, create_trainer_runner
from live_idea_bench.rl.trainers.base import build_config_fingerprint


def _paper_lookup(papers: list[PaperRecord]) -> dict[str, PaperRecord]:
    return {paper.paper_id: paper for paper in papers}


def _materialize_episode(episode: RLEpisode, paper_lookup: dict[str, PaperRecord]) -> tuple[list[PaperRecord], list[PaperRecord]]:
    train = [paper_lookup[paper_id] for paper_id in episode.train_paper_ids if paper_id in paper_lookup]
    future = [paper_lookup[paper_id] for paper_id in episode.future_paper_ids if paper_id in paper_lookup]
    return train, future


def _serialize_papers(papers: list[PaperRecord]) -> list[dict[str, Any]]:
    return [asdict(paper) for paper in papers]


def _deserialize_episodes(path: Path) -> list[RLEpisode]:
    payload = _read_json(path)
    rows = payload.get("episodes", []) if isinstance(payload, dict) else []
    return [RLEpisode(**row) for row in rows if isinstance(row, dict)]


def _temperature_schedule(config: CandidateGenerationConfig) -> list[float]:
    if config.num_candidate_lists <= 0:
        return []
    if config.num_candidate_lists == 1:
        return [round((config.min_temperature + config.max_temperature) / 2.0, 4)]
    step = (config.max_temperature - config.min_temperature) / (config.num_candidate_lists - 1)
    return [round(config.min_temperature + (idx * step), 4) for idx in range(config.num_candidate_lists)]


_ALLOWED_BACKENDS = frozenset({"auto", "api", "local_hf", "heuristic"})
_API_MODEL_PREFIXES = ("gpt-4o", "gpt-4", "gpt-3", "claude-", "o1", "o3")
_API_MODEL_SUBSTRINGS = ("gemini",)


def _resolve_generation_backend(model_name: str, backend: str) -> str:
    normalized = backend.strip().lower()
    if normalized not in _ALLOWED_BACKENDS:
        raise ValueError(
            f"Unsupported generation backend {backend!r}. "
            f"Allowed values: {sorted(_ALLOWED_BACKENDS)}"
        )
    if normalized != "auto":
        return normalized
    if any(model_name.startswith(prefix) for prefix in _API_MODEL_PREFIXES):
        return "api"
    if any(substr in model_name for substr in _API_MODEL_SUBSTRINGS):
        return "api"
    return "local_hf"


def _generate_candidate_predictions(
    train_papers: list[PaperRecord],
    cutoff_month: str,
    model_name: str,
    temperature: float,
    config: CandidateGenerationConfig,
    *,
    top_p: float | None = None,
    seed: int | None = None,
    context_shuffle_seed: int | None = None,
    base_model_name: str | None = None,
    fallback_to_heuristic: bool = False,
) -> list[Any]:
    backend = _resolve_generation_backend(model_name, config.backend)
    prompt_train_papers = list(train_papers)
    if context_shuffle_seed is not None:
        import random

        rng = random.Random(context_shuffle_seed)
        rng.shuffle(prompt_train_papers)
    if backend == "heuristic":
        return _heuristic_predictions(prompt_train_papers, cutoff_month, config.ideas_per_list)
    if backend == "api":
        return generate_predictions(
            train_papers=prompt_train_papers,
            cutoff_month=cutoff_month,
            top_k=config.ideas_per_list,
            model_name=model_name,
            predictor_config_path=config.predictor_config,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            fallback_to_heuristic=fallback_to_heuristic,
        )
    if backend == "local_hf":
        return generate_local_predictions(
            train_papers=prompt_train_papers,
            cutoff_month=cutoff_month,
            top_k=config.ideas_per_list,
            model_name_or_path=model_name,
            predictor_config_path=config.predictor_config,
            temperature=temperature,
            top_p=top_p if top_p is not None else config.top_p,
            sampling_top_k=config.top_k,
            max_new_tokens=config.max_new_tokens,
            repetition_penalty=config.repetition_penalty,
            enable_thinking=config.enable_thinking,
            seed=seed if seed is not None else config.seed + int(temperature * 1000),
            base_model_name=base_model_name,
            fallback_to_heuristic=fallback_to_heuristic,
        )
    raise ValueError(f"Unsupported candidate generation backend: {backend}")


def _single_idea_candidate_config(config: CandidateGenerationConfig) -> CandidateGenerationConfig:
    return replace(config, ideas_per_list=1)


def generate_episode_candidate_lists(
    papers: list[PaperRecord],
    episodes: list[RLEpisode],
    *,
    model_name: str,
    candidate_config: CandidateGenerationConfig,
    reward_config: RewardConfig,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    base_model_name: str | None = None,
    fallback_to_heuristic: bool = False,
) -> list[EpisodeCandidateLists]:
    paper_lookup = _paper_lookup(papers)
    single_config = _single_idea_candidate_config(candidate_config)
    temperatures = _temperature_schedule(single_config)
    outputs: list[EpisodeCandidateLists] = []
    for episode in episodes:
        train_papers, future_papers = _materialize_episode(episode, paper_lookup)
        prompt = build_prediction_prompt(
            train_papers,
            episode.cutoff_month,
            1,
            predictor_config_path=single_config.predictor_config,
        )
        candidates: list[CandidateListSample] = []
        for temperature in temperatures:
            predictions = _generate_candidate_predictions(
                train_papers,
                episode.cutoff_month,
                model_name,
                temperature,
                single_config,
                top_p=single_config.top_p,
                seed=single_config.seed + int(temperature * 1000),
                base_model_name=base_model_name,
                fallback_to_heuristic=fallback_to_heuristic,
            )
            backend = _resolve_generation_backend(model_name, single_config.backend)
            predictions = [
                replace(
                    prediction,
                    rank=idx,
                    metadata={
                        "sampling_temperature": temperature,
                        "sampling_top_p": single_config.top_p,
                        "generation_backend": backend,
                        **prediction.metadata,
                    },
                )
                for idx, prediction in enumerate(predictions, start=1)
            ]
            if len(predictions) != 1:
                reward = build_invalid_reward_evaluation(reward_config)
            else:
                reward = evaluate_rl_reward(
                    predictions=predictions,
                    train_papers=train_papers,
                    future_papers=future_papers,
                    reward_config=reward_config,
                    similarity_config_path=similarity_config_path,
                    runtime_config_path=runtime_config_path,
                    model_name=model_name,
                    cutoff_date=episode.cutoff_date,
                    future_end_date=episode.future_end_date,
                )
            candidates.append(CandidateListSample(predictions=predictions, reward=reward))
        episode_candidates = EpisodeCandidateLists(episode=episode, prompt=prompt, candidates=candidates)
        outputs.append(episode_candidates)
        if episode.cache_artifact_path:
            _write_json(
                Path(episode.cache_artifact_path),
                {
                    "episode": asdict(episode),
                    "prompt": prompt,
                    "candidates": [
                        {
                            "predictions": [asdict(prediction) for prediction in candidate.predictions],
                            "reward": serialize_reward_evaluation(candidate.reward),
                        }
                        for candidate in candidates
                    ],
                },
            )
    return outputs


def serialize_episode_candidate_lists(episodes: list[EpisodeCandidateLists]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_batch in episodes:
        rows.append(
            {
                "episode": asdict(episode_batch.episode),
                "prompt": episode_batch.prompt,
                "candidates": [
                    {
                        "predictions": [asdict(prediction) for prediction in candidate.predictions],
                        "reward": serialize_reward_evaluation(candidate.reward),
                    }
                    for candidate in episode_batch.candidates
                ],
            }
        )
    return rows


def build_grpo_prompt_rows(
    papers: list[PaperRecord],
    episodes: list[RLEpisode],
    *,
    candidate_config: CandidateGenerationConfig,
) -> list[dict[str, Any]]:
    paper_lookup = _paper_lookup(papers)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        train_papers, future_papers = _materialize_episode(episode, paper_lookup)
        rows.append(
            {
                "prompt": build_prediction_prompt(
                    train_papers,
                    episode.cutoff_month,
                    1,
                    predictor_config_path=candidate_config.predictor_config,
                ),
                "cutoff_month": episode.cutoff_month,
                "cutoff_date": episode.cutoff_date,
                "future_end_month": episode.future_end_month,
                "future_end_date": episode.future_end_date,
                "train_papers": _serialize_papers(train_papers),
                "future_papers": _serialize_papers(future_papers),
            }
        )
    return rows


def _select_episodes(episodes: list[RLEpisode], split: str, max_episodes: int | None) -> list[RLEpisode]:
    normalized_split = split.strip().lower()
    if normalized_split == "all":
        selected = list(episodes)
    else:
        selected = [episode for episode in episodes if episode.split == normalized_split]
    if max_episodes is not None:
        return selected[: max(0, max_episodes)]
    return selected


def _require_train_split_for_training(split: str, prepare_only: bool) -> None:
    normalized_split = split.strip().lower()
    if prepare_only:
        return
    if normalized_split != "train":
        raise ValueError(
            "RL training is restricted to --split train. "
            "Use --prepare-only if you need to inspect validation/test/all artifacts."
        )


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
    return round(sum(float(getattr(evaluation, name)) for evaluation in evaluations) / len(evaluations), 4)


def _shared_fingerprint(
    *,
    model_name: str,
    split: str,
    max_episodes: int | None,
    episode_config: EpisodeBuildConfig,
    candidate_config: CandidateGenerationConfig,
    reward_config: RewardConfig,
    selection_config: SelectionConfig,
    similarity_config_path: str,
) -> str:
    return build_config_fingerprint(
        {
            "model_name": model_name,
            "split": split,
            "max_episodes": max_episodes,
            "episode_config": episode_config,
            "candidate_config": candidate_config,
            "reward_config": reward_config,
            "selection_config": selection_config,
            "similarity_config_path": similarity_config_path,
        }
    )


def _load_cached_context(
    *,
    papers: list[PaperRecord],
    target_dir: Path,
    fingerprint: str,
    split: str,
    max_episodes: int | None,
    model_name: str,
    similarity_config_path: str,
    runtime_config_path: str | None,
) -> PreparedRLContext | None:
    shared_dir = target_dir / "shared"
    manifest_path = shared_dir / "shared_manifest.json"
    episodes_path = shared_dir / "episodes.json"
    prompt_rows_path = shared_dir / "prompts.jsonl"
    if not manifest_path.exists() or not episodes_path.exists() or not prompt_rows_path.exists():
        return None
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict):
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    all_episodes = _deserialize_episodes(episodes_path)
    selected_episodes = _select_episodes(all_episodes, split, max_episodes)
    return PreparedRLContext(
        papers=papers,
        all_episodes=all_episodes,
        selected_episodes=selected_episodes,
        prompt_rows=_read_jsonl(prompt_rows_path),
        shared_dir=shared_dir,
        episodes_path=episodes_path,
        prompt_rows_path=prompt_rows_path,
        paper_lookup=_paper_lookup(papers),
        config_fingerprint=fingerprint,
        selected_split=split,
        model_name=model_name,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
        episode_cache_root=target_dir / "episode_cache",
        shared_manifest_path=manifest_path,
    )


def prepare_common_rl_context(
    papers: list[PaperRecord],
    *,
    model_name: str,
    output_dir: str,
    episode_config: EpisodeBuildConfig,
    candidate_config: CandidateGenerationConfig,
    reward_config: RewardConfig,
    selection_config: SelectionConfig,
    split: str = "train",
    max_episodes: int | None = None,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
) -> PreparedRLContext:
    target_dir = Path(output_dir).resolve()
    shared_dir = target_dir / "shared"
    fingerprint = _shared_fingerprint(
        model_name=model_name,
        split=split,
        max_episodes=max_episodes,
        episode_config=episode_config,
        candidate_config=candidate_config,
        reward_config=reward_config,
        selection_config=selection_config,
        similarity_config_path=similarity_config_path,
    )
    cached = _load_cached_context(
        papers=papers,
        target_dir=target_dir,
        fingerprint=fingerprint,
        split=split,
        max_episodes=max_episodes,
        model_name=model_name,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
    )
    if cached is not None:
        return cached

    episode_cache_root = target_dir / "episode_cache"
    episodes = build_rl_episodes(papers, episode_config, cache_root=episode_cache_root)
    selected_episodes = _select_episodes(episodes, split, max_episodes)
    if not selected_episodes:
        raise ValueError("No RL episodes were selected. Adjust split, window, or date settings.")
    prompt_rows = build_grpo_prompt_rows(
        papers,
        selected_episodes,
        candidate_config=candidate_config,
    )
    episodes_path = shared_dir / "episodes.json"
    prompt_rows_path = shared_dir / "prompts.jsonl"
    shared_manifest_path = shared_dir / "shared_manifest.json"
    _write_json(episodes_path, {"episodes": serialize_episodes(episodes)})
    _write_jsonl(prompt_rows_path, prompt_rows)
    _write_json(
        shared_manifest_path,
        {
            "split": split,
            "training_split_policy": "train_only",
            "selected_episode_count": len(selected_episodes),
            "episode_config": asdict(episode_config),
            "candidate_config": asdict(candidate_config),
            "reward_config": asdict(reward_config),
            "selection_config": asdict(selection_config),
            "similarity_config_path": similarity_config_path,
            "model_name": model_name,
            "fingerprint": fingerprint,
        },
    )
    return PreparedRLContext(
        papers=papers,
        all_episodes=episodes,
        selected_episodes=selected_episodes,
        prompt_rows=prompt_rows,
        shared_dir=shared_dir,
        episodes_path=episodes_path,
        prompt_rows_path=prompt_rows_path,
        paper_lookup=_paper_lookup(papers),
        config_fingerprint=fingerprint,
        selected_split=split,
        model_name=model_name,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
        episode_cache_root=episode_cache_root,
        shared_manifest_path=shared_manifest_path,
    )


def _alignment_episodes(common_context: PreparedRLContext) -> list[RLEpisode]:
    validation_episodes = [episode for episode in common_context.all_episodes if getattr(episode, "split", "") == "validation"]
    if not validation_episodes:
        raise ValueError("Validation episodes are required for the RL alignment gate.")
    return validation_episodes


def _prompt_baseline_evaluation(
    *,
    train_papers: list[PaperRecord],
    future_papers: list[PaperRecord],
    cutoff_month: str,
    cutoff_date: str,
    future_end_date: str,
    policy_model_name: str,
    base_model_name: str | None,
    candidate_config: CandidateGenerationConfig,
    reward_config: RewardConfig,
    similarity_config_path: str,
    runtime_config_path: str | None,
) -> Any:
    single_config = _single_idea_candidate_config(candidate_config)
    predictions = _generate_candidate_predictions(
        train_papers,
        cutoff_month,
        policy_model_name,
        _midpoint_temperature(single_config),
        single_config,
        top_p=single_config.top_p,
        seed=single_config.seed,
        context_shuffle_seed=None,
        base_model_name=base_model_name,
        fallback_to_heuristic=False,
    )
    if len(predictions) != 1:
        return build_invalid_reward_evaluation(reward_config)
    prediction = replace(
        predictions[0],
        rank=1,
        metadata={
            "sampling_temperature": _midpoint_temperature(single_config),
            "sampling_top_p": single_config.top_p,
            "generation_backend": _resolve_generation_backend(policy_model_name, single_config.backend),
            **predictions[0].metadata,
        },
    )
    return evaluate_rl_reward(
        predictions=[prediction],
        train_papers=train_papers,
        future_papers=future_papers,
        reward_config=reward_config,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
        model_name=policy_model_name,
        cutoff_date=cutoff_date,
        future_end_date=future_end_date,
    )


def run_online_alignment_gate(
    common_context: PreparedRLContext,
    *,
    model_name: str,
    init_policy_path: str | None,
    candidate_config: CandidateGenerationConfig,
    reward_config: RewardConfig,
    trainer_config: Any,
    trainer_output_dir: Path,
) -> dict[str, Any]:
    episodes = _alignment_episodes(common_context)
    policy_model_name, base_model_name = _resolve_policy_source(
        model_name=model_name,
        init_policy_path=init_policy_path,
    )
    candidate_lists = generate_episode_candidate_lists(
        common_context.papers,
        episodes,
        model_name=policy_model_name,
        candidate_config=candidate_config,
        reward_config=reward_config,
        similarity_config_path=common_context.similarity_config_path,
        runtime_config_path=common_context.runtime_config_path,
        base_model_name=base_model_name,
        fallback_to_heuristic=False,
    )
    evaluations = [candidate.reward for batch in candidate_lists for candidate in batch.candidates]
    report = compute_reward_alignment(evaluations, trainer_config)
    reward_selected = [
        max(batch.candidates, key=lambda candidate: candidate.reward.list_reward).reward
        for batch in candidate_lists
        if batch.candidates
    ]
    prompt_baseline = []
    for episode in episodes:
        train_papers, future_papers = _materialize_episode(episode, common_context.paper_lookup)
        prompt_baseline.append(
            _prompt_baseline_evaluation(
                train_papers=train_papers,
                future_papers=future_papers,
                cutoff_month=episode.cutoff_month,
                cutoff_date=episode.cutoff_date,
                future_end_date=episode.future_end_date,
                policy_model_name=policy_model_name,
                base_model_name=base_model_name,
                candidate_config=candidate_config,
                reward_config=reward_config,
                similarity_config_path=common_context.similarity_config_path,
                runtime_config_path=common_context.runtime_config_path,
            )
        )
    invalid_count = sum(1 for evaluation in evaluations if evaluation.invalid_completion)
    parse_failure_count = sum(
        1
        for evaluation in evaluations
        if float(evaluation.reward_breakdown.get("parse_failure", 0.0)) > 0.0
    )
    zero_or_invalid_fraction = (
        round(
            sum(1 for evaluation in evaluations if evaluation.invalid_completion or evaluation.list_reward <= 0.0)
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
        "parse_failure_rate": round(parse_failure_count / len(evaluations), 4) if evaluations else 0.0,
        "invalid_completion_rate": round(invalid_count / len(evaluations), 4) if evaluations else 0.0,
        "zero_or_invalid_reward_fraction": zero_or_invalid_fraction,
        "sample_count": len(evaluations),
        "alignment_threshold": float(getattr(trainer_config, "reward_alignment_threshold", 0.0)),
    }
    _write_json(trainer_output_dir / "alignment_report.json", diagnostics)
    _write_json(trainer_output_dir / "alignment_summary.json", diagnostics)
    return diagnostics


def run_policy_rl_pipeline(
    papers: list[PaperRecord],
    *,
    trainer: str,
    model_name: str,
    output_dir: str,
    episode_config: EpisodeBuildConfig,
    candidate_config: CandidateGenerationConfig,
    reward_config: RewardConfig,
    selection_config: SelectionConfig,
    trainer_config: Any,
    trainer_config_path: str,
    selection_config_path: str,
    split: str = "train",
    max_episodes: int | None = None,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    prepare_only: bool = False,
    init_policy_path: str | None = None,
    skip_alignment_check: bool = False,
) -> dict[str, Any]:
    _require_train_split_for_training(split, prepare_only)
    runner = create_trainer_runner(trainer)
    target_dir = Path(output_dir).resolve()
    common_context = prepare_common_rl_context(
        papers,
        model_name=model_name,
        output_dir=output_dir,
        episode_config=episode_config,
        candidate_config=candidate_config,
        reward_config=reward_config,
        selection_config=selection_config,
        split=split,
        max_episodes=max_episodes,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
    )
    prepared = runner.prepare(
        common_context,
        model_name=model_name,
        candidate_config=candidate_config,
        reward_config=reward_config,
        trainer_config=trainer_config,
    )

    diagnostics: dict[str, Any] = {}
    if runner.trainer_name in {"grpo", "rloo"} and not skip_alignment_check:
        diagnostics = run_online_alignment_gate(
            common_context,
            model_name=model_name,
            init_policy_path=init_policy_path,
            candidate_config=candidate_config,
            reward_config=reward_config,
            trainer_config=trainer_config,
            trainer_output_dir=prepared.output_dir,
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
        "model_name": model_name,
        "split": split,
        "training_split_policy": "train_only",
        "selected_episode_count": len(common_context.selected_episodes),
        "shared_manifest_path": str(common_context.shared_manifest_path),
        "episodes_path": str(common_context.episodes_path),
        "prompt_rows_path": str(common_context.prompt_rows_path),
        "trainer_dataset_path": str(prepared.dataset_path.resolve()),
        "trainer_output_dir": str(prepared.output_dir.resolve()),
        "trainer_policy_manifest_path": str((prepared.output_dir / "policy_manifest.json").resolve()) if trainer_manifest else "",
        "prepare_only": prepare_only,
        "selection_config_path": selection_config_path,
        "recommended_small_models": list_small_model_payloads(),
        "shared_fingerprint": common_context.config_fingerprint,
        "trainer_metadata": prepared.metadata,
        "diagnostics": diagnostics,
    }
    _write_json(target_dir / "pipeline_manifest.json", manifest)
    return manifest
