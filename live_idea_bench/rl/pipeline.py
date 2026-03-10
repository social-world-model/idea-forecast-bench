from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from live_idea_bench.models import PaperRecord
from live_idea_bench.predictor import _heuristic_predictions, generate_predictions
from live_idea_bench.rl.config import (
    CandidateGenerationConfig,
    DPOTrainConfig,
    EpisodeBuildConfig,
    GRPOTrainConfig,
    RewardConfig,
)
from live_idea_bench.rl.dpo import CandidateListSample, EpisodeCandidateLists, build_dpo_pairs
from live_idea_bench.rl.episodes import RLEpisode, build_rl_episodes, serialize_episodes
from live_idea_bench.rl.local_generation import build_prediction_prompt, generate_local_predictions
from live_idea_bench.rl.model_zoo import list_small_model_payloads
from live_idea_bench.rl.reward import evaluate_rl_reward, serialize_reward_evaluation
from live_idea_bench.rl.io import _write_json, _write_jsonl
from live_idea_bench.rl.trainers import train_dpo_with_trl, train_grpo_with_trl


def _paper_lookup(papers: list[PaperRecord]) -> dict[str, PaperRecord]:
    return {paper.paper_id: paper for paper in papers}


def _materialize_episode(episode: RLEpisode, paper_lookup: dict[str, PaperRecord]) -> tuple[list[PaperRecord], list[PaperRecord]]:
    train = [paper_lookup[paper_id] for paper_id in episode.train_paper_ids if paper_id in paper_lookup]
    future = [paper_lookup[paper_id] for paper_id in episode.future_paper_ids if paper_id in paper_lookup]
    return train, future


def _serialize_papers(papers: list[PaperRecord]) -> list[dict[str, Any]]:
    return [asdict(paper) for paper in papers]


def _temperature_schedule(config: CandidateGenerationConfig) -> list[float]:
    if config.num_candidate_lists <= 0:
        return []
    if config.num_candidate_lists == 1:
        return [round((config.min_temperature + config.max_temperature) / 2.0, 4)]
    step = (config.max_temperature - config.min_temperature) / (config.num_candidate_lists - 1)
    return [round(config.min_temperature + (idx * step), 4) for idx in range(config.num_candidate_lists)]


def _resolve_generation_backend(model_name: str, backend: str) -> str:
    normalized = backend.strip().lower()
    if normalized != "auto":
        return normalized
    if model_name.startswith("gpt-4o") or model_name.startswith("gpt-5") or model_name.startswith("claude-"):
        return "api"
    if "gemini" in model_name:
        return "api"
    return "local_hf"


def _generate_candidate_predictions(
    train_papers: list[PaperRecord],
    cutoff_month: str,
    model_name: str,
    temperature: float,
    config: CandidateGenerationConfig,
) -> list[Any]:
    backend = _resolve_generation_backend(model_name, config.backend)
    if backend == "heuristic":
        return _heuristic_predictions(train_papers, cutoff_month, config.ideas_per_list)
    if backend == "api":
        return generate_predictions(
            train_papers=train_papers,
            cutoff_month=cutoff_month,
            top_k=config.ideas_per_list,
            model_name=model_name,
            predictor_config_path=config.predictor_config,
            temperature=temperature,
        )
    if backend == "local_hf":
        return generate_local_predictions(
            train_papers=train_papers,
            cutoff_month=cutoff_month,
            top_k=config.ideas_per_list,
            model_name_or_path=model_name,
            predictor_config_path=config.predictor_config,
            temperature=temperature,
            top_p=config.top_p,
            sampling_top_k=config.top_k,
            max_new_tokens=config.max_new_tokens,
            repetition_penalty=config.repetition_penalty,
            enable_thinking=config.enable_thinking,
            seed=config.seed + int(temperature * 1000),
        )
    raise ValueError(f"Unsupported candidate generation backend: {backend}")


def generate_episode_candidate_lists(
    papers: list[PaperRecord],
    episodes: list[RLEpisode],
    *,
    model_name: str,
    candidate_config: CandidateGenerationConfig,
    reward_config: RewardConfig,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
) -> list[EpisodeCandidateLists]:
    paper_lookup = _paper_lookup(papers)
    temperatures = _temperature_schedule(candidate_config)
    outputs: list[EpisodeCandidateLists] = []
    for episode in episodes:
        train_papers, future_papers = _materialize_episode(episode, paper_lookup)
        prompt = build_prediction_prompt(
            train_papers,
            episode.cutoff_month,
            candidate_config.ideas_per_list,
            predictor_config_path=candidate_config.predictor_config,
        )
        candidates: list[CandidateListSample] = []
        for temperature in temperatures:
            predictions = _generate_candidate_predictions(
                train_papers,
                episode.cutoff_month,
                model_name,
                temperature,
                candidate_config,
            )
            for idx, prediction in enumerate(predictions, start=1):
                prediction.rank = idx
                prediction.metadata.setdefault("sampling_temperature", temperature)
                prediction.metadata.setdefault("generation_backend", _resolve_generation_backend(model_name, candidate_config.backend))
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
                    candidate_config.ideas_per_list,
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


def run_policy_rl_pipeline(
    papers: list[PaperRecord],
    *,
    model_name: str,
    output_dir: str,
    episode_config: EpisodeBuildConfig,
    candidate_config: CandidateGenerationConfig,
    reward_config: RewardConfig,
    dpo_config: DPOTrainConfig,
    grpo_config: GRPOTrainConfig,
    stage: str = "prepare",
    split: str = "train",
    max_episodes: int | None = None,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
) -> dict[str, Any]:
    normalized_stage = stage.strip().lower()
    if normalized_stage not in {"prepare", "dpo", "grpo", "both"}:
        raise ValueError(f"Unsupported RL pipeline stage: {stage}")

    target_dir = Path(output_dir).resolve()
    episode_cache_root = target_dir / "episode_cache"
    episodes = build_rl_episodes(papers, episode_config, cache_root=episode_cache_root)
    selected_episodes = _select_episodes(episodes, split, max_episodes)
    if not selected_episodes:
        raise ValueError("No RL episodes were selected. Adjust split, window, or date settings.")

    episodes_path = target_dir / "episodes.json"
    _write_json(episodes_path, {"episodes": serialize_episodes(selected_episodes)})

    prompt_rows = build_grpo_prompt_rows(
        papers,
        selected_episodes,
        candidate_config=candidate_config,
    )
    grpo_prompt_path = target_dir / "grpo_prompts.jsonl"
    _write_jsonl(grpo_prompt_path, prompt_rows)

    dpo_pairs: list[dict[str, Any]] = []
    rollout_path = target_dir / "candidate_rollouts.json"
    dpo_pairs_path = target_dir / "dpo_pairs.jsonl"
    if normalized_stage in {"prepare", "dpo", "both"}:
        candidate_lists = generate_episode_candidate_lists(
            papers,
            selected_episodes,
            model_name=model_name,
            candidate_config=candidate_config,
            reward_config=reward_config,
            similarity_config_path=similarity_config_path,
            runtime_config_path=runtime_config_path,
        )
        dpo_pairs = build_dpo_pairs(candidate_lists, dpo_config)
        _write_json(rollout_path, {"episodes": serialize_episode_candidate_lists(candidate_lists)})
        _write_jsonl(dpo_pairs_path, dpo_pairs)

    dpo_manifest: dict[str, Any] | None = None
    if normalized_stage in {"dpo", "both"}:
        dpo_manifest = train_dpo_with_trl(
            dpo_pairs,
            dpo_config,
            model_name=model_name,
            predictor_config=candidate_config.predictor_config,
            output_dir=str(target_dir / "dpo"),
        )

    grpo_manifest: dict[str, Any] | None = None
    if normalized_stage in {"grpo", "both"}:
        grpo_manifest = train_grpo_with_trl(
            prompt_rows,
            grpo_config,
            model_name=model_name,
            predictor_config=candidate_config.predictor_config,
            output_dir=str(target_dir / "grpo"),
            reward_config=reward_config,
            similarity_config_path=similarity_config_path,
            runtime_config_path=runtime_config_path,
        )

    manifest = {
        "pipeline_manifest_version": 1,
        "stage": normalized_stage,
        "model_name": model_name,
        "split": split,
        "selected_episode_count": len(selected_episodes),
        "episodes_path": str(episodes_path),
        "grpo_prompt_path": str(grpo_prompt_path),
        "candidate_rollout_path": str(rollout_path) if rollout_path.exists() else "",
        "dpo_pairs_path": str(dpo_pairs_path) if dpo_pairs_path.exists() else "",
        "dpo_pair_count": len(dpo_pairs),
        "grpo_prompt_count": len(prompt_rows),
        "dpo_policy_manifest_path": str((target_dir / "dpo" / "policy_manifest.json")) if dpo_manifest else "",
        "grpo_policy_manifest_path": str((target_dir / "grpo" / "policy_manifest.json")) if grpo_manifest else "",
        "recommended_small_models": list_small_model_payloads(),
    }
    _write_json(target_dir / "pipeline_manifest.json", manifest)
    return manifest
