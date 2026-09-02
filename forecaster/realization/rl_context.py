from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from forecaster.config import RealizationConfig, load_realization_config
from forecaster.models import (
    HindsightSample,
    innovation_to_dict,
    strict_runtime_manifest_contract,
)
from forecaster.realization.config import (
    CandidateGenerationConfig,
    EpisodeBuildConfig,
    RewardConfig,
    SelectionConfig,
)
from forecaster.realization.episode_prompts import (
    _paper_lookup,
    build_grpo_prompt_rows,
    build_strict_rl_prompt_rows,
)
from forecaster.realization.episodes import (
    RLEpisode,
    build_rl_episodes,
    serialize_episodes,
)
from forecaster.realization.io import _read_json, _read_jsonl, _write_json, _write_jsonl
from forecaster.realization.trainers import (
    PreparedRLContext,
    build_config_fingerprint,
)
from idea_forecast_bench.models import PaperRecord

logger = logging.getLogger(__name__)


def _deserialize_episodes(path: Path) -> list[RLEpisode]:
    payload = _read_json(path)
    rows = payload.get("episodes", []) if isinstance(payload, dict) else []
    return [RLEpisode(**row) for row in rows if isinstance(row, dict)]


def _select_episodes(
    episodes: list[RLEpisode], split: str, max_episodes: int | None
) -> list[RLEpisode]:
    normalized_split = split.strip().lower()
    if normalized_split == "all":
        selected = list(episodes)
    else:
        selected = [
            episode for episode in episodes if episode.split == normalized_split
        ]
    if max_episodes is not None:
        return selected[: max(0, max_episodes)]
    return selected


def _shared_fingerprint(
    *,
    model_name: str,
    split: str,
    max_episodes: int | None,
    episode_config: EpisodeBuildConfig,
    candidate_config: CandidateGenerationConfig,
    realization_config: RealizationConfig,
    reward_config: RewardConfig,
    selection_config: SelectionConfig,
    similarity_config_path: str,
    strict_mode: bool = False,
    hindsight_samples: list[HindsightSample] | None = None,
) -> str:
    # model_name deliberately excluded — episodes and prompt rows are
    # model-independent, so the cache can be shared across model runs.
    return build_config_fingerprint(
        {
            "split": split,
            "max_episodes": max_episodes,
            "episode_config": episode_config,
            "candidate_config": candidate_config,
            "realization_config": realization_config,
            "reward_config": reward_config,
            "selection_config": selection_config,
            "similarity_config_path": similarity_config_path,
            "strict_mode": strict_mode,
            "hindsight_samples": [
                {
                    "cutoff_month": sample.cutoff_month,
                    "future_paper_id": sample.future_paper_id,
                    "future_paper_published_date": sample.future_paper_published_date,
                    "innovation": innovation_to_dict(sample.innovation),
                }
                for sample in (hindsight_samples or [])
            ],
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
    realization_config: RealizationConfig,
    hindsight_samples: list[HindsightSample] | None,
    similarity_config_path: str,
    runtime_config_path: str | None,
) -> PreparedRLContext | None:
    shared_dir = target_dir / "shared"
    manifest_path = shared_dir / "shared_manifest.json"
    episodes_path = shared_dir / "episodes.json"
    prompt_rows_path = shared_dir / "prompts.jsonl"
    if (
        not manifest_path.exists()
        or not episodes_path.exists()
        or not prompt_rows_path.exists()
    ):
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
        realization_config=realization_config,
        hindsight_samples=list(hindsight_samples or []),
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
    realization_config: RealizationConfig | None = None,
    reward_config: RewardConfig,
    selection_config: SelectionConfig,
    split: str = "train",
    max_episodes: int | None = None,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    strict_mode: bool = False,
    hindsight_samples: list[HindsightSample] | None = None,
) -> PreparedRLContext:
    target_dir = Path(output_dir).resolve()
    shared_dir = target_dir / "shared"
    resolved_realization_config = realization_config or load_realization_config()
    fingerprint = _shared_fingerprint(
        model_name=model_name,
        split=split,
        max_episodes=max_episodes,
        episode_config=episode_config,
        candidate_config=candidate_config,
        realization_config=resolved_realization_config,
        reward_config=reward_config,
        selection_config=selection_config,
        similarity_config_path=similarity_config_path,
        strict_mode=strict_mode,
        hindsight_samples=hindsight_samples,
    )
    cached = _load_cached_context(
        papers=papers,
        target_dir=target_dir,
        fingerprint=fingerprint,
        split=split,
        max_episodes=max_episodes,
        model_name=model_name,
        realization_config=resolved_realization_config,
        hindsight_samples=hindsight_samples,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
    )
    if cached is not None:
        return cached

    episode_cache_root = target_dir / "episode_cache"
    episodes = build_rl_episodes(papers, episode_config, cache_root=episode_cache_root)
    selected_episodes = _select_episodes(episodes, split, max_episodes)
    if not selected_episodes:
        raise ValueError(
            "No RL episodes were selected. Adjust split, window, or date settings."
        )
    prompt_rows = (
        build_strict_rl_prompt_rows(
            papers,
            selected_episodes,
            candidate_config=candidate_config,
            realization_config=resolved_realization_config,
            hindsight_samples=hindsight_samples,
        )
        if strict_mode
        else build_grpo_prompt_rows(
            papers,
            selected_episodes,
            candidate_config=candidate_config,
            realization_config=resolved_realization_config,
            hindsight_samples=hindsight_samples,
        )
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
            "realization_config": asdict(resolved_realization_config),
            "reward_config": asdict(reward_config),
            "selection_config": asdict(selection_config),
            "similarity_config_path": similarity_config_path,
            "model_name": model_name,
            "fingerprint": fingerprint,
            "prompt_mode": "strict_interactive_realization"
            if strict_mode
            else "z_conditioned_realization",
            "strict_mode": strict_mode,
            "strict_contract": strict_runtime_manifest_contract(),
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
        realization_config=resolved_realization_config,
        hindsight_samples=list(hindsight_samples or []),
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
        episode_cache_root=episode_cache_root,
        shared_manifest_path=shared_manifest_path,
    )
