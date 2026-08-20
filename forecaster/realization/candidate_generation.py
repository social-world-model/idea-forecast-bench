"""Generate realization candidates for an episode (API / local HF / heuristic).

Split out of pipeline.py, which had grown to 1,573 lines.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from forecaster.config import RealizationConfig, load_realization_config
from forecaster.models import (
    HindsightSample,
    Innovation,
    realization_trajectory_to_dict,
)
from forecaster.realization.candidates import CandidateListSample, EpisodeCandidateLists
from forecaster.realization.config import (
    CandidateGenerationConfig,
    RewardConfig,
)
from forecaster.realization.episode_prompts import (
    build_grpo_prompt_rows,
)
from forecaster.realization.episodes import (
    RLEpisode,
)
from forecaster.realization.io import _write_json
from forecaster.realization.proposal_generator import (
    generate_local_proposal,
    generate_proposal,
    proposal_to_idea_prediction,
)
from forecaster.realization.reward import (
    build_invalid_reward_evaluation,
    evaluate_rl_reward,
    evaluate_strict_rl_reward,
    serialize_reward_evaluation,
)
from forecaster.realization.strict_runtime import (
    parse_strict_rollout_completion,
    run_strict_realization_rollout,
    serialize_strict_rollout_completion,
)
from live_idea_bench.llm import create_client
from live_idea_bench.models import PaperRecord

logger = logging.getLogger(__name__)


_ALLOWED_BACKENDS = frozenset({"auto", "api", "local_hf", "heuristic"})


_API_MODEL_PREFIXES = ("gpt-4o", "gpt-4", "gpt-3", "claude-", "o1", "o3")


_API_MODEL_SUBSTRINGS = ("gemini",)


def _temperature_schedule(config: CandidateGenerationConfig) -> list[float]:
    if config.num_candidate_lists <= 0:
        return []
    if config.num_candidate_lists == 1:
        return [round((config.min_temperature + config.max_temperature) / 2.0, 4)]
    step = (config.max_temperature - config.min_temperature) / (
        config.num_candidate_lists - 1
    )
    return [
        round(config.min_temperature + (idx * step), 4)
        for idx in range(config.num_candidate_lists)
    ]


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


def _heuristic_proposal_text(
    innovation: Innovation, evidence_papers: list[PaperRecord]
) -> str:
    title = f"{innovation.base_direction.title()} via {innovation.operator.title()}"
    evidence_clause = (
        f" It builds on evidence from {', '.join(paper.title for paper in evidence_papers[:2])}."
        if evidence_papers
        else ""
    )
    body = (
        f"We {innovation.operator} {innovation.base_direction} to address {innovation.gap}."
        f"{evidence_clause} The proposal is grounded in historical work available before the cutoff."
    )
    return f"{title}\n{body}"


def _prediction_from_proposal(
    proposal_text: str,
    innovation: Innovation,
    *,
    backend: str,
    temperature: float,
    top_p: float,
    target_future_paper_id: str,
) -> list[Any]:
    prediction = proposal_to_idea_prediction(proposal_text, innovation, rank=1)
    prediction = replace(
        prediction,
        rank=1,
        metadata={
            **prediction.metadata,
            "proposal_text": proposal_text,
            "sampling_temperature": temperature,
            "sampling_top_p": top_p,
            "generation_backend": backend,
            "target_future_paper_id": target_future_paper_id,
            "prompt_mode": "z_conditioned_realization",
        },
    )
    return [prediction]


def _generate_realization_candidate_predictions(
    row: dict[str, Any],
    model_name: str,
    temperature: float,
    config: CandidateGenerationConfig,
    *,
    realization_config: RealizationConfig,
    top_p: float | None = None,
    seed: int | None = None,
    base_model_name: str | None = None,
    fallback_to_heuristic: bool = False,
) -> list[Any]:
    backend = _resolve_generation_backend(model_name, config.backend)
    top_p_value = top_p if top_p is not None else config.top_p
    innovation = Innovation(**dict(row.get("innovation", {})))
    train_papers = [PaperRecord(**paper) for paper in row.get("train_papers", [])]
    evidence_papers = [PaperRecord(**paper) for paper in row.get("evidence_papers", [])]
    target_future_paper_id = str(row.get("target_future_paper_id", "") or "")
    if backend == "heuristic":
        proposal_text = _heuristic_proposal_text(innovation, evidence_papers)
        return _prediction_from_proposal(
            proposal_text,
            innovation,
            backend=backend,
            temperature=temperature,
            top_p=top_p_value,
            target_future_paper_id=target_future_paper_id,
        )
    if backend == "api":
        client, resolved_model = create_client(model_name)
        proposal_text = generate_proposal(
            innovation=innovation,
            evidence=evidence_papers,
            context_papers=train_papers,
            llm_client=client,
            model=resolved_model,
            config=realization_config,
            temperature=temperature,
            top_p=top_p_value,
            seed=seed,
        )
        return _prediction_from_proposal(
            proposal_text,
            innovation,
            backend=backend,
            temperature=temperature,
            top_p=top_p_value,
            target_future_paper_id=target_future_paper_id,
        )
    if backend == "local_hf":
        proposal_text = generate_local_proposal(
            innovation=innovation,
            evidence=evidence_papers,
            context_papers=train_papers,
            model_name_or_path=model_name,
            config=realization_config,
            base_model_name=base_model_name,
            temperature=temperature,
            top_p=top_p_value,
            seed=seed,
        )
        return _prediction_from_proposal(
            proposal_text,
            innovation,
            backend=backend,
            temperature=temperature,
            top_p=top_p_value,
            target_future_paper_id=target_future_paper_id,
        )
    if fallback_to_heuristic:
        proposal_text = _heuristic_proposal_text(innovation, evidence_papers)
        return _prediction_from_proposal(
            proposal_text,
            innovation,
            backend="heuristic_fallback",
            temperature=temperature,
            top_p=top_p_value,
            target_future_paper_id=target_future_paper_id,
        )
    raise ValueError(f"Unsupported candidate generation backend: {backend}")


def _generate_strict_realization_completion(
    row: dict[str, Any],
    model_name: str,
    temperature: float,
    config: CandidateGenerationConfig,
    *,
    realization_config: RealizationConfig,
    top_p: float | None = None,
    seed: int | None = None,
    base_model_name: str | None = None,
) -> tuple[str, str]:
    backend = _resolve_generation_backend(model_name, config.backend)
    innovation = Innovation(**dict(row.get("innovation", {})))
    train_papers = [PaperRecord(**paper) for paper in row.get("train_papers", [])]
    search_env = row.get("search_env", {})
    top_p_value = top_p if top_p is not None else config.top_p
    if backend == "heuristic":
        trajectory, _ = run_strict_realization_rollout(
            innovation,
            train_papers,
            llm_client=None,
            model=None,
            realization_config=realization_config,
            search_env_payload=search_env if isinstance(search_env, dict) else None,
            backend="heuristic",
        )
        return (
            serialize_strict_rollout_completion(trajectory),
            backend,
        )
    if backend == "api":
        client, resolved_model = create_client(model_name)
        trajectory, _ = run_strict_realization_rollout(
            innovation,
            train_papers,
            llm_client=client,
            model=resolved_model,
            realization_config=realization_config,
            search_env_payload=search_env if isinstance(search_env, dict) else None,
            temperature=temperature,
            top_p=top_p_value,
            seed=seed,
        )
        return (
            serialize_strict_rollout_completion(trajectory),
            backend,
        )
    if backend == "local_hf":
        trajectory, _ = run_strict_realization_rollout(
            innovation,
            train_papers,
            llm_client=None,
            model=None,
            realization_config=realization_config,
            realization_model_path=model_name,
            search_env_payload=search_env if isinstance(search_env, dict) else None,
            temperature=temperature,
            top_p=top_p_value,
            seed=seed,
            base_model_name=base_model_name,
        )
        return (
            serialize_strict_rollout_completion(trajectory),
            backend,
        )
    raise ValueError(f"Unsupported strict generation backend: {backend}")


def _strict_prediction_from_completion(
    row: dict[str, Any],
    completion_text: str,
    *,
    backend: str,
    temperature: float,
    top_p: float,
    realization_config: RealizationConfig,
) -> list[Any]:
    innovation = Innovation(**dict(row.get("innovation", {})))
    train_papers = [PaperRecord(**paper) for paper in row.get("train_papers", [])]
    trajectory = parse_strict_rollout_completion(completion_text)
    if trajectory is None or trajectory.invalid_reason or trajectory.result is None:
        return []
    paper_lookup = {paper.paper_id: paper for paper in train_papers}
    evidence_papers = [
        paper_lookup[paper_id]
        for paper_id in trajectory.result.selected_evidence_ids
        if paper_id in paper_lookup
    ]
    prediction = proposal_to_idea_prediction(
        trajectory.result.proposal_text,
        innovation,
        rank=1,
    )
    prediction = replace(
        prediction,
        rank=1,
        metadata={
            **prediction.metadata,
            "proposal_text": trajectory.result.proposal_text,
            "policy_rollout": completion_text,
            "strict_trajectory": realization_trajectory_to_dict(trajectory),
            "selected_evidence_ids": list(trajectory.result.selected_evidence_ids),
            "search_queries": list(trajectory.result.search_queries),
            "surfaced_paper_ids_by_step": [
                [observation.paper_id for observation in step.observation]
                for step in trajectory.steps
                if step.action.action_type == "search"
            ],
            "sampling_temperature": temperature,
            "sampling_top_p": top_p,
            "generation_backend": backend,
            "prompt_mode": "strict_interactive_realization",
            "evidence_paper_ids": [paper.paper_id for paper in evidence_papers],
            "target_future_paper_id": str(row.get("target_future_paper_id", "") or ""),
        },
    )
    return [prediction]


def _single_idea_candidate_config(
    config: CandidateGenerationConfig,
) -> CandidateGenerationConfig:
    return replace(config, ideas_per_list=1)


def generate_episode_candidate_lists(
    papers: list[PaperRecord],
    episodes: list[RLEpisode],
    *,
    model_name: str,
    candidate_config: CandidateGenerationConfig,
    reward_config: RewardConfig,
    realization_config: RealizationConfig | None = None,
    prompt_rows: list[dict[str, Any]] | None = None,
    hindsight_samples: list[HindsightSample] | None = None,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    base_model_name: str | None = None,
    fallback_to_heuristic: bool = False,
) -> list[EpisodeCandidateLists]:
    single_config = _single_idea_candidate_config(candidate_config)
    temperatures = _temperature_schedule(single_config)
    resolved_realization_config = realization_config or load_realization_config()
    resolved_rows = prompt_rows
    if resolved_rows is None:
        resolved_rows = build_grpo_prompt_rows(
            papers,
            episodes,
            candidate_config=single_config,
            realization_config=resolved_realization_config,
            hindsight_samples=hindsight_samples,
        )
    outputs: list[EpisodeCandidateLists] = []
    for row in resolved_rows:
        episode_payload = row.get("episode", {})
        if not isinstance(episode_payload, dict):
            continue
        episode = RLEpisode(**episode_payload)
        train_papers = [PaperRecord(**paper) for paper in row.get("train_papers", [])]
        future_papers = [PaperRecord(**paper) for paper in row.get("future_papers", [])]
        innovation = Innovation(**dict(row.get("innovation", {})))
        evidence_papers = [
            PaperRecord(**paper) for paper in row.get("evidence_papers", [])
        ]
        prompt = str(row.get("prompt", ""))
        prompt_mode = str(
            row.get("prompt_mode", "z_conditioned_realization")
            or "z_conditioned_realization"
        )
        candidates: list[CandidateListSample] = []
        for temperature in temperatures:
            try:
                if prompt_mode == "strict_interactive_realization":
                    completion_text, backend = _generate_strict_realization_completion(
                        row,
                        model_name,
                        temperature,
                        single_config,
                        realization_config=resolved_realization_config,
                        top_p=single_config.top_p,
                        seed=single_config.seed + int(temperature * 1000),
                        base_model_name=base_model_name,
                    )
                    predictions = _strict_prediction_from_completion(
                        row,
                        completion_text,
                        backend=backend,
                        temperature=temperature,
                        top_p=single_config.top_p,
                        realization_config=resolved_realization_config,
                    )
                else:
                    backend = _resolve_generation_backend(
                        model_name, single_config.backend
                    )
                    predictions = _generate_realization_candidate_predictions(
                        row,
                        model_name,
                        temperature,
                        single_config,
                        realization_config=resolved_realization_config,
                        top_p=single_config.top_p,
                        seed=single_config.seed + int(temperature * 1000),
                        base_model_name=base_model_name,
                        fallback_to_heuristic=fallback_to_heuristic,
                    )
            except Exception:
                if not fallback_to_heuristic:
                    raise
                if prompt_mode == "strict_interactive_realization":
                    completion_text, _ = _generate_strict_realization_completion(
                        row,
                        model_name,
                        temperature,
                        replace(single_config, backend="heuristic"),
                        realization_config=resolved_realization_config,
                        top_p=single_config.top_p,
                        seed=single_config.seed + int(temperature * 1000),
                        base_model_name=base_model_name,
                    )
                    predictions = _strict_prediction_from_completion(
                        row,
                        completion_text,
                        backend="heuristic_fallback",
                        temperature=temperature,
                        top_p=single_config.top_p,
                        realization_config=resolved_realization_config,
                    )
                    backend = "heuristic_fallback"
                else:
                    predictions = _generate_realization_candidate_predictions(
                        row,
                        model_name,
                        temperature,
                        replace(single_config, backend="heuristic"),
                        realization_config=resolved_realization_config,
                        top_p=single_config.top_p,
                        seed=single_config.seed + int(temperature * 1000),
                        base_model_name=base_model_name,
                        fallback_to_heuristic=True,
                    )
                    backend = "heuristic_fallback"
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
            if prompt_mode == "strict_interactive_realization":
                policy_rollout = str(completion_text or "")
                reward = (
                    evaluate_strict_rl_reward(
                        policy_rollout,
                        innovation=innovation,
                        train_papers=train_papers,
                        future_papers=future_papers,
                        reward_config=reward_config,
                        realization_config=resolved_realization_config,
                        search_env_payload=row.get("search_env")
                        if isinstance(row.get("search_env"), dict)
                        else None,
                        similarity_config_path=similarity_config_path,
                        runtime_config_path=runtime_config_path,
                        model_name=model_name,
                        cutoff_date=episode.cutoff_date,
                        future_end_date=episode.future_end_date,
                    )
                    if policy_rollout
                    else build_invalid_reward_evaluation(reward_config)
                )
            elif len(predictions) != 1:
                reward = build_invalid_reward_evaluation(reward_config)
            else:
                reward = evaluate_rl_reward(
                    predictions=predictions,
                    train_papers=train_papers,
                    future_papers=future_papers,
                    reward_config=reward_config,
                    innovation=innovation,
                    evidence_papers=evidence_papers,
                    proposal_text=str(
                        predictions[0].metadata.get("proposal_text", "") or ""
                    ),
                    realization_config=resolved_realization_config,
                    similarity_config_path=similarity_config_path,
                    runtime_config_path=runtime_config_path,
                    model_name=model_name,
                    cutoff_date=episode.cutoff_date,
                    future_end_date=episode.future_end_date,
                )
            candidates.append(
                CandidateListSample(predictions=predictions, reward=reward)
            )
        episode_candidates = EpisodeCandidateLists(
            episode=episode, prompt=prompt, candidates=candidates
        )
        outputs.append(episode_candidates)
        if episode.cache_artifact_path:
            _write_json(
                Path(episode.cache_artifact_path),
                {
                    "episode": asdict(episode),
                    "prompt": prompt,
                    "prompt_mode": prompt_mode,
                    "innovation": row.get("innovation", {}),
                    "target_future_paper_id": row.get("target_future_paper_id", ""),
                    "evidence_papers": row.get("evidence_papers", []),
                    "search_env": row.get("search_env", {}),
                    "realization_config": row.get("realization_config", {}),
                    "candidates": [
                        {
                            "predictions": [
                                asdict(prediction)
                                for prediction in candidate.predictions
                            ],
                            "reward": serialize_reward_evaluation(candidate.reward),
                        }
                        for candidate in candidates
                    ],
                },
            )
    return outputs


def serialize_episode_candidate_lists(
    episodes: list[EpisodeCandidateLists],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_batch in episodes:
        rows.append(
            {
                "episode": asdict(episode_batch.episode),
                "prompt": episode_batch.prompt,
                "candidates": [
                    {
                        "predictions": [
                            asdict(prediction) for prediction in candidate.predictions
                        ],
                        "reward": serialize_reward_evaluation(candidate.reward),
                    }
                    for candidate in episode_batch.candidates
                ],
            }
        )
    return rows
