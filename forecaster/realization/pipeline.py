from __future__ import annotations

import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from live_idea_bench.llm import create_client
from live_idea_bench.models import PaperRecord
from forecaster.config import RealizationConfig, load_realization_config
from forecaster.models import (
    HindsightSample,
    Innovation,
    innovation_to_dict,
    realization_trajectory_to_dict,
    strict_search_contract,
    strict_runtime_manifest_contract,
)
from forecaster.realization.candidates import CandidateListSample, EpisodeCandidateLists
from forecaster.realization.config import CandidateGenerationConfig, EpisodeBuildConfig, RewardConfig, SelectionConfig
from forecaster.realization.evidence import retrieve_evidence
from forecaster.realization.episodes import RLEpisode, build_rl_episodes, serialize_episodes
from forecaster.realization.grpo import compute_reward_alignment
from forecaster.realization.io import _read_json, _read_jsonl, _write_json, _write_jsonl
from forecaster.realization.model_zoo import list_small_model_payloads
from forecaster.realization.proposal_generator import (
    build_realization_messages,
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
    build_strict_interactive_messages,
    parse_strict_rollout_completion,
    run_strict_realization_rollout,
    serialize_strict_rollout_completion,
)
from forecaster.realization.trainers import PreparedRLContext, TrainerPreparedArtifacts, build_config_fingerprint, create_trainer_runner


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
_OPERATOR_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("benchmark", ("benchmark", "evaluation", "leaderboard", "dataset")),
    ("compose", ("combine", "composition", "integrat", "hybrid")),
    ("transfer", ("transfer", "cross-domain", "domain adaptation", "generalize")),
    ("simplify", ("efficient", "efficiency", "lightweight", "compact")),
    ("scale", ("scal", "large-scale", "billion", "massive")),
    ("adapt", ("adapt", "fine-tun", "personaliz", "customiz")),
    ("analyze", ("analysis", "analyze", "study", "empirical")),
)


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


def _infer_operator_from_future_paper(future_paper: PaperRecord) -> str:
    text = f"{future_paper.title} {future_paper.summary}".lower()
    for operator, hints in _OPERATOR_HINTS:
        if any(hint in text for hint in hints):
            return operator
    return "extend"


def _derive_episode_innovation(future_paper: PaperRecord) -> Innovation:
    base_terms = future_paper.keywords[:3] if future_paper.keywords else future_paper.title.split()[:4]
    base_direction = " ".join(term.strip() for term in base_terms if str(term).strip()).strip()
    if not base_direction:
        base_direction = " ".join(future_paper.title.split()[:4]).strip() or "emerging direction"
    gap = future_paper.summary[:220].strip() if future_paper.summary.strip() else future_paper.title.strip()
    return Innovation(
        base_direction=base_direction,
        operator=_infer_operator_from_future_paper(future_paper),
        gap=gap,
    )


def _select_episode_target_future_paper(future_papers: list[PaperRecord]) -> PaperRecord | None:
    if not future_papers:
        return None
    return sorted(
        future_papers,
        key=lambda paper: (paper.published_date or f"{paper.month}-01", paper.paper_id),
    )[0]


def _resolve_episode_innovation(
    episode: RLEpisode,
    target_future_paper: PaperRecord,
    hindsight_samples: list[HindsightSample] | None,
) -> Innovation:
    if hindsight_samples:
        exact_matches = [
            sample
            for sample in hindsight_samples
            if sample.future_paper_id == target_future_paper.paper_id
            and sample.cutoff_month == episode.cutoff_month
        ]
        if exact_matches:
            return exact_matches[0].innovation

        paper_matches = [
            sample
            for sample in hindsight_samples
            if sample.future_paper_id == target_future_paper.paper_id
        ]
        if paper_matches:
            paper_matches.sort(
                key=lambda sample: (sample.cutoff_month, sample.future_paper_published_date),
                reverse=True,
            )
            return paper_matches[0].innovation
    return _derive_episode_innovation(target_future_paper)


_METRIC_SAMPLE_CAP = 30


def _sample_for_metric(papers: list[PaperRecord], cap: int) -> list[PaperRecord]:
    """Deterministic cap for the per-row metric-reward paper set.

    Keeps the head of the list (already ordered by month/popularity upstream)
    so per-row payload stays under ~60 KB even on large episodes. The cap
    only affects what ships into ``extra_info`` for the
    soft/coverage/novelty rewards; the composite reward path is unchanged.
    """
    if cap <= 0 or len(papers) <= cap:
        return list(papers)
    return list(papers[:cap])


def _serialize_episode_prompt_row(
    *,
    episode: RLEpisode,
    train_papers: list[PaperRecord],
    future_papers: list[PaperRecord],
    target_future_paper: PaperRecord,
    innovation: Innovation,
    evidence_papers: list[PaperRecord],
    realization_config: RealizationConfig,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    # Only serialize evidence papers (top-5) and target future paper into the
    # prompt row — NOT the full train/future sets (50k+ papers each, causing
    # 31GB+ prompts.jsonl and OOM). The reward function uses evidence_papers
    # for the dense reward (evidence accuracy, operator adherence, coherence)
    # and the target paper for future matching.
    #
    # For the single-metric GRPO rewards (soft / coverage / novelty) we
    # additionally ship a capped sample of train + future papers so the
    # reward has enough signal — coverage clustering and novelty cosine
    # both degenerate with only one paper per side. The cap keeps each row
    # well under ~100 KB so 6-episode runs still produce small JSONL.
    return {
        "episode": asdict(episode),
        "prompt_mode": "z_conditioned_realization",
        "prompt": f"{system_prompt}\n\n{user_prompt}".strip(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "cutoff_month": episode.cutoff_month,
        "cutoff_date": episode.cutoff_date,
        "future_end_month": episode.future_end_month,
        "future_end_date": episode.future_end_date,
        "train_papers": _serialize_papers(evidence_papers),
        "future_papers": _serialize_papers([target_future_paper]),
        "metric_train_papers": _serialize_papers(
            _sample_for_metric(train_papers, _METRIC_SAMPLE_CAP)
        ),
        "metric_future_papers": _serialize_papers(
            _sample_for_metric(future_papers, _METRIC_SAMPLE_CAP)
        ),
        "target_future_paper": asdict(target_future_paper),
        "target_future_paper_id": target_future_paper.paper_id,
        "innovation": innovation_to_dict(innovation),
        "evidence_papers": _serialize_papers(evidence_papers),
        "realization_config": asdict(realization_config),
    }


def _build_strict_interactive_prompt(innovation: Innovation) -> tuple[str, str]:
    return build_strict_interactive_messages(innovation)


def _serialize_strict_episode_prompt_row(
    *,
    episode: RLEpisode,
    train_papers: list[PaperRecord],
    future_papers: list[PaperRecord],
    target_future_paper: PaperRecord,
    innovation: Innovation,
    realization_config: RealizationConfig,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    return {
        "episode": asdict(episode),
        "prompt_mode": "strict_interactive_realization",
        "prompt": f"{system_prompt}\n\n{user_prompt}".strip(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "cutoff_month": episode.cutoff_month,
        "cutoff_date": episode.cutoff_date,
        "future_end_month": episode.future_end_month,
        "future_end_date": episode.future_end_date,
        "train_papers": _serialize_papers(train_papers),
        "future_papers": _serialize_papers(future_papers),
        "target_future_paper": asdict(target_future_paper),
        "target_future_paper_id": target_future_paper.paper_id,
        "innovation": innovation_to_dict(innovation),
        "evidence_papers": [],
        "realization_config": asdict(realization_config),
        "search_env": strict_search_contract()["search_env_defaults"],
        "strict_contract": strict_runtime_manifest_contract(),
    }


def _build_episode_prompt_row(
    episode: RLEpisode,
    train_papers: list[PaperRecord],
    future_papers: list[PaperRecord],
    *,
    realization_config: RealizationConfig,
    hindsight_samples: list[HindsightSample] | None,
) -> dict[str, Any] | None:
    target_future_paper = _select_episode_target_future_paper(future_papers)
    if target_future_paper is None:
        return None
    innovation = _resolve_episode_innovation(
        episode,
        target_future_paper,
        hindsight_samples,
    )
    evidence_papers = retrieve_evidence(
        innovation,
        train_papers,
        top_k=realization_config.evidence_top_k,
        similarity_threshold=realization_config.evidence_similarity_threshold,
    )
    system_prompt, user_prompt = build_realization_messages(
        innovation,
        evidence_papers,
        context_papers=train_papers,
        config=realization_config,
    )
    return _serialize_episode_prompt_row(
        episode=episode,
        train_papers=train_papers,
        future_papers=future_papers,
        target_future_paper=target_future_paper,
        innovation=innovation,
        evidence_papers=evidence_papers,
        realization_config=realization_config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _build_strict_episode_prompt_row(
    episode: RLEpisode,
    train_papers: list[PaperRecord],
    future_papers: list[PaperRecord],
    *,
    realization_config: RealizationConfig,
    hindsight_samples: list[HindsightSample] | None,
) -> dict[str, Any] | None:
    target_future_paper = _select_episode_target_future_paper(future_papers)
    if target_future_paper is None:
        return None
    innovation = _resolve_episode_innovation(
        episode,
        target_future_paper,
        hindsight_samples,
    )
    system_prompt, user_prompt = _build_strict_interactive_prompt(innovation)
    return _serialize_strict_episode_prompt_row(
        episode=episode,
        train_papers=train_papers,
        future_papers=future_papers,
        target_future_paper=target_future_paper,
        innovation=innovation,
        realization_config=realization_config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _heuristic_proposal_text(innovation: Innovation, evidence_papers: list[PaperRecord]) -> str:
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


def _single_idea_candidate_config(config: CandidateGenerationConfig) -> CandidateGenerationConfig:
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
    paper_lookup = _paper_lookup(papers)
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
        evidence_papers = [PaperRecord(**paper) for paper in row.get("evidence_papers", [])]
        prompt = str(row.get("prompt", ""))
        prompt_mode = str(row.get("prompt_mode", "z_conditioned_realization") or "z_conditioned_realization")
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
                    backend = _resolve_generation_backend(model_name, single_config.backend)
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
                reward = evaluate_strict_rl_reward(
                    policy_rollout,
                    innovation=innovation,
                    train_papers=train_papers,
                    future_papers=future_papers,
                    reward_config=reward_config,
                    realization_config=resolved_realization_config,
                    search_env_payload=row.get("search_env") if isinstance(row.get("search_env"), dict) else None,
                    similarity_config_path=similarity_config_path,
                    runtime_config_path=runtime_config_path,
                    model_name=model_name,
                    cutoff_date=episode.cutoff_date,
                    future_end_date=episode.future_end_date,
                ) if policy_rollout else build_invalid_reward_evaluation(reward_config)
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
                    proposal_text=str(predictions[0].metadata.get("proposal_text", "") or ""),
                    realization_config=resolved_realization_config,
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
                    "prompt_mode": prompt_mode,
                    "innovation": row.get("innovation", {}),
                    "target_future_paper_id": row.get("target_future_paper_id", ""),
                    "evidence_papers": row.get("evidence_papers", []),
                    "search_env": row.get("search_env", {}),
                    "realization_config": row.get("realization_config", {}),
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
    realization_config: RealizationConfig | None = None,
    hindsight_samples: list[HindsightSample] | None = None,
) -> list[dict[str, Any]]:
    paper_lookup = _paper_lookup(papers)
    resolved_realization_config = realization_config or load_realization_config()
    rows: list[dict[str, Any]] = []

    # Index hindsight samples by future_paper_id for fast lookup
    hs_by_paper: dict[str, list[HindsightSample]] = {}
    for sample in (hindsight_samples or []):
        hs_by_paper.setdefault(sample.future_paper_id, []).append(sample)

    for i, episode in enumerate(episodes):
        train_papers, future_papers = _materialize_episode(episode, paper_lookup)
        future_ids = {p.paper_id for p in future_papers}

        # Collect all hindsight samples whose future paper falls in this episode
        episode_samples = [
            s for pid in future_ids
            for s in hs_by_paper.get(pid, [])
        ]

        if episode_samples:
            # One training row per hindsight sample (each has a unique innovation z)
            logger.info("Episode %d/%d: %d train papers, %d future papers, %d hindsight samples",
                         i + 1, len(episodes), len(train_papers), len(future_papers), len(episode_samples))
            for sample in episode_samples:
                target = paper_lookup.get(sample.future_paper_id)
                if target is None:
                    continue
                row = _build_episode_prompt_row(
                    episode,
                    train_papers,
                    [target],
                    realization_config=resolved_realization_config,
                    hindsight_samples=[sample],
                )
                if row is not None:
                    rows.append(row)
        else:
            # Fallback: single row from first future paper
            logger.info("Episode %d/%d: %d train papers, %d future papers, 0 hindsight samples (fallback)",
                         i + 1, len(episodes), len(train_papers), len(future_papers))
            row = _build_episode_prompt_row(
                episode,
                train_papers,
                future_papers,
                realization_config=resolved_realization_config,
                hindsight_samples=hindsight_samples,
            )
            if row is not None:
                rows.append(row)

    logger.info("Built %d GRPO prompt rows from %d episodes", len(rows), len(episodes))
    return rows


def build_strict_rl_prompt_rows(
    papers: list[PaperRecord],
    episodes: list[RLEpisode],
    *,
    candidate_config: CandidateGenerationConfig,
    realization_config: RealizationConfig | None = None,
    hindsight_samples: list[HindsightSample] | None = None,
) -> list[dict[str, Any]]:
    del candidate_config
    paper_lookup = _paper_lookup(papers)
    resolved_realization_config = realization_config or load_realization_config()
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        train_papers, future_papers = _materialize_episode(episode, paper_lookup)
        row = _build_strict_episode_prompt_row(
            episode,
            train_papers,
            future_papers,
            realization_config=resolved_realization_config,
            hindsight_samples=hindsight_samples,
        )
        if row is not None:
            rows.append(row)
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
            "RL training is restricted to the train split. "
            "Use prepare_only if you need to inspect validation/test/all artifacts."
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
        raise ValueError("No RL episodes were selected. Adjust split, window, or date settings.")
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
            "prompt_mode": "strict_interactive_realization" if strict_mode else "z_conditioned_realization",
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


def _alignment_episodes(common_context: PreparedRLContext) -> list[RLEpisode]:
    validation_episodes = [episode for episode in common_context.all_episodes if getattr(episode, "split", "") == "validation"]
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
    prompt_mode = str(prompt_row.get("prompt_mode", "z_conditioned_realization") or "z_conditioned_realization")
    train_papers = [PaperRecord(**paper) for paper in prompt_row.get("train_papers", [])]
    future_papers = [PaperRecord(**paper) for paper in prompt_row.get("future_papers", [])]
    innovation = Innovation(**dict(prompt_row.get("innovation", {})))
    evidence_papers = [PaperRecord(**paper) for paper in prompt_row.get("evidence_papers", [])]
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
            search_env_payload=prompt_row.get("search_env") if isinstance(prompt_row.get("search_env"), dict) else None,
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
    evaluations = [candidate.reward for batch in candidate_lists for candidate in batch.candidates]
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
        "trainer_policy_manifest_path": str((prepared.output_dir / "policy_manifest.json").resolve()) if trainer_manifest else "",
        "prepare_only": prepare_only,
        "selection_config_path": selection_config_path,
        "prompt_mode": "strict_interactive_realization" if strict_mode else "z_conditioned_realization",
        "strict_mode": strict_mode,
        "recommended_small_models": list_small_model_payloads(),
        "shared_fingerprint": common_context.config_fingerprint,
        "trainer_metadata": prepared.metadata,
        "diagnostics": diagnostics,
        "strict_contract": strict_runtime_manifest_contract(),
    }
    _write_json(target_dir / "pipeline_manifest.json", manifest)
    return manifest
