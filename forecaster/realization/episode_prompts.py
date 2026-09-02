"""Turn RL episodes into the prompt rows the trainer consumes.

Split out of pipeline.py, which had grown to 1,573 lines.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from forecaster.config import RealizationConfig, load_realization_config
from forecaster.models import (
    HindsightSample,
    Innovation,
    innovation_to_dict,
    strict_runtime_manifest_contract,
    strict_search_contract,
)
from forecaster.realization.config import (
    CandidateGenerationConfig,
)
from forecaster.realization.episodes import (
    RLEpisode,
)
from forecaster.realization.evidence import retrieve_evidence
from forecaster.realization.proposal_generator import (
    build_realization_messages,
)
from forecaster.realization.strict_runtime import (
    build_strict_interactive_messages,
)
from idea_forecast_bench.models import PaperRecord

logger = logging.getLogger(__name__)


_OPERATOR_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("benchmark", ("benchmark", "evaluation", "leaderboard", "dataset")),
    ("compose", ("combine", "composition", "integrat", "hybrid")),
    ("transfer", ("transfer", "cross-domain", "domain adaptation", "generalize")),
    ("simplify", ("efficient", "efficiency", "lightweight", "compact")),
    ("scale", ("scal", "large-scale", "billion", "massive")),
    ("adapt", ("adapt", "fine-tun", "personaliz", "customiz")),
    ("analyze", ("analysis", "analyze", "study", "empirical")),
)


_METRIC_SAMPLE_CAP = 30


def _paper_lookup(papers: list[PaperRecord]) -> dict[str, PaperRecord]:
    return {paper.paper_id: paper for paper in papers}


def _materialize_episode(
    episode: RLEpisode, paper_lookup: dict[str, PaperRecord]
) -> tuple[list[PaperRecord], list[PaperRecord]]:
    train = [
        paper_lookup[paper_id]
        for paper_id in episode.train_paper_ids
        if paper_id in paper_lookup
    ]
    future = [
        paper_lookup[paper_id]
        for paper_id in episode.future_paper_ids
        if paper_id in paper_lookup
    ]
    return train, future


def _serialize_papers(papers: list[PaperRecord]) -> list[dict[str, Any]]:
    return [asdict(paper) for paper in papers]


def _infer_operator_from_future_paper(future_paper: PaperRecord) -> str:
    text = f"{future_paper.title} {future_paper.summary}".lower()
    for operator, hints in _OPERATOR_HINTS:
        if any(hint in text for hint in hints):
            return operator
    return "extend"


def _derive_episode_innovation(future_paper: PaperRecord) -> Innovation:
    base_terms = (
        future_paper.keywords[:3]
        if future_paper.keywords
        else future_paper.title.split()[:4]
    )
    base_direction = " ".join(
        term.strip() for term in base_terms if str(term).strip()
    ).strip()
    if not base_direction:
        base_direction = (
            " ".join(future_paper.title.split()[:4]).strip() or "emerging direction"
        )
    gap = (
        future_paper.summary[:220].strip()
        if future_paper.summary.strip()
        else future_paper.title.strip()
    )
    return Innovation(
        base_direction=base_direction,
        operator=_infer_operator_from_future_paper(future_paper),
        gap=gap,
    )


def _select_episode_target_future_paper(
    future_papers: list[PaperRecord],
) -> PaperRecord | None:
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
                key=lambda sample: (
                    sample.cutoff_month,
                    sample.future_paper_published_date,
                ),
                reverse=True,
            )
            return paper_matches[0].innovation
    return _derive_episode_innovation(target_future_paper)


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
    for sample in hindsight_samples or []:
        hs_by_paper.setdefault(sample.future_paper_id, []).append(sample)

    for i, episode in enumerate(episodes):
        train_papers, future_papers = _materialize_episode(episode, paper_lookup)
        future_ids = {p.paper_id for p in future_papers}

        # Collect all hindsight samples whose future paper falls in this episode
        episode_samples = [s for pid in future_ids for s in hs_by_paper.get(pid, [])]

        if episode_samples:
            # One training row per hindsight sample (each has a unique innovation z)
            logger.info(
                "Episode %d/%d: %d train papers, %d future papers, %d hindsight samples",
                i + 1,
                len(episodes),
                len(train_papers),
                len(future_papers),
                len(episode_samples),
            )
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
            logger.info(
                "Episode %d/%d: %d train papers, %d future papers, 0 hindsight samples (fallback)",
                i + 1,
                len(episodes),
                len(train_papers),
                len(future_papers),
            )
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
