"""Shared helpers for topic-based hindsight manifest and preview scripts."""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from live_idea_bench.backtest import split_train_future_by_cutoff
from live_idea_bench.config import TopicDefinition, load_topics
from live_idea_bench.models import PaperRecord
from live_idea_bench.papers import (
    date_to_ordinal,
    get_paper_published_date,
    load_papers_from_markdown,
)
from live_idea_bench.topics import classify_papers_by_topic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOPICS_CONFIG_PATH = "config/topics_v2.yaml"
TOPIC_HINDSIGHT_START_MONTH = "2023-01"
TOPIC_HINDSIGHT_END_MONTH = "2025-03"
TOPIC_HINDSIGHT_HORIZON_MONTHS = 3


@dataclass(frozen=True)
class TopicHindsightEpisode:
    episode_id: str
    cutoff_date: str
    future_start_date: str
    future_end_date: str

    @property
    def cutoff_month(self) -> str:
        return self.cutoff_date[:7]

    def to_dict(self) -> dict[str, str]:
        return {
            "episode_id": self.episode_id,
            "cutoff_date": self.cutoff_date,
            "cutoff_month": self.cutoff_month,
            "future_start_date": self.future_start_date,
            "future_end_date": self.future_end_date,
        }


@dataclass(frozen=True)
class TopicHindsightContext:
    input_dir: Path
    topics_config_path: Path
    papers: tuple[PaperRecord, ...]
    topics: tuple[TopicDefinition, ...]
    grouped_papers: OrderedDict[str, tuple[PaperRecord, ...]]


FIXED_TOPIC_HINDSIGHT_EPISODES: tuple[TopicHindsightEpisode, ...] = (
    TopicHindsightEpisode(
        episode_id="E1",
        cutoff_date="2023-03-31",
        future_start_date="2023-04-01",
        future_end_date="2023-06-30",
    ),
    TopicHindsightEpisode(
        episode_id="E2",
        cutoff_date="2023-06-30",
        future_start_date="2023-07-01",
        future_end_date="2023-09-30",
    ),
    TopicHindsightEpisode(
        episode_id="E3",
        cutoff_date="2023-09-30",
        future_start_date="2023-10-01",
        future_end_date="2023-12-31",
    ),
    TopicHindsightEpisode(
        episode_id="E4",
        cutoff_date="2023-12-31",
        future_start_date="2024-01-01",
        future_end_date="2024-03-31",
    ),
    TopicHindsightEpisode(
        episode_id="E5",
        cutoff_date="2024-03-31",
        future_start_date="2024-04-01",
        future_end_date="2024-06-30",
    ),
    TopicHindsightEpisode(
        episode_id="E6",
        cutoff_date="2024-06-30",
        future_start_date="2024-07-01",
        future_end_date="2024-09-30",
    ),
)


def _resolve_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw.resolve()
    return (PROJECT_ROOT / raw).resolve()


def load_topic_hindsight_context(
    input_dir: str | Path,
    *,
    topics_config_path: str = DEFAULT_TOPICS_CONFIG_PATH,
) -> TopicHindsightContext:
    resolved_input_dir = Path(input_dir).resolve()
    resolved_topics_config = _resolve_path(topics_config_path)
    papers = tuple(
        load_papers_from_markdown(
            resolved_input_dir,
            start_month=TOPIC_HINDSIGHT_START_MONTH,
            end_month=TOPIC_HINDSIGHT_END_MONTH,
        )
    )
    topics = tuple(load_topics(str(resolved_topics_config)))
    grouped_raw = classify_papers_by_topic(list(papers), list(topics))
    grouped = OrderedDict(
        (topic.id, tuple(grouped_raw.get(topic.id, [])))
        for topic in topics
    )
    return TopicHindsightContext(
        input_dir=resolved_input_dir,
        topics_config_path=resolved_topics_config,
        papers=papers,
        topics=topics,
        grouped_papers=grouped,
    )


def _sort_papers_chronologically(papers: list[PaperRecord]) -> list[PaperRecord]:
    return sorted(
        papers,
        key=lambda paper: (
            date_to_ordinal(get_paper_published_date(paper)),
            paper.paper_id,
        ),
    )


def sample_future_papers_deterministically(
    future_papers: list[PaperRecord],
    *,
    limit: int = 2,
) -> list[PaperRecord]:
    if limit <= 0:
        raise ValueError("limit must be positive")

    ordered = _sort_papers_chronologically(future_papers)
    if len(ordered) <= limit:
        return ordered

    selected: list[PaperRecord] = []
    seen_paper_ids: set[str] = set()
    for slot in range(limit):
        quantile = (slot + 0.5) / limit
        index = min(len(ordered) - 1, int(quantile * len(ordered)))
        paper = ordered[index]
        if paper.paper_id in seen_paper_ids:
            continue
        seen_paper_ids.add(paper.paper_id)
        selected.append(paper)
    return selected


def build_topic_hindsight_manifest(
    context: TopicHindsightContext,
    *,
    per_topic_per_episode: int = 2,
) -> dict[str, Any]:
    if per_topic_per_episode <= 0:
        raise ValueError("per_topic_per_episode must be positive")

    rows: list[dict[str, Any]] = []
    total_selected_samples = 0

    for topic in context.topics:
        topic_papers = list(context.grouped_papers.get(topic.id, ()))
        for episode in FIXED_TOPIC_HINDSIGHT_EPISODES:
            train_papers, future_papers, _future_end_month, _future_end_date = split_train_future_by_cutoff(
                papers=topic_papers,
                cutoff_date=episode.cutoff_date,
                horizon_months=TOPIC_HINDSIGHT_HORIZON_MONTHS,
            )
            selected_future = sample_future_papers_deterministically(
                future_papers,
                limit=per_topic_per_episode,
            ) if future_papers else []
            total_selected_samples += len(selected_future)
            rows.append(
                {
                    "topic_id": topic.id,
                    "topic_name": topic.name,
                    "episode_id": episode.episode_id,
                    "cutoff_date": episode.cutoff_date,
                    "cutoff_month": episode.cutoff_month,
                    "future_start_date": episode.future_start_date,
                    "future_end_date": episode.future_end_date,
                    "train_paper_count": len(train_papers),
                    "future_paper_count": len(future_papers),
                    "selected_future_paper_ids": [paper.paper_id for paper in selected_future],
                    "selected_future_source_paths": [paper.source_path for paper in selected_future],
                }
            )

    return {
        "topics_config_path": str(context.topics_config_path),
        "input_dir": str(context.input_dir),
        "episodes": [episode.to_dict() for episode in FIXED_TOPIC_HINDSIGHT_EPISODES],
        "selection_policy": {
            "type": "deterministic_quantile_sampling",
            "per_topic_per_episode": per_topic_per_episode,
            "horizon_months": TOPIC_HINDSIGHT_HORIZON_MONTHS,
            "future_range": {
                "start_month": "2023-04",
                "end_month": "2024-09",
            },
            "selection_positions": "quantile_midpoints",
            "sort_order": "published_date_then_paper_id",
            "llm_free": True,
        },
        "total_selected_samples": total_selected_samples,
        "topic_episode_rows": rows,
    }


def summarize_topic_hindsight_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = list(manifest.get("topic_episode_rows", []))
    selected_windows = sum(1 for row in rows if row.get("selected_future_paper_ids"))
    empty_windows = sum(1 for row in rows if int(row.get("future_paper_count", 0) or 0) == 0)
    return {
        "topics_config_path": manifest.get("topics_config_path"),
        "input_dir": manifest.get("input_dir"),
        "topic_count": len({str(row.get("topic_id", "")) for row in rows}),
        "episode_count": len(list(manifest.get("episodes", []))),
        "topic_episode_row_count": len(rows),
        "empty_window_count": empty_windows,
        "selected_window_count": selected_windows,
        "total_selected_samples": int(manifest.get("total_selected_samples", 0) or 0),
        "selection_policy": dict(manifest.get("selection_policy", {})),
    }


def select_preview_targets(
    manifest: dict[str, Any],
    *,
    preview_count: int = 10,
) -> list[dict[str, Any]]:
    if preview_count <= 0:
        raise ValueError("preview_count must be positive")

    rows = sorted(
        list(manifest.get("topic_episode_rows", [])),
        key=lambda row: (str(row.get("episode_id", "")), str(row.get("topic_id", ""))),
    )
    max_selected = max(
        (len(list(row.get("selected_future_paper_ids", []))) for row in rows),
        default=0,
    )
    targets: list[dict[str, Any]] = []

    for position in range(max_selected):
        for row in rows:
            paper_ids = list(row.get("selected_future_paper_ids", []))
            source_paths = list(row.get("selected_future_source_paths", []))
            if position >= len(paper_ids):
                continue
            targets.append(
                {
                    "topic_id": row["topic_id"],
                    "topic_name": row["topic_name"],
                    "episode_id": row["episode_id"],
                    "cutoff_date": row["cutoff_date"],
                    "cutoff_month": row["cutoff_month"],
                    "future_start_date": row["future_start_date"],
                    "future_end_date": row["future_end_date"],
                    "future_paper_id": paper_ids[position],
                    "future_source_path": source_paths[position] if position < len(source_paths) else "",
                    "selection_rank": position + 1,
                }
            )
            if len(targets) >= preview_count:
                return targets
    return targets


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target

