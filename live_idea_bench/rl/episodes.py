from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from live_idea_bench.backtest import split_train_future_by_cutoff
from live_idea_bench.models import PaperRecord
from live_idea_bench.papers import month_start_date, month_to_index
from live_idea_bench.rl.config import EpisodeBuildConfig


@dataclass
class RLEpisode:
    cutoff_month: str
    cutoff_date: str
    future_end_month: str
    future_end_date: str
    train_paper_ids: list[str]
    future_paper_ids: list[str]
    split: str
    cache_artifact_path: str = ""


def _filter_by_month(
    papers: list[PaperRecord],
    *,
    start_month: str | None,
    end_month: str | None,
) -> list[PaperRecord]:
    start_idx = month_to_index(start_month) if start_month else None
    end_idx = month_to_index(end_month) if end_month else None
    filtered: list[PaperRecord] = []
    for paper in papers:
        idx = month_to_index(paper.month)
        if start_idx is not None and idx < start_idx:
            continue
        if end_idx is not None and idx > end_idx:
            continue
        filtered.append(paper)
    return filtered


def _split_labels(total: int, train_ratio: float, validation_ratio: float) -> list[str]:
    if total <= 0:
        return []

    train_count = int(total * train_ratio)
    validation_count = int(total * validation_ratio)
    test_count = total - train_count - validation_count

    if total >= 3:
        train_count = max(1, train_count)
        validation_count = max(1, validation_count)
        test_count = max(1, total - train_count - validation_count)
        while train_count + validation_count + test_count > total:
            if train_count > validation_count and train_count > 1:
                train_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                test_count -= 1
    else:
        train_count = max(1, total - 1)
        validation_count = 0
        test_count = total - train_count

    labels = (["train"] * train_count) + (["validation"] * validation_count) + (["test"] * test_count)
    if len(labels) < total:
        labels.extend(["test"] * (total - len(labels)))
    return labels[:total]


def build_rl_episodes(
    papers: list[PaperRecord],
    config: EpisodeBuildConfig,
    *,
    cache_root: Path | None = None,
) -> list[RLEpisode]:
    scoped_papers = _filter_by_month(
        papers,
        start_month=config.start_month,
        end_month=config.end_month,
    )
    if not scoped_papers:
        return []

    month_values = sorted(set(paper.month for paper in scoped_papers), key=month_to_index)
    candidate_rows: list[dict[str, Any]] = []
    for cutoff in month_values:
        cutoff_date = month_start_date(cutoff)
        train, future, future_end_month, future_end_date = split_train_future_by_cutoff(
            papers=scoped_papers,
            cutoff_month=cutoff,
            horizon_months=config.horizon_months,
            cutoff_date=cutoff_date,
        )
        if len(train) < config.min_train_papers or not future:
            continue
        candidate_rows.append(
            {
                "cutoff_month": cutoff,
                "cutoff_date": cutoff_date,
                "future_end_month": future_end_month,
                "future_end_date": future_end_date,
                "train_paper_ids": [paper.paper_id for paper in train],
                "future_paper_ids": [paper.paper_id for paper in future],
            }
        )

    labels = _split_labels(len(candidate_rows), config.train_ratio, config.validation_ratio)
    episodes: list[RLEpisode] = []
    for idx, row in enumerate(candidate_rows):
        split = labels[idx] if idx < len(labels) else "test"
        cache_artifact_path = ""
        if cache_root is not None:
            cache_artifact_path = str((cache_root / split / f"{row['cutoff_month']}.json").resolve())
        episodes.append(
            RLEpisode(
                cutoff_month=row["cutoff_month"],
                cutoff_date=row["cutoff_date"],
                future_end_month=row["future_end_month"],
                future_end_date=row["future_end_date"],
                train_paper_ids=row["train_paper_ids"],
                future_paper_ids=row["future_paper_ids"],
                split=split,
                cache_artifact_path=cache_artifact_path,
            )
        )
    return episodes


def serialize_episodes(episodes: list[RLEpisode]) -> list[dict[str, Any]]:
    return [asdict(episode) for episode in episodes]
