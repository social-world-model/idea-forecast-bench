"""Shared helpers for topic-based hindsight manifest and preview scripts."""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from idea_forecast_bench.backtest import split_train_future_by_cutoff
from idea_forecast_bench.config import TopicDefinition, load_topics
from idea_forecast_bench.models import PaperRecord
from idea_forecast_bench.papers import (
    date_to_ordinal,
    get_paper_published_date,
    month_to_index,
    normalize_month,
    parse_markdown_paper,
)
from idea_forecast_bench.topics import classify_papers_by_topic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOPICS_CONFIG_PATH = "config/topics_v2.yaml"
TOPIC_HINDSIGHT_START_MONTH = "2023-01"
TOPIC_HINDSIGHT_END_MONTH = "2024-09"
TOPIC_HINDSIGHT_HORIZON_MONTHS = 3
TOPIC_HINDSIGHT_DEFAULT_PROGRESS_EVERY = 2_000


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
    # E6 (cutoff 2024-06-30, future 2024-07..09) is deliberately absent. The
    # LAST episode's future window sets the earliest cutoff MDF can be evaluated
    # on without training-set contamination: with E6 present the labels cite
    # papers through 2024-09, so evaluation could only start at 2024-10. Ending
    # at E5 stops the labels at 2024-06, which lines the boundary up exactly
    # with gpt-4.1's 2024-06 pretraining cutoff -- the evaluation range is then
    # pinned by an external fact about the baseline model rather than by our own
    # method's training data, and it gains three cutoff months (12 instead of 9).
    #
    # Two properties depend on the quarterly spacing, so do not switch to a
    # monthly grid without handling both: (1) output/foresight_artifacts/indices
    # only holds indices at these quarter boundaries -- a monthly grid needs 8
    # more, built with examples/forecaster/build_indices.py; (2) quarterly future
    # windows do not overlap, whereas monthly ones overlap by 2/3 and
    # sample_future_papers_deterministically has no cross-episode dedup (its
    # seen_paper_ids is function-local), so the same paper would be selected as
    # the target for adjacent episodes.
)


def _resolve_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw.resolve()
    return (PROJECT_ROOT / raw).resolve()


def _resolve_loader_workers() -> int:
    raw = str(os.environ.get("TOPIC_HINDSIGHT_LOAD_WORKERS", "") or "").strip()
    if raw:
        return max(1, int(raw))
    cpu_count = os.cpu_count() or 1
    return max(4, min(32, cpu_count * 4))


def _resolve_progress_every() -> int:
    raw = str(os.environ.get("TOPIC_HINDSIGHT_PROGRESS_EVERY", "") or "").strip()
    if raw:
        return max(1, int(raw))
    return TOPIC_HINDSIGHT_DEFAULT_PROGRESS_EVERY


def _resolve_max_files() -> int | None:
    raw = str(os.environ.get("TOPIC_HINDSIGHT_MAX_FILES", "") or "").strip()
    if not raw:
        return None
    limit = int(raw)
    return max(1, limit)


def _parse_markdown_path(path: Path) -> PaperRecord | None:
    if path.name.lower() == "readme.md":
        return None
    return parse_markdown_paper(path)


def _paper_within_month_range(
    paper: PaperRecord,
    *,
    start_month: str,
    end_month: str,
) -> bool:
    paper_month_idx = month_to_index(paper.month)
    return month_to_index(start_month) <= paper_month_idx <= month_to_index(end_month)


def _print_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _infer_month_from_path_for_prefilter(path: Path) -> str | None:
    stem = path.stem.strip()
    if len(stem) >= 4 and stem[:4].isdigit():
        try:
            return normalize_month(stem[:4])
        except ValueError:
            pass

    for parent in path.parents:
        name = parent.name.strip()
        if re.match(r"^\d{4}-\d{2}$", name):
            return normalize_month(name)
    return None


def _path_in_month_range_or_unknown(
    path: Path,
    *,
    start_month: str,
    end_month: str,
) -> bool:
    inferred_month = _infer_month_from_path_for_prefilter(path)
    if inferred_month is None:
        return True
    inferred_idx = month_to_index(inferred_month)
    return month_to_index(start_month) <= inferred_idx <= month_to_index(end_month)


def _discover_markdown_files_with_progress(input_dir: Path) -> list[Path]:
    progress_every = _resolve_progress_every()
    max_files = _resolve_max_files()
    _print_progress(f"[topic-hindsight] discovering markdown files under {input_dir}")

    files: list[Path] = []
    discovered_count = 0
    pruned_dir_count = 0
    filtered_by_month_count = 0
    scanned_dirs = 0
    for dirpath, _dirnames, filenames in os.walk(input_dir):
        current_dir = Path(dirpath)
        kept_dirnames: list[str] = []
        for dirname in sorted(_dirnames):
            child_path = current_dir / dirname
            if _path_in_month_range_or_unknown(
                child_path,
                start_month=TOPIC_HINDSIGHT_START_MONTH,
                end_month=TOPIC_HINDSIGHT_END_MONTH,
            ):
                kept_dirnames.append(dirname)
            else:
                pruned_dir_count += 1
        _dirnames[:] = kept_dirnames
        _dirnames.sort()
        filenames.sort()
        scanned_dirs += 1
        for filename in filenames:
            if not filename.lower().endswith(".md"):
                continue
            if filename.lower() == "readme.md":
                continue
            path = Path(dirpath) / filename
            if not _path_in_month_range_or_unknown(
                path,
                start_month=TOPIC_HINDSIGHT_START_MONTH,
                end_month=TOPIC_HINDSIGHT_END_MONTH,
            ):
                filtered_by_month_count += 1
                continue
            discovered_count += 1
            files.append(path)
            if discovered_count % progress_every == 0:
                _print_progress(
                    f"[topic-hindsight] discovered {discovered_count} markdown files "
                    f"after scanning {scanned_dirs} directories "
                    f"(prefiltered_out={filtered_by_month_count}, pruned_dirs={pruned_dir_count})"
                )
            if max_files is not None and discovered_count >= max_files:
                _print_progress(
                    f"[topic-hindsight] early-stop smoke sample enabled: "
                    f"stopping discovery after {discovered_count} markdown files"
                )
                break
        if scanned_dirs % progress_every == 0 and discovered_count == 0:
            _print_progress(
                f"[topic-hindsight] scanned {scanned_dirs} directories, "
                f"no markdown files discovered yet (pruned_dirs={pruned_dir_count})"
            )
        if max_files is not None and discovered_count >= max_files:
            break

    files.sort()
    _print_progress(
        f"[topic-hindsight] discovery complete: found {len(files)} markdown files "
        f"across {scanned_dirs} directories "
        f"(prefiltered_out={filtered_by_month_count}, pruned_dirs={pruned_dir_count})"
    )
    return files


def _load_papers_with_progress(
    input_dir: Path,
    *,
    start_month: str,
    end_month: str,
) -> tuple[PaperRecord, ...]:
    files = _discover_markdown_files_with_progress(input_dir)
    total_files = len(files)
    if total_files == 0:
        _print_progress(f"[topic-hindsight] no markdown files found under {input_dir}")
        return ()

    workers = _resolve_loader_workers()
    progress_every = _resolve_progress_every()
    _print_progress(
        f"[topic-hindsight] parsing {total_files} markdown files under {input_dir} "
        f"with {workers} worker threads"
    )

    kept_records: list[PaperRecord] = []
    parsed_count = 0
    kept_count = 0
    skipped_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(_parse_markdown_path, path): path for path in files
        }
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                paper = future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed to parse markdown file: {path}") from exc

            parsed_count += 1
            if paper is not None and _paper_within_month_range(
                paper,
                start_month=start_month,
                end_month=end_month,
            ):
                kept_records.append(paper)
                kept_count += 1
            else:
                skipped_count += 1

            if parsed_count == total_files or parsed_count % progress_every == 0:
                _print_progress(
                    f"[topic-hindsight] parsed {parsed_count}/{total_files} files "
                    f"(kept={kept_count}, skipped={skipped_count})"
                )

    kept_records.sort(
        key=lambda paper: (
            date_to_ordinal(get_paper_published_date(paper)),
            paper.paper_id,
        )
    )
    _print_progress(
        f"[topic-hindsight] finished loading papers: kept={len(kept_records)} "
        f"within {start_month}..{end_month}"
    )
    return tuple(kept_records)


def load_topic_hindsight_context(
    input_dir: str | Path,
    *,
    topics_config_path: str = DEFAULT_TOPICS_CONFIG_PATH,
) -> TopicHindsightContext:
    resolved_input_dir = Path(input_dir).resolve()
    resolved_topics_config = _resolve_path(topics_config_path)
    papers = _load_papers_with_progress(
        resolved_input_dir,
        start_month=TOPIC_HINDSIGHT_START_MONTH,
        end_month=TOPIC_HINDSIGHT_END_MONTH,
    )
    topics = tuple(load_topics(str(resolved_topics_config)))
    _print_progress(
        f"[topic-hindsight] classifying {len(papers)} papers across {len(topics)} topics "
        f"from {resolved_topics_config}"
    )
    grouped_raw = classify_papers_by_topic(list(papers), list(topics))
    grouped = OrderedDict(
        (topic.id, tuple(grouped_raw.get(topic.id, []))) for topic in topics
    )
    _print_progress("[topic-hindsight] topic classification complete")
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
    total_windows = len(context.topics) * len(FIXED_TOPIC_HINDSIGHT_EPISODES)
    window_index = 0
    _print_progress(
        f"[topic-hindsight] building manifest rows for {len(context.topics)} topics "
        f"x {len(FIXED_TOPIC_HINDSIGHT_EPISODES)} episodes = {total_windows} windows"
    )

    for topic in context.topics:
        topic_papers = list(context.grouped_papers.get(topic.id, ()))
        for episode in FIXED_TOPIC_HINDSIGHT_EPISODES:
            window_index += 1
            train_papers, future_papers, _future_end_month, _future_end_date = (
                split_train_future_by_cutoff(
                    papers=topic_papers,
                    cutoff_date=episode.cutoff_date,
                    horizon_months=TOPIC_HINDSIGHT_HORIZON_MONTHS,
                )
            )
            selected_future = (
                sample_future_papers_deterministically(
                    future_papers,
                    limit=per_topic_per_episode,
                )
                if future_papers
                else []
            )
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
                    "selected_future_paper_ids": [
                        paper.paper_id for paper in selected_future
                    ],
                    "selected_future_source_paths": [
                        paper.source_path for paper in selected_future
                    ],
                }
            )
            if window_index == total_windows or window_index % 50 == 0:
                _print_progress(
                    f"[topic-hindsight] built {window_index}/{total_windows} manifest windows "
                    f"(selected_samples={total_selected_samples})"
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
    empty_windows = sum(
        1 for row in rows if int(row.get("future_paper_count", 0) or 0) == 0
    )
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
        manifest.get("topic_episode_rows", []),
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
                    "future_source_path": source_paths[position]
                    if position < len(source_paths)
                    else "",
                    "selection_rank": position + 1,
                }
            )
            if len(targets) >= preview_count:
                return targets
    return targets


def expand_all_targets(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand ALL selected papers from the manifest into target dicts.

    Same target dict format as :func:`select_preview_targets` but without any
    count limit — every ``selected_future_paper_id`` in every
    ``topic_episode_row`` is included.
    """
    rows = sorted(
        manifest.get("topic_episode_rows", []),
        key=lambda row: (str(row.get("episode_id", "")), str(row.get("topic_id", ""))),
    )
    targets: list[dict[str, Any]] = []
    for row in rows:
        paper_ids = list(row.get("selected_future_paper_ids", []))
        source_paths = list(row.get("selected_future_source_paths", []))
        for position, paper_id in enumerate(paper_ids):
            targets.append(
                {
                    "topic_id": row["topic_id"],
                    "topic_name": row["topic_name"],
                    "episode_id": row["episode_id"],
                    "cutoff_date": row["cutoff_date"],
                    "cutoff_month": row["cutoff_month"],
                    "future_start_date": row["future_start_date"],
                    "future_end_date": row["future_end_date"],
                    "future_paper_id": paper_id,
                    "future_source_path": source_paths[position]
                    if position < len(source_paths)
                    else "",
                    "selection_rank": position + 1,
                }
            )
    return targets


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target
