from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, replace as _dc_replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from live_idea_bench.models import BacktestWindowResult, EvaluationResult, IdeaPrediction, PaperRecord
from live_idea_bench.papers import (
    add_months,
    date_to_ordinal,
    get_paper_published_date,
    load_papers_from_markdown,
    month_end_date,
    month_start_date,
    month_to_index,
    normalize_date,
    normalize_month,
)
from live_idea_bench.popularity import enrich_papers_with_popularity
from live_idea_bench.similarity import evaluate_predictions, score_prediction_list
from live_idea_bench.strategy.base import IdeaStrategy


@dataclass
class BacktestConfig:
    top_k: int = 5
    horizon_months: int = 3
    min_train_papers: int = 6
    start_month: Optional[str] = None
    end_month: Optional[str] = None
    similarity_config: str = "similarity.yaml"
    popularity_cache_path: Optional[str] = None  # Path to popularity cache JSON
    candidate_limit: Optional[int] = None  # max future papers per prediction for LLM eval


def _filter_by_month(
    papers: List[PaperRecord],
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
) -> List[PaperRecord]:
    start_idx = month_to_index(start_month) if start_month else None
    end_idx = month_to_index(end_month) if end_month else None
    output: List[PaperRecord] = []
    for paper in papers:
        idx = month_to_index(paper.month)
        if start_idx is not None and idx < start_idx:
            continue
        if end_idx is not None and idx > end_idx:
            continue
        output.append(paper)
    return output


def _resolve_cutoff(
    *,
    cutoff_month: Optional[str],
    cutoff_date: Optional[str],
) -> Tuple[str, str]:
    if cutoff_date:
        resolved_cutoff_date = normalize_date(cutoff_date)
        resolved_cutoff_month = normalize_month(resolved_cutoff_date)
        return resolved_cutoff_month, resolved_cutoff_date
    if cutoff_month:
        resolved_cutoff_month = normalize_month(cutoff_month)
        return resolved_cutoff_month, month_start_date(resolved_cutoff_month)
    raise ValueError("cutoff_month or cutoff_date is required")


def generate_at_cutoff(
    papers: List[PaperRecord],
    strategy: IdeaStrategy,
    cutoff_month: Optional[str] = None,
    top_k: int = 5,
    cutoff_date: Optional[str] = None,
) -> List[IdeaPrediction]:
    resolved_month, resolved_date = _resolve_cutoff(
        cutoff_month=cutoff_month,
        cutoff_date=cutoff_date,
    )
    cutoff_ord = date_to_ordinal(resolved_date)
    train = [
        paper
        for paper in papers
        if date_to_ordinal(get_paper_published_date(paper)) <= cutoff_ord
    ]
    return strategy.generate(train_papers=train, cutoff_month=resolved_month, top_k=top_k)


def split_train_future_by_cutoff(
    papers: List[PaperRecord],
    cutoff_month: Optional[str] = None,
    horizon_months: int = 3,
    cutoff_date: Optional[str] = None,
) -> Tuple[List[PaperRecord], List[PaperRecord], str, str]:
    resolved_cutoff_month, resolved_cutoff_date = _resolve_cutoff(
        cutoff_month=cutoff_month,
        cutoff_date=cutoff_date,
    )
    future_end_month = add_months(resolved_cutoff_month, horizon_months)
    future_end_date = month_end_date(future_end_month)
    cutoff_ord = date_to_ordinal(resolved_cutoff_date)
    future_end_ord = date_to_ordinal(future_end_date)

    train = [
        paper
        for paper in papers
        if date_to_ordinal(get_paper_published_date(paper)) <= cutoff_ord
    ]
    future = [
        paper
        for paper in papers
        if cutoff_ord < date_to_ordinal(get_paper_published_date(paper)) <= future_end_ord
    ]
    return train, future, future_end_month, future_end_date


def evaluate_at_cutoff(
    papers: List[PaperRecord],
    strategy: IdeaStrategy,
    cutoff_month: Optional[str] = None,
    top_k: int = 5,
    horizon_months: int = 3,
    cutoff_date: Optional[str] = None,
    similarity_config: str = "similarity.yaml",
    model_name: str | None = None,
) -> EvaluationResult:
    resolved_cutoff_month, resolved_cutoff_date = _resolve_cutoff(
        cutoff_month=cutoff_month,
        cutoff_date=cutoff_date,
    )
    train, future, _future_end_month, future_end_date = split_train_future_by_cutoff(
        papers=papers,
        cutoff_month=resolved_cutoff_month,
        horizon_months=horizon_months,
        cutoff_date=cutoff_date,
    )
    predictions = strategy.generate(
        train_papers=train,
        cutoff_month=resolved_cutoff_month,
        top_k=top_k,
    )
    return evaluate_predictions(
        predictions=predictions,
        train_papers=train,
        future_papers=future,
        k=top_k,
        similarity_config_path=similarity_config,
        model_name=model_name,
        cutoff_date=resolved_cutoff_date,
        future_end_date=future_end_date,
    )


def _summarize_windows(windows: list[BacktestWindowResult]) -> dict[str, float]:
    if not windows:
        return {
            "windows": 0,
            "avg_hit_at_k": 0.0,
            "avg_recall_at_k": 0.0,
            "avg_precision_at_k": 0.0,
            "avg_mrr": 0.0,
            "avg_novelty": 0.0,
            "avg_diversity": 0.0,
            "avg_lead_time": 0.0,
            "avg_duplicate_rate": 0.0,
            "avg_weighted_hit_at_k": 0.0,
            "avg_weighted_precision_at_k": 0.0,
            "avg_weighted_mrr": 0.0,
            "avg_popularity_recall_at_k": 0.0,
        }

    def _avg(name: str) -> float:
        return round(sum(float(getattr(window.evaluation, name)) for window in windows) / len(windows), 4)

    return {
        "windows": len(windows),
        "avg_hit_at_k": _avg("hit_at_k"),
        "avg_recall_at_k": _avg("recall_at_k"),
        "avg_precision_at_k": _avg("precision_at_k"),
        "avg_mrr": _avg("mrr"),
        "avg_novelty": _avg("novelty"),
        "avg_diversity": _avg("diversity"),
        "avg_lead_time": _avg("lead_time"),
        "avg_duplicate_rate": _avg("duplicate_rate"),
        "avg_weighted_hit_at_k": _avg("weighted_hit_at_k"),
        "avg_weighted_precision_at_k": _avg("weighted_precision_at_k"),
        "avg_weighted_mrr": _avg("weighted_mrr"),
        "avg_popularity_recall_at_k": _avg("popularity_recall_at_k"),
    }


def run_backtest(
    papers: List[PaperRecord],
    strategy: IdeaStrategy,
    config: BacktestConfig,
    *,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
) -> Dict[str, object]:
    scoped_papers = _filter_by_month(
        papers,
        start_month=config.start_month,
        end_month=config.end_month,
    )
    if not scoped_papers:
        return {"summary": _summarize_windows([]), "windows": []}

    month_values = sorted(set(paper.month for paper in scoped_papers), key=month_to_index)
    last_allowed_cutoff = add_months(month_values[-1], -config.horizon_months)
    max_cutoff_idx = month_to_index(last_allowed_cutoff)

    eligible_cutoffs = [c for c in month_values if month_to_index(c) <= max_cutoff_idx]
    window_results: List[BacktestWindowResult] = []
    for cutoff in tqdm(eligible_cutoffs, desc="windows", unit="win", leave=False):
        cutoff_date = month_start_date(cutoff)
        train, future, future_end, future_end_date = split_train_future_by_cutoff(
            papers=scoped_papers,
            cutoff_month=cutoff,
            horizon_months=config.horizon_months,
            cutoff_date=cutoff_date,
        )
        if len(train) < config.min_train_papers or not future:
            continue

        predictions = strategy.generate(train_papers=train, cutoff_month=cutoff, top_k=config.top_k)
        if len(predictions) < config.top_k:
            import sys as _sys
            print(
                f"[backtest WARNING] cutoff={cutoff}: got {len(predictions)}/{config.top_k} predictions "
                f"(strategy={strategy.__class__.__name__})",
                file=_sys.stderr, flush=True,
            )
        popularity_weights = None
        if config.popularity_cache_path:
            popularity_weights = enrich_papers_with_popularity(
                future, cache_path=Path(config.popularity_cache_path)
            )
            future = [
                _dc_replace(paper, popularity_score=popularity_weights.get(paper.paper_id, 0.0))
                for paper in future
            ]
        scored = score_prediction_list(
            predictions=predictions,
            train_papers=train,
            future_papers=future,
            k=config.top_k,
            similarity_config_path=config.similarity_config,
            model_name=model_name,
            cutoff_date=cutoff_date,
            future_end_date=future_end_date,
            popularity_weights=popularity_weights,
            reasoning_effort=reasoning_effort,
            candidate_limit=config.candidate_limit,
        )

        window_results.append(
            BacktestWindowResult(
                cutoff_month=cutoff,
                cutoff_date=cutoff_date,
                future_end_month=future_end,
                future_end_date=future_end_date,
                train_papers=len(train),
                future_papers=len(future),
                predictions=predictions,
                evaluation=scored.evaluation,
                matches=scored.matches,
            )
        )

    return {
        "summary": _summarize_windows(window_results),
        "windows": [asdict(window) for window in window_results],
    }


generate = generate_at_cutoff
evaluate = evaluate_at_cutoff
backtest = run_backtest


@dataclass(frozen=True)
class TimeWindow:
    index: int
    start: str
    end: str

    @property
    def window_id(self) -> str:
        return f"w{self.index:04d}_{self.start}_to_{self.end}".replace("-", "")


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _normalize_window_month(value: str) -> str:
    raw = value.strip()
    if len(raw) == 4 and raw.isdigit():
        year = 2000 + int(raw[:2])
        month = int(raw[2:])
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month '{value}'. Month must be 01..12.")
        return f"{year:04d}-{month:02d}"
    if len(raw) == 7 and raw[4] == "-":
        year_str, month_str = raw.split("-", maxsplit=1)
        if not (year_str.isdigit() and month_str.isdigit()):
            raise ValueError(f"Invalid month '{value}'. Expected YYYY-MM or YYMM.")
        year = int(year_str)
        month = int(month_str)
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month '{value}'. Month must be 01..12.")
        return f"{year:04d}-{month:02d}"
    raise ValueError(f"Invalid month '{value}'. Expected YYYY-MM or YYMM.")


def _parse_window_month(value: str) -> tuple[int, int]:
    normalized = _normalize_window_month(value)
    year_str, month_str = normalized.split("-", maxsplit=1)
    return int(year_str), int(month_str)


def _month_to_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def _index_to_month(value: int) -> str:
    year = value // 12
    month = (value % 12) + 1
    return f"{year:04d}-{month:02d}"


def _to_yymm(month: str) -> str:
    year_str, month_str = month.split("-", maxsplit=1)
    return f"{int(year_str) % 100:02d}{month_str}"


def generate_windows(
    start: str,
    end: str,
    window_months: int,
    step_months: int,
) -> list[TimeWindow]:
    if window_months <= 0:
        raise ValueError("window_months must be > 0")
    if step_months <= 0:
        raise ValueError("step_months must be > 0")

    start_year, start_month = _parse_window_month(start)
    end_year, end_month = _parse_window_month(end)
    start_idx = _month_to_index(start_year, start_month)
    end_idx = _month_to_index(end_year, end_month)

    if start_idx > end_idx:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    windows: list[TimeWindow] = []
    cursor = start_idx
    i = 0
    while cursor <= end_idx:
        window_end_idx = min(cursor + window_months - 1, end_idx)
        windows.append(
            TimeWindow(
                index=i,
                start=_index_to_month(cursor),
                end=_index_to_month(window_end_idx),
            )
        )
        i += 1
        cursor += step_months
    return windows


class BacktestRunner:
    def __init__(
        self,
        artifacts_dir: Path,
        command_template: str,
        resume: bool = True,
        rerun_failed: bool = False,
        stop_on_error: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.command_template = command_template
        self.resume = resume
        self.rerun_failed = rerun_failed
        self.stop_on_error = stop_on_error
        self.dry_run = dry_run
        self.windows_dir = self.artifacts_dir / "windows"
        self.state_path = self.artifacts_dir / "state.json"
        self.manifest_path = self.artifacts_dir / "manifest.json"

    def run(self, windows: list[TimeWindow]) -> dict[str, Any]:
        self._ensure_dirs()
        state = self._load_or_init_state(windows)
        state["total_windows"] = len(windows)
        self._write_manifest(windows)

        for window in windows:
            window_id = window.window_id
            existing_status = state["windows"].get(window_id, {}).get("status", "pending")
            if self.resume and existing_status == "completed":
                continue
            if self.resume and not self.rerun_failed and existing_status == "failed":
                continue

            self._run_window(window=window, state=state)
            if self.stop_on_error and state["windows"].get(window_id, {}).get("status") == "failed":
                break

        state["updated_at"] = _utc_now_iso()
        self._save_json(self.state_path, state)
        return state

    def _run_window(self, window: TimeWindow, state: dict[str, Any]) -> None:
        window_id = window.window_id
        window_dir = self.windows_dir / window_id
        window_dir.mkdir(parents=True, exist_ok=True)
        start_ts = _utc_now_iso()
        attempt = state["windows"].get(window_id, {}).get("attempt", 0) + 1

        context = {
            "window_id": window_id,
            "window_index": window.index,
            "window_start": window.start,
            "window_end": window.end,
            "window_start_yymm": _to_yymm(window.start),
            "window_end_yymm": _to_yymm(window.end),
            "artifacts_dir": str(self.artifacts_dir),
            "window_dir": str(window_dir),
        }
        command = self.command_template.format(**context)
        base_record = {
            "status": "running",
            "attempt": attempt,
            "window": {
                "index": window.index,
                "start": window.start,
                "end": window.end,
                "id": window_id,
            },
            "command": command,
            "started_at": start_ts,
        }
        state["windows"][window_id] = base_record
        state["updated_at"] = _utc_now_iso()
        self._save_json(self.state_path, state)

        stdout_path = window_dir / "stdout.log"
        stderr_path = window_dir / "stderr.log"
        if self.dry_run:
            duration = 0.0
            return_code = 0
            status = "completed"
            stdout_path.write_text("[dry-run] command not executed\n" + command + "\n", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
        else:
            start_time = time.monotonic()
            with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
                completed = subprocess.run(shlex.split(command), stdout=stdout_f, stderr=stderr_f, check=False)
            duration = round(time.monotonic() - start_time, 3)
            return_code = completed.returncode
            status = "completed" if return_code == 0 else "failed"

        record = {
            **base_record,
            "status": status,
            "finished_at": _utc_now_iso(),
            "duration_seconds": duration,
            "return_code": return_code,
            "artifacts": {"stdout": str(stdout_path), "stderr": str(stderr_path)},
        }
        state["windows"][window_id] = record
        state["updated_at"] = _utc_now_iso()
        self._save_json(window_dir / "metadata.json", record)
        self._save_json(self.state_path, state)

    def _ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.windows_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_init_state(self, windows: list[TimeWindow]) -> dict[str, Any]:
        if self.resume and self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict) and isinstance(loaded.get("windows"), dict):
                return loaded
        return {
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "total_windows": len(windows),
            "windows": {window.window_id: {"status": "pending"} for window in windows},
        }

    def _write_manifest(self, windows: list[TimeWindow]) -> None:
        payload = {
            "total_windows": len(windows),
            "windows": [asdict(window) | {"window_id": window.window_id} for window in windows],
        }
        self._save_json(self.manifest_path, payload)

    @staticmethod
    def _save_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
