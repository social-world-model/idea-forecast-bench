from __future__ import annotations

import json
import os
import threading
import traceback
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from backend import config


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class RunRecord:
    run_id: str
    status: str
    keywords: List[str]
    n: int
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    output_path: Optional[str] = None
    ideas_count: int = 0
    report: Optional[Dict[str, Any]] = None


class RunService:
    def __init__(self, project_root: str, idea_generator: Any = None) -> None:
        self.project_root = project_root
        self.data_dir = os.path.join(project_root, "data")
        self.runs_dir = os.path.join(self.data_dir, "runs")
        self.runs_index_file = os.path.join(self.data_dir, "runs_index.json")
        self._lock = threading.Lock()
        self._runs: Dict[str, RunRecord] = {}
        self._idea_generator = idea_generator

        os.makedirs(self.runs_dir, exist_ok=True)
        self._load_runs()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _load_runs(self) -> None:
        if not os.path.exists(self.runs_index_file):
            return

        try:
            with open(self.runs_index_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)

            runs = payload.get("runs", [])
            for item in runs:
                record = RunRecord(**item)
                self._runs[record.run_id] = record
        except Exception:
            # Keep service available even when persisted state is corrupted.
            self._runs = {}

    def _persist_runs(self) -> None:
        runs = [asdict(run) for run in self._runs.values()]
        runs.sort(key=lambda item: item["created_at"], reverse=True)
        payload = {"runs": runs, "updated_at": self._now_iso()}
        with open(self.runs_index_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _build_report(self, ideas: List[Dict[str, Any]], run: RunRecord) -> Dict[str, Any]:
        score_values = [self._safe_float(idea.get("Score")) for idea in ideas]
        novelty_values = [self._safe_float(idea.get("Novelty")) for idea in ideas]
        feasibility_values = [self._safe_float(idea.get("Feasibility")) for idea in ideas]

        clean_scores = [v for v in score_values if v is not None]
        clean_novelty = [v for v in novelty_values if v is not None]
        clean_feasibility = [v for v in feasibility_values if v is not None]

        def avg(values: List[float]) -> float:
            if not values:
                return 0.0
            return round(sum(values) / len(values), 3)

        return {
            "run_id": run.run_id,
            "keywords": run.keywords,
            "n": run.n,
            "ideas_count": len(ideas),
            "average_score": avg(clean_scores),
            "average_novelty": avg(clean_novelty),
            "average_feasibility": avg(clean_feasibility),
            "generated_at": self._now_iso(),
            "model": config.MODEL,
        }

    def start_run(self, keywords: Optional[List[str]] = None, n: Optional[int] = None) -> RunRecord:
        if keywords is None:
            keywords = list(config.KEYWORDS)
        if not isinstance(keywords, list) or not keywords:
            raise ValueError("keywords must be a non-empty list")
        if n is None:
            n = config.NUM_PAPERS_TO_FETCH
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        run_id = str(uuid.uuid4())
        now = self._now_iso()
        record = RunRecord(
            run_id=run_id,
            status=RunStatus.PENDING.value,
            keywords=[str(item) for item in keywords],
            n=n,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            self._runs[run_id] = record
            self._persist_runs()

        thread = threading.Thread(target=self._execute_run, args=(run_id,), daemon=True)
        thread.start()
        return record

    def _execute_run(self, run_id: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.status = RunStatus.RUNNING.value
            run.started_at = self._now_iso()
            run.updated_at = run.started_at
            self._persist_runs()

        try:
            if self._idea_generator is None:
                from backend.idea_generator import generate_ideas

                idea_generator = generate_ideas
            else:
                idea_generator = self._idea_generator

            ideas = idea_generator(run.keywords, run.n)
            output_path = os.path.join(self.runs_dir, f"{run_id}.json")
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(ideas, fh, indent=2)

            finished_at = self._now_iso()
            duration = (
                datetime.fromisoformat(finished_at) - datetime.fromisoformat(run.started_at or finished_at)
            ).total_seconds()

            report = self._build_report(ideas, run)

            with self._lock:
                persisted = self._runs.get(run_id)
                if persisted is None:
                    return
                persisted.status = RunStatus.SUCCESS.value
                persisted.finished_at = finished_at
                persisted.updated_at = finished_at
                persisted.duration_seconds = round(duration, 3)
                persisted.output_path = os.path.relpath(output_path, self.project_root)
                persisted.ideas_count = len(ideas)
                persisted.report = report
                self._persist_runs()
        except Exception as exc:
            finished_at = self._now_iso()
            error = f"{exc}\n{traceback.format_exc(limit=8)}"
            with self._lock:
                failed = self._runs.get(run_id)
                if failed is None:
                    return
                failed.status = RunStatus.FAILED.value
                failed.finished_at = finished_at
                failed.updated_at = finished_at
                failed.error = error
                self._persist_runs()

    def get_run(self, run_id: str, include_ideas: bool = False) -> Optional[Dict[str, Any]]:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            payload: Dict[str, Any] = asdict(run)

        if include_ideas and run.output_path:
            abs_path = os.path.join(self.project_root, run.output_path)
            if os.path.exists(abs_path):
                with open(abs_path, "r", encoding="utf-8") as fh:
                    payload["ideas"] = json.load(fh)
        return payload

    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            runs = [asdict(item) for item in self._runs.values()]
        runs.sort(key=lambda item: item["created_at"], reverse=True)
        return runs[:limit]

    def build_global_report(self) -> Dict[str, Any]:
        runs = self.list_runs(limit=500)
        total = len(runs)
        successful = [run for run in runs if run.get("status") == RunStatus.SUCCESS.value]
        failed = [run for run in runs if run.get("status") == RunStatus.FAILED.value]
        running = [run for run in runs if run.get("status") == RunStatus.RUNNING.value]

        durations = [float(run["duration_seconds"]) for run in successful if run.get("duration_seconds") is not None]
        avg_duration = round(sum(durations) / len(durations), 3) if durations else 0.0
        avg_ideas = round(
            sum(int(run.get("ideas_count", 0)) for run in successful) / len(successful),
            3,
        ) if successful else 0.0

        keyword_frequency: Dict[str, int] = {}
        for run in runs:
            for keyword in run.get("keywords", []):
                keyword_frequency[keyword] = keyword_frequency.get(keyword, 0) + 1

        score_trend = []
        for run in successful[:20]:
            report = run.get("report") or {}
            score_trend.append(
                {
                    "run_id": run["run_id"],
                    "timestamp": run["created_at"],
                    "average_score": report.get("average_score", 0.0),
                    "ideas_count": run.get("ideas_count", 0),
                }
            )

        top_runs = sorted(
            successful,
            key=lambda item: float((item.get("report") or {}).get("average_score", 0.0)),
            reverse=True,
        )[:2]
        comparison = [
            {
                "run_id": run["run_id"],
                "average_score": (run.get("report") or {}).get("average_score", 0.0),
                "average_novelty": (run.get("report") or {}).get("average_novelty", 0.0),
                "average_feasibility": (run.get("report") or {}).get("average_feasibility", 0.0),
                "ideas_count": run.get("ideas_count", 0),
                "keywords": run.get("keywords", []),
            }
            for run in top_runs
        ]

        return {
            "summary": {
                "total_runs": total,
                "running_runs": len(running),
                "successful_runs": len(successful),
                "failed_runs": len(failed),
                "success_rate": round((len(successful) / total), 3) if total else 0.0,
                "average_duration_seconds": avg_duration,
                "average_ideas_per_run": avg_ideas,
            },
            "keyword_frequency": keyword_frequency,
            "score_trend": score_trend,
            "comparison": comparison,
            "generated_at": self._now_iso(),
        }
