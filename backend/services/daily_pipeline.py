from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import strategy_store
from live_idea_bench.daily import (
    compute_leaderboard_score,
    daily_cutoff_date,
    evaluate_previous_generation,
)
from live_idea_bench.ingest import ingest_latest_arxiv_papers
from live_idea_bench.papers import load_papers_from_markdown, month_to_index


class PipelineAlreadyRunningError(RuntimeError):
    pass


def _lock_ttl_seconds() -> int:
    raw = os.environ.get("LIVE_IDEA_PIPELINE_LOCK_TTL_SECONDS")
    if raw is None:
        return 6 * 60 * 60
    try:
        return max(60, int(raw))
    except ValueError:
        return 6 * 60 * 60


class _FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    @staticmethod
    def _process_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _read_metadata(self) -> tuple[int | None, datetime | None]:
        try:
            payload = self.path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None, None
        except OSError:
            return None, None

        pid: int | None = None
        created_at: datetime | None = None
        for token in payload.split():
            if token.startswith("pid="):
                try:
                    pid = int(token.split("=", 1)[1])
                except ValueError:
                    pid = None
            if token.startswith("created_at="):
                raw = token.split("=", 1)[1]
                try:
                    created_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except ValueError:
                    created_at = None
        return pid, created_at

    def _clear_if_stale(self) -> bool:
        pid, created_at = self._read_metadata()
        now = datetime.now(timezone.utc)
        expired = created_at is None or (now - created_at.astimezone(timezone.utc)).total_seconds() >= _lock_ttl_seconds()
        process_dead = pid is None or not self._process_alive(pid)
        if not expired and not process_dead:
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return True

    def __enter__(self) -> _FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError as exc:
                if not self._clear_if_stale():
                    raise PipelineAlreadyRunningError(f"Lock exists: {self.path}") from exc
        else:
            raise PipelineAlreadyRunningError(f"Lock exists: {self.path}")

        payload = f"pid={os.getpid()} created_at={datetime.now(timezone.utc).isoformat()}\n"
        os.write(self.fd, payload.encode("utf-8"))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _latest_month_from_data_dir(data_dir: Path) -> str | None:
    papers = load_papers_from_markdown(data_dir)
    if not papers:
        return None
    return papers[-1].month


def _fallback_backtest_score(strategy: dict[str, Any]) -> float | None:
    summary = strategy_store._aggregate_topic_backtest_summary(strategy.get("topic_runs") or [])
    if summary is None:
        summary = (strategy.get("backtest_result") or {}).get("summary") or {}
    score = summary.get("avg_hit_at_k")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _ensure_strategy_end_month(strategy: dict[str, Any], latest_month: str | None) -> dict[str, Any]:
    if not latest_month:
        return strategy

    current = dict(strategy.get("config") or {})
    current_end = str(current.get("end_month") or "").strip()
    should_update = False
    if not current_end:
        should_update = True
    else:
        try:
            should_update = month_to_index(current_end) < month_to_index(latest_month)
        except Exception:
            should_update = True

    if not should_update:
        return strategy

    current["end_month"] = latest_month
    updated = strategy_store.update_strategy(strategy["id"], {"config": current})
    return updated or strategy


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _extract_new_paper_ids(ingest_result: dict[str, Any]) -> set[str]:
    raw = ingest_result.get("new_papers") or []
    ids: set[str] = set()
    if not isinstance(raw, list):
        return ids
    for item in raw:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "").strip()
        if paper_id:
            ids.add(paper_id)
    return ids


def run_daily_pipeline(
    *,
    now: datetime | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    utc_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cutoff_date = daily_cutoff_date(utc_now)
    project_root = strategy_store.PROJECT_ROOT
    run_dir = project_root / "data" / "daily_runs"
    lock_path = run_dir / "pipeline.lock"

    with _FileLock(lock_path):
        ingest_result = ingest_latest_arxiv_papers(data_dir=data_dir, now=utc_now)
        raw_data_dir = str(ingest_result.get("data_dir") or "").strip()
        resolved_data_dir = Path(raw_data_dir).expanduser() if raw_data_dir else strategy_store.DEFAULT_DATA_DIR
        latest_month = _latest_month_from_data_dir(resolved_data_dir)
        new_paper_ids = _extract_new_paper_ids(ingest_result)

        strategy_results: list[dict[str, Any]] = []
        for strategy in strategy_store.list_strategies():
            strategy_id = strategy.get("id")
            if not strategy_id:
                continue

            current = strategy_store.get_strategy(strategy_id) or strategy
            current = _ensure_strategy_end_month(current, latest_month)
            papers = strategy_store.load_papers_for_strategy(current)
            daily_eval = evaluate_previous_generation(
                current,
                papers=papers,
                new_paper_ids=new_paper_ids,
                evaluated_at=utc_now,
            )

            updates: dict[str, Any] = {"last_daily_run_at": _iso(utc_now)}
            if latest_month:
                updates["last_generation_cutoff_month"] = latest_month
            if daily_eval is not None:
                updates["daily_evaluation"] = daily_eval
                updates["leaderboard_score"] = compute_leaderboard_score(daily_eval)
            elif current.get("leaderboard_score") is None:
                updates["leaderboard_score"] = _fallback_backtest_score(current)
            strategy_store.update_strategy(strategy_id, updates)

            generation_status = "skipped_no_data"
            generation_error = None
            if latest_month:
                strategy_store.run_generation_sync(strategy_id, cutoff_date=cutoff_date)
                after_generation = strategy_store.get_strategy(strategy_id) or {}
                generation_status = str(after_generation.get("generation_status") or "unknown")
                generation_error = after_generation.get("generation_error")

            final = strategy_store.get_strategy(strategy_id) or {}
            strategy_results.append(
                {
                    "id": strategy_id,
                    "name": final.get("name"),
                    "leaderboard_score": final.get("leaderboard_score"),
                    "generation_status": generation_status,
                    "generation_error": generation_error,
                    "daily_evaluation": final.get("daily_evaluation"),
                }
            )

        report = {
            "ran_at": _iso(utc_now),
            "latest_month": latest_month,
            "ingest": ingest_result,
            "strategies_processed": len(strategy_results),
            "strategies": strategy_results,
        }
        report_path = run_dir / f"{utc_now.date().isoformat()}.json"
        latest_path = run_dir / "latest.json"
        _write_json(report_path, report)
        _write_json(latest_path, report)
        return report
