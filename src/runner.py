from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TimeWindow:
    index: int
    start: str  # YYYY-MM
    end: str  # YYYY-MM

    @property
    def window_id(self) -> str:
        return f"w{self.index:04d}_{self.start}_to_{self.end}".replace("-", "")


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _normalize_month(value: str) -> str:
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


def _parse_month(value: str) -> tuple[int, int]:
    normalized = _normalize_month(value)
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

    start_year, start_month = _parse_month(start)
    end_year, end_month = _parse_month(end)

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
    """Orchestrates an end-to-end backtest loop over time windows.

    For each window it writes artifacts under:
      <artifacts_dir>/windows/<window_id>/
        - metadata.json
        - stdout.log
        - stderr.log

    It also persists resumable run state at:
      <artifacts_dir>/state.json
    """

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
        self._write_manifest(windows)

        for window in windows:
            window_id = window.window_id
            existing_status = state["windows"].get(window_id, {}).get("status", "pending")

            if self.resume and existing_status == "completed":
                print(f"[skip] {window_id} already completed")
                continue
            if self.resume and not self.rerun_failed and existing_status == "failed":
                print(f"[skip] {window_id} previously failed (use --rerun-failed)")
                continue

            self._run_window(window=window, state=state)

            if self.stop_on_error:
                status = state["windows"].get(window_id, {}).get("status")
                if status == "failed":
                    print("[stop] terminating after first failure due to --stop-on-error")
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
            print(f"[dry-run] {window_id}: {command}")
        else:
            start_time = time.monotonic()
            with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_f:
                completed = subprocess.run(
                    shlex.split(command),
                    stdout=stdout_f,
                    stderr=stderr_f,
                    check=False,
                )

            duration = round(time.monotonic() - start_time, 3)
            return_code = completed.returncode
            status = "completed" if return_code == 0 else "failed"
            print(f"[{status}] {window_id} rc={return_code} duration={duration}s")

        finished_ts = _utc_now_iso()
        record = {
            **base_record,
            "status": status,
            "finished_at": finished_ts,
            "duration_seconds": duration,
            "return_code": return_code,
            "artifacts": {
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            },
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
            with self.state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            if "windows" not in state or not isinstance(state["windows"], dict):
                raise ValueError(f"Invalid state file format: {self.state_path}")
            return state

        return {
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "total_windows": len(windows),
            "windows": {},
        }

    def _write_manifest(self, windows: list[TimeWindow]) -> None:
        manifest = {
            "created_at": _utc_now_iso(),
            "command_template": self.command_template,
            "resume": self.resume,
            "rerun_failed": self.rerun_failed,
            "stop_on_error": self.stop_on_error,
            "dry_run": self.dry_run,
            "total_windows": len(windows),
            "windows": [
                {
                    "index": w.index,
                    "id": w.window_id,
                    "start": w.start,
                    "end": w.end,
                    "start_yymm": _to_yymm(w.start),
                    "end_yymm": _to_yymm(w.end),
                }
                for w in windows
            ],
        }
        self._save_json(self.manifest_path, manifest)

    @staticmethod
    def _save_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
