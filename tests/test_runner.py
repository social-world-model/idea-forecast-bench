from __future__ import annotations

import json
from pathlib import Path

from live_idea_bench.backtest import BacktestRunner, generate_windows


def test_generate_windows_basic() -> None:
    windows = generate_windows(
        start="2024-01", end="2024-05", window_months=3, step_months=1
    )

    assert [w.start for w in windows] == [
        "2024-01",
        "2024-02",
        "2024-03",
        "2024-04",
        "2024-05",
    ]
    assert [w.end for w in windows] == [
        "2024-03",
        "2024-04",
        "2024-05",
        "2024-05",
        "2024-05",
    ]


def test_generate_windows_accepts_yymm() -> None:
    windows = generate_windows(start="2401", end="2402", window_months=1, step_months=1)
    assert [w.start for w in windows] == ["2024-01", "2024-02"]
    assert [w.end for w in windows] == ["2024-01", "2024-02"]


def test_runner_resume_skips_completed(tmp_path: Path) -> None:
    windows = generate_windows(
        start="2024-01", end="2024-02", window_months=1, step_months=1
    )
    artifacts_dir = tmp_path / "backtest"

    runner = BacktestRunner(
        artifacts_dir=artifacts_dir,
        command_template="/bin/echo {window_start} {window_end}",
        resume=True,
        dry_run=False,
    )
    state1 = runner.run(windows)

    completed1 = sum(
        1 for v in state1["windows"].values() if v.get("status") == "completed"
    )
    assert completed1 == 2

    # Running again with resume should not create a second attempt.
    state2 = runner.run(windows)
    attempts = [window_data["attempt"] for window_data in state2["windows"].values()]
    assert attempts == [1, 1]

    state_path = artifacts_dir / "state.json"
    with state_path.open("r", encoding="utf-8") as f:
        state_payload = json.load(f)

    assert state_payload["total_windows"] == 2
    assert len(state_payload["windows"]) == 2


def test_runner_exposes_yymm_placeholders(tmp_path: Path) -> None:
    windows = generate_windows(
        start="2024-01", end="2024-01", window_months=1, step_months=1
    )
    artifacts_dir = tmp_path / "backtest"

    runner = BacktestRunner(
        artifacts_dir=artifacts_dir,
        command_template="/bin/echo {window_start_yymm} {window_end_yymm}",
        dry_run=True,
    )
    state = runner.run(windows)

    only_window = next(iter(state["windows"].values()))
    assert only_window["command"] == "/bin/echo 2401 2401"
