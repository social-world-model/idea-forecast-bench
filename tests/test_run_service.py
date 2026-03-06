from __future__ import annotations

import time
from pathlib import Path

from backend.services.run_service import RunService, RunStatus


def _wait_for_status(service: RunService, run_id: str, expected: str, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = service.get_run(run_id)
        if run and run.get("status") == expected:
            return
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not reach status {expected}")


def test_run_service_start_and_finish(tmp_path: Path) -> None:
    def fake_generate_ideas(keywords, n):
        return [
            {"Title": "A", "Score": 8.0, "Novelty": 9.0, "Feasibility": 7.0},
            {"Title": "B", "Score": 7.0, "Novelty": 8.0, "Feasibility": 8.0},
        ]

    service = RunService(str(tmp_path), idea_generator=fake_generate_ideas)
    run = service.start_run(keywords=["llm"], n=2)

    _wait_for_status(service, run.run_id, RunStatus.SUCCESS.value)
    details = service.get_run(run.run_id, include_ideas=True)

    assert details is not None
    assert details["status"] == RunStatus.SUCCESS.value
    assert details["ideas_count"] == 2
    assert details["report"]["average_score"] == 7.5
    assert len(details["ideas"]) == 2


def test_run_service_report(tmp_path: Path) -> None:
    def fake_generate_ideas(keywords, n):
        return [{"Title": "A", "Score": 9.0, "Novelty": 8.0, "Feasibility": 7.0}]

    service = RunService(str(tmp_path), idea_generator=fake_generate_ideas)
    run = service.start_run(keywords=["agents"], n=1)

    _wait_for_status(service, run.run_id, RunStatus.SUCCESS.value)

    report = service.build_global_report()
    assert report["summary"]["total_runs"] == 1
    assert report["summary"]["successful_runs"] == 1
    assert report["summary"]["failed_runs"] == 0
    assert report["keyword_frequency"]["agents"] == 1
    assert len(report["score_trend"]) == 1


def test_run_service_failure_does_not_persist_traceback(tmp_path: Path) -> None:
    def failing_generate_ideas(keywords, n):
        raise RuntimeError("boom")

    service = RunService(str(tmp_path), idea_generator=failing_generate_ideas)
    run = service.start_run(keywords=["llm"], n=1)

    _wait_for_status(service, run.run_id, RunStatus.FAILED.value)
    details = service.get_run(run.run_id)

    assert details is not None
    assert details["status"] == RunStatus.FAILED.value
    assert details["error"] == "RuntimeError: boom"
