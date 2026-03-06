from __future__ import annotations

import time

from backend.services.run_service import RunService, RunStatus


def _wait_for_status(service: RunService, run_id: str, expected: str, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = service.get_run(run_id)
        if run and run.get("status") == expected:
            return
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not reach status {expected}")


def test_runs_api_flow(monkeypatch, tmp_path) -> None:
    from backend import app as app_module

    def fake_generate_ideas(keywords, n):
        return [{"Title": "A", "Score": 9, "Novelty": 9, "Feasibility": 8}]

    monkeypatch.setattr(app_module, "run_service", RunService(str(tmp_path), idea_generator=fake_generate_ideas))

    client = app_module.app.test_client()

    start_resp = client.post("/api/runs/start", json={"keywords": ["vision"], "n": 1})
    assert start_resp.status_code == 202
    run_id = start_resp.get_json()["run"]["run_id"]

    _wait_for_status(app_module.run_service, run_id, RunStatus.SUCCESS.value)

    list_resp = client.get("/api/runs/list")
    assert list_resp.status_code == 200
    assert len(list_resp.get_json()["runs"]) == 1

    detail_resp = client.get(f"/api/runs/{run_id}?includeIdeas=true")
    assert detail_resp.status_code == 200
    payload = detail_resp.get_json()["run"]
    assert payload["run_id"] == run_id
    assert payload["ideas_count"] == 1
    assert len(payload["ideas"]) == 1

    report_resp = client.get("/api/runs/report")
    assert report_resp.status_code == 200
    report = report_resp.get_json()
    assert report["summary"]["successful_runs"] == 1


def test_runs_api_validation(monkeypatch, tmp_path) -> None:
    from backend import app as app_module

    monkeypatch.setattr(app_module, "run_service", RunService(str(tmp_path)))
    client = app_module.app.test_client()

    resp = client.post("/api/runs/start", json={"keywords": [], "n": 0})
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "bad_request"

    resp_not_found = client.get("/api/runs/not-exists")
    assert resp_not_found.status_code == 404
    assert resp_not_found.get_json()["error"]["code"] == "not_found"


def test_generate_ideas_get_is_read_only(monkeypatch, tmp_path) -> None:
    from backend import app as app_module

    ideas_file = tmp_path / "generated_ideas.json"
    monkeypatch.setattr(app_module, "GENERATED_IDEAS_FILE", str(ideas_file))
    client = app_module.app.test_client()

    resp = client.get("/api/generate-ideas")
    assert resp.status_code == 200
    assert resp.get_json() == []
    assert not ideas_file.exists()


def test_protected_write_requires_admin_token(monkeypatch, tmp_path) -> None:
    from backend import app as app_module

    monkeypatch.setenv("LIVE_IDEA_ADMIN_TOKEN", "secret-token")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(app_module, "run_service", RunService(str(tmp_path)))

    client = app_module.app.test_client()
    resp = client.post("/api/runs/start", json={"keywords": ["vision"], "n": 1})

    assert resp.status_code == 403
    assert resp.get_json()["error"]["code"] == "forbidden"


def test_protected_write_accepts_valid_admin_token(monkeypatch, tmp_path) -> None:
    from backend import app as app_module

    def fake_generate_ideas(keywords, n):
        return [{"Title": "A", "Score": 9, "Novelty": 9, "Feasibility": 8}]

    monkeypatch.setenv("LIVE_IDEA_ADMIN_TOKEN", "secret-token")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(
        app_module,
        "run_service",
        RunService(str(tmp_path), idea_generator=fake_generate_ideas),
    )

    client = app_module.app.test_client()
    resp = client.post(
        "/api/runs/start",
        json={"keywords": ["vision"], "n": 1},
        headers={"X-Live-Idea-Admin-Token": "secret-token"},
    )

    assert resp.status_code == 202
    assert "run" in resp.get_json()


def test_unexpected_errors_do_not_leak_internal_details(monkeypatch) -> None:
    from backend import app as app_module

    def _boom():
        raise RuntimeError("secret stack detail")

    monkeypatch.setattr(app_module, "_load_generated_ideas", _boom)

    client = app_module.app.test_client()
    resp = client.get("/api/generate-ideas")

    assert resp.status_code == 500
    payload = resp.get_json()
    assert payload["error"]["message"] == "Unexpected server error"
    assert payload["error"]["details"] is None
