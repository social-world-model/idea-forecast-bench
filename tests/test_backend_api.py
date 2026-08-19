from __future__ import annotations


def test_protected_strategy_write_requires_admin_token(monkeypatch, tmp_path) -> None:
    from backend import app as app_module
    from backend import strategy_store

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(strategy_store, "STRATEGIES_DIR", strategies_dir)
    monkeypatch.setenv("LIVE_IDEA_ADMIN_TOKEN", "secret-token")
    monkeypatch.setattr(app_module.app, "testing", False)

    client = app_module.app.test_client()
    resp = client.post("/api/strategies", json={"strategy_name": "keyword_trend"})

    assert resp.status_code == 403
    assert resp.get_json()["error"]["code"] == "forbidden"


def test_protected_strategy_write_accepts_valid_admin_token(
    monkeypatch, tmp_path
) -> None:
    from backend import app as app_module
    from backend import strategy_store

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(strategy_store, "STRATEGIES_DIR", strategies_dir)
    monkeypatch.setenv("LIVE_IDEA_ADMIN_TOKEN", "secret-token")
    monkeypatch.setattr(app_module.app, "testing", False)

    client = app_module.app.test_client()
    resp = client.post(
        "/api/strategies",
        json={"strategy_name": "keyword_trend"},
        headers={"X-Live-Idea-Admin-Token": "secret-token"},
    )

    assert resp.status_code == 201
    payload = resp.get_json()
    assert payload["strategy_name"] == "keyword_trend"


def test_unexpected_errors_do_not_leak_internal_details(monkeypatch) -> None:
    from backend import app as app_module

    def _boom():
        raise RuntimeError("secret stack detail")

    monkeypatch.setattr(app_module, "list_strategies", _boom)

    client = app_module.app.test_client()
    resp = client.get("/api/strategies")

    assert resp.status_code == 500
    payload = resp.get_json()
    assert payload["error"]["message"] == "Unexpected server error"
    assert payload["error"]["details"] is None
