from __future__ import annotations


def test_health_metrics_and_views_endpoints() -> None:
    from backend import app as app_module

    client = app_module.app.test_client()

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"

    before = client.get("/api/views")
    assert before.status_code == 200
    before_views = before.get_json()["views"]

    inc = client.post("/api/views")
    assert inc.status_code == 200

    after = client.get("/api/views")
    assert after.status_code == 200
    assert after.get_json()["views"] >= before_views

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    payload = metrics.get_json()
    assert "strategies_total" in payload
    assert "backtests_done" in payload
    assert "generations_done" in payload
