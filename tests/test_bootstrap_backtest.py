from __future__ import annotations

from backend import strategy_store


def _isolate_strategy_store(monkeypatch, tmp_path) -> None:
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(strategy_store, "STRATEGIES_DIR", strategies_dir)


def test_bootstrap_backtest_runs_once_when_baseline_missing(monkeypatch, tmp_path) -> None:
    _isolate_strategy_store(monkeypatch, tmp_path)

    s1 = strategy_store.create_strategy({"strategy_name": "keyword_trend"})
    s2 = strategy_store.create_strategy({"strategy_name": "keyword_trend"})
    executed: list[str] = []

    def _fake_run_backtest(strategy_id: str) -> None:
        executed.append(strategy_id)
        strategy_store.update_strategy(
            strategy_id,
            {
                "backtest_status": "done",
                "backtest_result": {
                    "summary": {"windows": 1, "avg_hit_at_k": 0.5},
                    "windows": [],
                },
            },
        )

    monkeypatch.setattr(strategy_store, "run_backtest_sync", _fake_run_backtest)

    result = strategy_store.bootstrap_backtest_if_missing()
    assert result["triggered"] is True
    assert result["count"] == 2
    assert set(executed) == {s1["id"], s2["id"]}


def test_bootstrap_backtest_skips_when_baseline_exists(monkeypatch, tmp_path) -> None:
    _isolate_strategy_store(monkeypatch, tmp_path)

    s = strategy_store.create_strategy({"strategy_name": "keyword_trend"})
    strategy_store.update_strategy(
        s["id"],
        {
            "backtest_status": "done",
            "backtest_result": {
                "summary": {"windows": 2, "avg_hit_at_k": 0.7},
                "windows": [],
            },
        },
    )

    called = {"value": False}

    def _should_not_run(strategy_id: str) -> None:
        called["value"] = True

    monkeypatch.setattr(strategy_store, "run_backtest_sync", _should_not_run)
    result = strategy_store.bootstrap_backtest_if_missing()

    assert result["triggered"] is False
    assert result["reason"] == "baseline_exists"
    assert called["value"] is False


def test_bootstrap_backtest_skips_when_completed_backtest_data_exists(
    monkeypatch, tmp_path
) -> None:
    _isolate_strategy_store(monkeypatch, tmp_path)

    s = strategy_store.create_strategy({"strategy_name": "keyword_trend"})
    strategy_store.update_strategy(
        s["id"],
        {
            "backtest_status": "done",
            "backtest_result": {"summary": {}},
        },
    )

    called = {"value": False}

    def _should_not_run(strategy_id: str) -> None:
        called["value"] = True

    monkeypatch.setattr(strategy_store, "run_backtest_sync", _should_not_run)
    result = strategy_store.bootstrap_backtest_if_missing()

    assert result["triggered"] is False
    assert result["reason"] == "baseline_exists"
    assert called["value"] is False
