from __future__ import annotations

import json
from pathlib import Path


def _isolate_strategy_store(monkeypatch, tmp_path: Path) -> Path:
    from backend import strategy_store

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(strategy_store, "STRATEGIES_DIR", strategies_dir)
    return strategies_dir


def test_prompt_llm_params_round_trip(monkeypatch, tmp_path: Path) -> None:
    from backend import strategy_store

    _isolate_strategy_store(monkeypatch, tmp_path)

    created = strategy_store.create_strategy(
        {
            "strategy_name": "prompt_llm",
            "params": {
                "model_id": "gpt-4o-mini",
                "prompt_id": "llm_baseline",
                "prompt_version": "v1",
                "temperature": 0.2,
            },
        }
    )

    loaded = strategy_store.get_strategy(created["id"])
    assert loaded is not None
    assert loaded["strategy_name"] == "prompt_llm"
    assert loaded["params"]["model_id"] == "gpt-4o-mini"
    assert loaded["params"]["prompt_id"] == "llm_baseline"
    assert loaded["params"]["prompt_version"] == "v1"
    assert loaded["params"]["temperature"] == 0.2
    assert "recent_months" not in loaded["params"]
    assert "min_keyword_freq" not in loaded["params"]


def test_legacy_keyword_strategy_missing_prompt_model_keys_still_generates(
    monkeypatch, tmp_path: Path
) -> None:
    from backend import strategy_store
    from src.backtest.models import PaperRecord

    strategies_dir = _isolate_strategy_store(monkeypatch, tmp_path)
    legacy_id = "legacykw"
    legacy_record = {
        "id": legacy_id,
        "name": "Legacy Keyword Strategy",
        "strategy_name": "keyword_trend",
        "params": {},
        "config": {"top_k": 3, "end_month": "2024-06"},
        "created_at": "2026-01-01T00:00:00",
        "backtest_status": "pending",
        "generation_status": "pending",
        "backtest_result": None,
        "generation": None,
    }
    (strategies_dir / f"{legacy_id}.json").write_text(
        json.dumps(legacy_record, indent=2), encoding="utf-8"
    )

    papers = [
        PaperRecord(
            paper_id="p1",
            title="T1",
            month="2024-06",
            summary="S1",
            keywords=["attention", "transformer"],
            source_path="/fake/p1.md",
        ),
    ]
    monkeypatch.setattr(strategy_store, "_load_papers", lambda _s: papers)

    loaded = strategy_store.get_strategy(legacy_id)
    assert loaded is not None
    assert loaded["params"]["recent_months"] == 3
    assert loaded["params"]["min_keyword_freq"] == 2
    assert "model_id" not in loaded["params"]
    assert "prompt_id" not in loaded["params"]
    assert "prompt_version" not in loaded["params"]

    strategy_store.run_generation_sync(legacy_id, cutoff_month="2024-06")

    refreshed = strategy_store.get_strategy(legacy_id)
    assert refreshed is not None
    assert refreshed["generation_status"] == "done"
    assert refreshed["generation"]["cutoff_month"] == "2024-06"
    assert isinstance(refreshed["generation"]["predictions"], list)


def test_make_strategy_obj_passes_prompt_llm_params(monkeypatch, tmp_path: Path) -> None:
    from backend import strategy_store
    from src.strategy.prompt_llm import PromptLLMStrategy

    _isolate_strategy_store(monkeypatch, tmp_path)

    strategy_data = {
        "strategy_name": "prompt_llm",
        "params": {
            "model_id": "test-model",
            "prompt_id": "test-prompt",
            "prompt_version": "v2",
            "temperature": 0.5,
        },
    }
    created = strategy_store.create_strategy(strategy_data)

    # This calls _make_strategy_obj internally
    obj = strategy_store._make_strategy_obj(created)

    assert isinstance(obj, PromptLLMStrategy)
    assert obj.model_id == "test-model"
    assert obj.prompt_id == "test-prompt"
    assert obj.prompt_version == "v2"
    assert obj.temperature == 0.5