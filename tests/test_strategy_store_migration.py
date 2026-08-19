from __future__ import annotations

import json
from pathlib import Path


def _isolate_strategy_store(monkeypatch, tmp_path: Path) -> Path:
    from backend import strategy_store

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(strategy_store, "STRATEGIES_DIR", strategies_dir)
    return strategies_dir


def test_prompt_llm_params_round_trip_migrates_to_predictor_mainline(
    monkeypatch, tmp_path: Path
) -> None:
    from backend import strategy_store

    _isolate_strategy_store(monkeypatch, tmp_path)

    created = strategy_store.create_strategy(
        {
            "strategy_name": "prompt_llm",
            "params": {
                "model_id": "gpt-4o-mini",
                "temperature": 0.2,
            },
        }
    )

    loaded = strategy_store.get_strategy(created["id"])
    assert loaded is not None
    assert loaded["strategy_name"] == "predictor_llm"
    assert loaded["params"]["model_name"] == "gpt-4o-mini"
    assert loaded["params"]["predictor_config"] == "predictor.yaml"
    assert loaded["params"]["similarity_config"] == "similarity.yaml"
    assert loaded["params"]["temperature"] == 0.2
    assert "recent_months" not in loaded["params"]
    assert "min_keyword_freq" not in loaded["params"]


def test_legacy_keyword_strategy_missing_prompt_model_keys_still_generates(
    monkeypatch, tmp_path: Path
) -> None:
    from backend import strategy_store
    from live_idea_bench.config import TopicDefinition
    from live_idea_bench.models import PaperRecord

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
            keywords=["optimizer"],
            source_path="/fake/p1.md",
            published_date="2024-06-20",
        ),
    ]
    monkeypatch.setattr(strategy_store, "_load_papers", lambda _s: papers)
    monkeypatch.setattr(
        strategy_store,
        "load_topics",
        lambda: [
            TopicDefinition(id="optimizer", name="Optimizer", keywords=["optimizer"])
        ],
    )

    loaded = strategy_store.get_strategy(legacy_id)
    assert loaded is not None
    assert loaded["params"]["recent_months"] == 3
    assert loaded["params"]["min_keyword_freq"] == 2
    assert "model_id" not in loaded["params"]
    assert "prompt_id" not in loaded["params"]
    assert "prompt_version" not in loaded["params"]

    strategy_store.run_generation_sync(legacy_id, cutoff_date="2024-06-30")

    refreshed = strategy_store.get_strategy(legacy_id)
    assert refreshed is not None
    assert refreshed["generation_status"] == "done"
    assert refreshed["generation"] is None
    assert len(refreshed["topic_runs"]) == 1
    topic_run = refreshed["topic_runs"][0]
    assert topic_run["topic_id"] == "optimizer"
    assert topic_run["generation_status"] == "done"
    assert topic_run["generation"]["cutoff_month"] == "2024-06"
    assert topic_run["generation"]["cutoff_date"] == "2024-06-30"
    assert isinstance(topic_run["generation"]["predictions"], list)


def test_make_strategy_obj_passes_predictor_params(monkeypatch, tmp_path: Path) -> None:
    from backend import strategy_store
    from live_idea_bench.strategy.predictor_llm import PredictorLLMStrategy

    _isolate_strategy_store(monkeypatch, tmp_path)

    strategy_data = {
        "strategy_name": "predictor_llm",
        "params": {
            "model_name": "test-model",
            "predictor_config": "predictor.yaml",
            "similarity_config": "similarity.yaml",
            "temperature": 0.5,
        },
    }
    created = strategy_store.create_strategy(strategy_data)

    # This calls _make_strategy_obj internally
    obj = strategy_store._make_strategy_obj(created)

    assert isinstance(obj, PredictorLLMStrategy)
    assert obj.model_name == "test-model"
    assert obj.predictor_config == "predictor.yaml"
    assert obj.similarity_config == "similarity.yaml"
    assert obj.temperature == 0.5


def test_create_strategy_keeps_data_dir_blank_when_not_provided(
    monkeypatch, tmp_path: Path
) -> None:
    from backend import strategy_store

    _isolate_strategy_store(monkeypatch, tmp_path)

    created = strategy_store.create_strategy({"strategy_name": "keyword_trend"})
    assert created["config"]["data_dir"] == ""

    loaded = strategy_store.get_strategy(created["id"])
    assert loaded is not None
    assert loaded["config"]["data_dir"] == ""


def test_resolve_data_dir_falls_back_for_missing_absolute_path(
    monkeypatch, tmp_path: Path
) -> None:
    from backend import strategy_store

    _isolate_strategy_store(monkeypatch, tmp_path)
    default_dir = tmp_path / "runtime-default"
    default_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LIVE_IDEA_BENCH_DATA_DIR", str(default_dir))

    strategy = {"config": {"data_dir": "/Users/someone/old-machine/path"}}
    resolved = strategy_store._resolve_data_dir(strategy)
    assert resolved == default_dir


def test_resolve_data_dir_uses_project_relative_path(
    monkeypatch, tmp_path: Path
) -> None:
    from backend import strategy_store

    _isolate_strategy_store(monkeypatch, tmp_path)
    relative_path = "data/arxiv_csml/raw_markdown"

    strategy = {"config": {"data_dir": relative_path}}
    resolved = strategy_store._resolve_data_dir(strategy)
    assert resolved == (strategy_store.PROJECT_ROOT / relative_path).resolve()
