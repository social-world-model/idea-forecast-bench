"""
Strategy store: CRUD for Strategy objects, backed by per-file JSON in backend/strategies/.

A Strategy bundles:
  - strategy_name : which IdeaStrategy implementation ("keyword_trend", …)
  - params        : strategy hyper-params (recent_months, min_keyword_freq)
  - config        : BacktestConfig fields + data_dir
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from live_idea_bench.papers import load_papers_from_markdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "arxiv_csml" / "raw_markdown"
DEFAULT_MODEL_NAME = "gpt-4o"

STRATEGIES_DIR = Path(__file__).parent / "strategies"
STRATEGIES_DIR.mkdir(exist_ok=True)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _path(strategy_id: str) -> Path:
    return STRATEGIES_DIR / f"{strategy_id}.json"


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _runtime_default_data_dir() -> Path:
    override = os.environ.get("LIVE_IDEA_BENCH_DATA_DIR", "").strip()
    if not override:
        return DEFAULT_DATA_DIR
    return Path(override).expanduser()


def _normalize_params(
    strategy_name: str,
    params: object,
    *,
    strict_keyword_coercion: bool = False,
) -> dict:
    raw_params = params if isinstance(params, dict) else {}
    normalized_params = dict(raw_params)

    if strategy_name == "keyword_trend":
        if strict_keyword_coercion:
            normalized_params["recent_months"] = int(raw_params.get("recent_months", 3))
            normalized_params["min_keyword_freq"] = int(
                raw_params.get("min_keyword_freq", 2)
            )
        else:
            normalized_params["recent_months"] = _coerce_int(
                raw_params.get("recent_months", 3), 3
            )
            normalized_params["min_keyword_freq"] = _coerce_int(
                raw_params.get("min_keyword_freq", 2), 2
            )
    else:
        if raw_params.get("model_name") in {None, ""} and raw_params.get("model_id") not in {None, ""}:
            normalized_params["model_name"] = raw_params.get("model_id")
        normalized_params.setdefault("model_name", DEFAULT_MODEL_NAME)
        normalized_params.setdefault("predictor_config", "predictor.yaml")
        normalized_params.setdefault("similarity_config", "similarity.yaml")
        if raw_params.get("temperature") is not None:
            normalized_params["temperature"] = float(raw_params.get("temperature"))

    normalized_params.pop("model_id", None)
    normalized_params.pop("prompt_id", None)
    normalized_params.pop("prompt_version", None)

    return normalized_params


def _normalize_prediction(raw: object, rank_fallback: int) -> dict | None:
    if not isinstance(raw, dict):
        return None

    key_terms_raw = raw.get("key_terms") or raw.get("keywords") or []
    if not isinstance(key_terms_raw, list):
        key_terms_raw = []

    def _coerce_score(value: object, default: float = 0.0) -> float:
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            resolved = default
        if resolved > 1.0:
            resolved = resolved / 10.0
        return round(min(1.0, max(0.0, resolved)), 4)

    confidence = _coerce_score(
        raw.get("confidence", raw.get("Confidence", raw.get("score", raw.get("Score", 0.0)))),
        default=0.0,
    )
    score = _coerce_score(raw.get("score", raw.get("Score", confidence)), default=confidence)

    try:
        rank = int(raw.get("rank", rank_fallback))
    except (TypeError, ValueError):
        rank = rank_fallback

    return {
        "rank": rank,
        "title": str(raw.get("title") or raw.get("Title") or ""),
        "rationale": str(raw.get("rationale") or raw.get("Rationale") or ""),
        "approach": str(raw.get("approach") or raw.get("Approach") or ""),
        "score": score,
        "confidence": confidence,
        "key_terms": [str(term).strip() for term in key_terms_raw if str(term).strip()],
    }


def _normalize_generation(generation: object) -> object:
    if not isinstance(generation, dict):
        return generation

    predictions_raw = generation.get("predictions")
    predictions = []
    if isinstance(predictions_raw, list):
        for idx, raw in enumerate(predictions_raw, start=1):
            normalized = _normalize_prediction(raw, idx)
            if normalized is not None:
                predictions.append(normalized)

    normalized = dict(generation)
    if predictions:
        normalized["predictions"] = predictions
    cutoff_month = str(normalized.get("cutoff_month") or "").strip()
    cutoff_date = str(normalized.get("cutoff_date") or "").strip()
    if not cutoff_date and cutoff_month:
        normalized["cutoff_date"] = f"{cutoff_month}-01"
    return normalized


def _normalize_evaluation(evaluation: object) -> object:
    if not isinstance(evaluation, dict):
        return evaluation
    normalized = dict(evaluation)
    normalized.setdefault("matched_paper_ids", [])
    normalized.pop("matched_terms", None)
    return normalized


def _normalize_backtest_result(backtest_result: object) -> object:
    if not isinstance(backtest_result, dict):
        return backtest_result
    normalized = dict(backtest_result)
    windows = normalized.get("windows")
    if isinstance(windows, list):
        normalized_windows = []
        for window in windows:
            if not isinstance(window, dict):
                continue
            updated = dict(window)
            predictions_raw = updated.get("predictions")
            if isinstance(predictions_raw, list):
                predictions = []
                for idx, raw in enumerate(predictions_raw, start=1):
                    pred = _normalize_prediction(raw, idx)
                    if pred is not None:
                        predictions.append(pred)
                updated["predictions"] = predictions
            updated["evaluation"] = _normalize_evaluation(updated.get("evaluation"))
            normalized_windows.append(updated)
        normalized["windows"] = normalized_windows
    return normalized


def _normalize_strategy(strategy: dict) -> dict:
    strategy_name = str(strategy.get("strategy_name") or "keyword_trend")
    if strategy_name == "prompt_llm":
        strategy_name = "predictor_llm"
    normalized = dict(strategy)
    normalized["strategy_name"] = strategy_name
    normalized["params"] = _normalize_params(strategy_name, strategy.get("params"))
    normalized["generation"] = _normalize_generation(strategy.get("generation"))
    normalized["backtest_result"] = _normalize_backtest_result(strategy.get("backtest_result"))
    normalized["daily_evaluation"] = _normalize_evaluation(strategy.get("daily_evaluation"))
    normalized.setdefault("leaderboard_score", None)
    normalized.setdefault("last_daily_run_at", None)
    normalized.setdefault("last_generation_cutoff_month", None)
    return normalized


def _read(strategy_id: str) -> Optional[dict]:
    p = _path(strategy_id)
    if not p.exists():
        return None
    try:
        loaded = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return None
        return _normalize_strategy(loaded)
    except Exception:
        return None


def _write(strategy: dict) -> None:
    _path(strategy["id"]).write_text(
        json.dumps(strategy, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _sort_key(s: dict) -> tuple:
    score = s.get("leaderboard_score")
    if score is not None:
        try:
            return (float(score), s.get("created_at", ""))
        except (TypeError, ValueError):
            pass
    summary = (s.get("backtest_result") or {}).get("summary") or {}
    hit = summary.get("avg_hit_at_k", -1)
    return (hit, s.get("created_at", ""))


# ── Public API ────────────────────────────────────────────────────────────────

def list_strategies() -> List[dict]:
    strategies = []
    for p in sorted(STRATEGIES_DIR.glob("*.json")):
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                strategies.append(_normalize_strategy(loaded))
        except Exception:
            pass
    strategies.sort(key=_sort_key, reverse=True)
    return strategies


def get_strategy(strategy_id: str) -> Optional[dict]:
    return _read(strategy_id)


def create_strategy(data: dict) -> dict:
    strategy_id = uuid.uuid4().hex[:8]
    strategy_name = str(data.get("strategy_name", "keyword_trend"))
    if strategy_name == "prompt_llm":
        strategy_name = "predictor_llm"
    config = data.get("config") or {}
    raw_data_dir = config.get("data_dir", "")
    data_dir = str(raw_data_dir).strip() if raw_data_dir is not None else ""

    strategy = {
        "id": strategy_id,
        "name": data.get("name") or "{} [{}]".format(
            strategy_name, strategy_id
        ),
        # Which IdeaStrategy implementation to use (matches IdeaStrategy.name)
        "strategy_name": strategy_name,
        # Strategy hyper-parameters passed to the strategy constructor
        "params": _normalize_params(
            strategy_name,
            data.get("params") or {},
            strict_keyword_coercion=True,
        ),
        # BacktestConfig fields + data path
        "config": {
            "top_k": int(config.get("top_k", 5)),
            "horizon_months": int(config.get("horizon_months", 3)),
            "min_train_papers": int(config.get("min_train_papers", 6)),
            "start_month": config.get("start_month", "2024-01"),
            "end_month": config.get("end_month", "2024-12"),
            "data_dir": data_dir,
        },
        "created_at": datetime.now().isoformat(),
        "backtest_status": "pending",
        "generation_status": "pending",
        "backtest_result": None,    # { summary: {...}, windows: [...] }
        "generation": None,         # { cutoff_date: str, cutoff_month: str, predictions: [...] }
        "leaderboard_score": None,
        "daily_evaluation": None,
        "last_daily_run_at": None,
        "last_generation_cutoff_month": None,
    }
    _write(strategy)
    return strategy


def update_strategy(strategy_id: str, updates: dict) -> Optional[dict]:
    strategy = _read(strategy_id)
    if strategy is None:
        return None
    strategy.update(updates)
    _write(strategy)
    return strategy


def delete_strategy(strategy_id: str) -> bool:
    p = _path(strategy_id)
    if p.exists():
        p.unlink()
        return True
    return False


# ── Engine helpers ─────────────────────────────────────────────────────────────

def _resolve_data_dir(s: dict) -> Path:
    raw_data_dir = (s.get("config") or {}).get("data_dir", "")
    data_dir = str(raw_data_dir).strip() if raw_data_dir is not None else ""

    if not data_dir:
        return _runtime_default_data_dir()

    configured = Path(data_dir).expanduser()
    if configured.is_absolute() and not configured.exists():
        return _runtime_default_data_dir()

    if configured.is_absolute():
        return configured

    return (PROJECT_ROOT / configured).resolve()


def _load_papers(s: dict):
    """Load PaperRecord list for this strategy's data_dir and time window."""
    return load_papers_from_markdown(
        _resolve_data_dir(s),
        start_month=s["config"].get("start_month"),
        end_month=s["config"].get("end_month"),
    )


def resolve_data_dir_for_strategy(strategy: dict) -> Path:
    return _resolve_data_dir(strategy)


def load_papers_for_strategy(strategy: dict):
    return _load_papers(strategy)


def _make_strategy_obj(s: dict):
    """Instantiate the IdeaStrategy from the live_idea_bench registry."""
    from live_idea_bench.strategy.execution import build_strategy

    return build_strategy(s)


# ── Synchronous execution (used for seeding / testing) ───────────────────────

def run_backtest_sync(strategy_id: str) -> None:
    """Run backtest synchronously and persist the result."""
    from live_idea_bench.strategy.execution import run_strategy_backtest

    s = _read(strategy_id)
    if s is None:
        return

    update_strategy(strategy_id, {"backtest_status": "running"})
    try:
        papers = _load_papers(s)
        result = run_strategy_backtest(s, papers)
        update_strategy(strategy_id, {
            "backtest_status": "done",
            "backtest_result": result,
        })
    except Exception as e:
        update_strategy(strategy_id, {
            "backtest_status": "failed",
            "backtest_error": str(e),
        })


def run_generation_sync(strategy_id: str, cutoff_date: str | None = None) -> None:
    """Run idea generation synchronously and persist the result."""
    s = _read(strategy_id)
    if s is None:
        return

    update_strategy(strategy_id, {"generation_status": "running"})
    try:
        # Resolve cutoff_date: use supplied value or derive from config end_month.
        if not cutoff_date:
            config_end_month = (s.get("config") or {}).get("end_month") or ""
            cutoff_date = f"{config_end_month}-01" if config_end_month else ""

        if not cutoff_date:
            raise ValueError("cutoff_date is required for generation")

        from live_idea_bench.strategy.execution import run_strategy_generation

        generation = run_strategy_generation(
            s,
            _load_papers(s),
            cutoff_date=cutoff_date,
        )

        update_strategy(strategy_id, {
            "generation_status": "done",
            "generation": generation,
        })
    except Exception as e:
        update_strategy(strategy_id, {
            "generation_status": "failed",
            "generation_error": str(e),
        })
# ── Demo seeding ───────────────────────────────────────────────────────────────

def seed_demo_strategies() -> None:
    """Seed demo strategies and run their backtests if no strategies exist yet."""
    if list_strategies():
        return

    demos = [
        {
            "name": "Keyword Trend · 2024-H1 → 2024-H2",
            "strategy_name": "keyword_trend",
            "params": {"recent_months": 3, "min_keyword_freq": 1},
            "config": {
                "top_k": 5,
                "horizon_months": 3,
                "min_train_papers": 2,
                "start_month": "2024-01",
                "end_month": "2024-09",
            },
        },
        {
            "name": "Keyword Trend · 2024 Full Year",
            "strategy_name": "keyword_trend",
            "params": {"recent_months": 6, "min_keyword_freq": 1},
            "config": {
                "top_k": 8,
                "horizon_months": 3,
                "min_train_papers": 2,
                "start_month": "2024-01",
                "end_month": "2024-12",
            },
        },
        {
            "name": "Keyword Trend · 2025 Preview",
            "strategy_name": "keyword_trend",
            "params": {"recent_months": 2, "min_keyword_freq": 1},
            "config": {
                "top_k": 5,
                "horizon_months": 2,
                "min_train_papers": 2,
                "start_month": "2025-01",
                "end_month": "2025-06",
            },
        },
    ]

    for demo in demos:
        s = create_strategy(demo)
        print(f"  Seeding: {s['name']}")
        run_backtest_sync(s["id"])
        result = _read(s["id"]) or {}
        summary = (result.get("backtest_result") or {}).get("summary", {})
        print(f"    → status={result.get('backtest_status')}  "
              f"windows={summary.get('windows', 0)}  "
              f"avg_hit@k={summary.get('avg_hit_at_k', 0)}")


def has_leaderboard_baseline() -> bool:
    for strategy in list_strategies():
        backtest_result = strategy.get("backtest_result")
        if isinstance(backtest_result, dict) and backtest_result:
            return True
    return False


def bootstrap_backtest_if_missing() -> dict:
    strategies = list_strategies()
    if not strategies:
        return {"triggered": False, "reason": "no_strategies", "count": 0, "strategy_ids": []}

    if has_leaderboard_baseline():
        return {"triggered": False, "reason": "baseline_exists", "count": 0, "strategy_ids": []}

    executed_ids = []
    for strategy in strategies:
        strategy_id = strategy.get("id")
        if not strategy_id:
            continue
        run_backtest_sync(strategy_id)
        executed_ids.append(strategy_id)

    return {
        "triggered": True,
        "reason": "missing_baseline",
        "count": len(executed_ids),
        "strategy_ids": executed_ids,
    }
