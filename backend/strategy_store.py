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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "arxiv_csml" / "raw_markdown"

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

    return normalized_params


def _normalize_strategy(strategy: dict) -> dict:
    strategy_name = str(strategy.get("strategy_name") or "keyword_trend")
    normalized = dict(strategy)
    normalized["strategy_name"] = strategy_name
    normalized["params"] = _normalize_params(strategy_name, strategy.get("params"))
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
        "generation": None,         # { cutoff_month: str, predictions: [...] }
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
    from src.backtest import load_papers_from_markdown
    return load_papers_from_markdown(
        _resolve_data_dir(s),
        start_month=s["config"].get("start_month"),
        end_month=s["config"].get("end_month"),
    )


def _make_strategy_obj(s: dict):
    """Instantiate the IdeaStrategy from src.strategy registry."""
    from src import create_strategy as src_create

    strategy_name = str(s.get("strategy_name") or "keyword_trend")
    params = s.get("params") or {}

    return src_create(
        strategy_name,
        recent_months=_coerce_int(params.get("recent_months", 3), 3),
        min_keyword_freq=_coerce_int(params.get("min_keyword_freq", 2), 2),
        model_id=str(params.get("model_id", "gpt-4o-mini")),
        prompt_id=str(params.get("prompt_id", "llm_baseline")),
        prompt_version=str(params.get("prompt_version", "v1")),
        temperature=(
            float(params.get("temperature"))  # type: ignore[arg-type]
            if params.get("temperature") is not None
            else None
        ),
    )


# ── Synchronous execution (used for seeding / testing) ───────────────────────

def run_backtest_sync(strategy_id: str) -> None:
    """Run backtest synchronously and persist the result."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from src import BacktestConfig, backtest

    s = _read(strategy_id)
    if s is None:
        return

    update_strategy(strategy_id, {"backtest_status": "running"})
    try:
        papers = _load_papers(s)
        strategy_obj = _make_strategy_obj(s)
        cfg = BacktestConfig(
            top_k=s["config"]["top_k"],
            horizon_months=s["config"]["horizon_months"],
            min_train_papers=s["config"]["min_train_papers"],
            start_month=s["config"].get("start_month"),
            end_month=s["config"].get("end_month"),
        )
        result = backtest(papers, strategy_obj, cfg)
        update_strategy(strategy_id, {
            "backtest_status": "done",
            "backtest_result": result,
        })
    except Exception as e:
        update_strategy(strategy_id, {
            "backtest_status": "failed",
            "backtest_error": str(e),
        })


def run_generation_sync(strategy_id: str, cutoff_month: str | None = None) -> None:
    """Run idea generation synchronously and persist the result."""
    import dataclasses
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    s = _read(strategy_id)
    if s is None:
        return

    update_strategy(strategy_id, {"generation_status": "running"})
    try:
        # Resolve cutoff_month: use supplied value or derive from config end_month
        if not cutoff_month:
            cutoff_month = (s.get("config") or {}).get("end_month") or ""

        if not cutoff_month:
            raise ValueError("cutoff_month is required for generation")

        from src.backtest.data import month_to_index

        top_k = int((s.get("config") or {}).get("top_k", 5))
        cutoff_idx = month_to_index(cutoff_month)

        all_papers = _load_papers(s)
        # Filter to train-only: papers whose month index <= cutoff_month index
        train_papers = [p for p in all_papers if month_to_index(p.month) <= cutoff_idx]

        strategy_obj = _make_strategy_obj(s)

        # generate() signature: (train_papers, cutoff_month, top_k)
        predictions_raw = strategy_obj.generate(
            train_papers=train_papers,
            cutoff_month=cutoff_month,
            top_k=top_k,
        )

        # IdeaPrediction is a dataclass — serialize with dataclasses.asdict
        predictions = [
            dataclasses.asdict(p) if dataclasses.is_dataclass(p) else
            (p._asdict() if hasattr(p, "_asdict") else
             (dict(p) if hasattr(p, "keys") else str(p)))
            for p in predictions_raw
        ]

        update_strategy(strategy_id, {
            "generation_status": "done",
            "generation": {
                "cutoff_month": cutoff_month,
                "predictions": predictions,
            },
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
