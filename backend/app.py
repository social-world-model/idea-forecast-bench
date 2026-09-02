from __future__ import annotations

import json
import os
import secrets
import threading
from datetime import datetime, timezone
from typing import Any

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from backend.strategy_store import (
    bootstrap_backtest_if_missing,
    create_strategy,
    delete_strategy,
    get_strategy,
    list_strategies,
    run_backtest_sync,
    run_generation_sync,
    update_strategy,
)

app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWS_FILE = os.path.join(PROJECT_ROOT, "data", "views.json")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return value not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _cors_origins() -> list[str]:
    raw = os.environ.get("LIVE_IDEA_CORS_ORIGINS", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def _admin_token() -> str:
    return os.environ.get("LIVE_IDEA_ADMIN_TOKEN", "").strip()


def _request_admin_token() -> str:
    bearer = request.headers.get("Authorization", "").strip()
    if bearer.lower().startswith("bearer "):
        return bearer[7:].strip()
    return request.headers.get("X-Live-Idea-Admin-Token", "").strip()


def _auth_bypass_enabled() -> bool:
    # Only Flask's own testing flag, which a test client sets explicitly.
    # This used to also honour the PYTEST_CURRENT_TEST environment variable,
    # which meant admin auth on every write endpoint could be switched off by
    # anything able to set an env var in the server's process.
    return bool(app.testing)


def _is_protected_write_request() -> bool:
    if request.method not in {"POST", "DELETE", "PUT", "PATCH"}:
        return False
    protected_paths = ("/api/strategies",)
    return any(
        request.path == path or request.path.startswith(f"{path}/")
        for path in protected_paths
    )


CORS(
    app,
    resources={
        r"/api/*": {"origins": _cors_origins()},
        r"/healthz": {"origins": _cors_origins()},
        r"/metrics": {"origins": _cors_origins()},
    },
)


def _bootstrap_leaderboard_async() -> None:
    if not _env_flag("LIVE_IDEA_BOOTSTRAP_BACKTEST", True):
        return

    def _worker() -> None:
        try:
            bootstrap_backtest_if_missing()
        except Exception as exc:  # pragma: no cover - defensive startup guard
            print(f"[bootstrap] backtest bootstrap failed: {exc}")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


_bootstrap_leaderboard_async()


class APIError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        code: str = "bad_request",
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details


@app.before_request
def guard_protected_write_endpoints() -> None:
    if not _is_protected_write_request() or _auth_bypass_enabled():
        return None

    expected = _admin_token()
    if not expected:
        raise APIError(
            "Protected write endpoints are disabled until LIVE_IDEA_ADMIN_TOKEN is configured",
            status_code=503,
            code="admin_token_not_configured",
        )

    provided = _request_admin_token()
    if not provided or not secrets.compare_digest(provided, expected):
        raise APIError("Admin token required", status_code=403, code="forbidden")
    return None


@app.errorhandler(APIError)
def handle_api_error(error: APIError) -> tuple[Response, int]:
    payload = {
        "error": {
            "code": error.code,
            "message": error.message,
            "details": error.details if app.debug or app.testing else None,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(payload), error.status_code


@app.errorhandler(HTTPException)
def handle_http_exception(error: HTTPException) -> tuple[Response, int]:
    """Keep routing errors as routing errors.

    Flask dispatches HTTPException (404, 405, 400, ...) to the most specific
    handler. Without this one they fell through to the Exception handler below
    and every mistyped URL was reported as a 500 "internal_error" -- to clients
    and, more expensively, in the logs.
    """
    return (
        jsonify(
            {
                "error": {
                    "code": (error.name or "http_error").lower().replace(" ", "_"),
                    "message": error.description or error.name,
                    "details": None,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ),
        error.code or 500,
    )


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception) -> tuple[Response, int]:
    app.logger.exception("Unhandled server error")
    payload = {
        "error": {
            "code": "internal_error",
            "message": "Unexpected server error",
            "details": None,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(payload), 500


def _read_views() -> int:
    if not os.path.exists(VIEWS_FILE):
        return 0
    try:
        with open(VIEWS_FILE, encoding="utf-8") as fh:
            payload = json.load(fh)
        return int(payload.get("views", 0))
    except Exception:
        return 0


def _write_views(views: int) -> None:
    os.makedirs(os.path.dirname(VIEWS_FILE), exist_ok=True)
    with open(VIEWS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"views": views}, fh)


@app.route("/healthz", methods=["GET"])
@app.route("/api/health", methods=["GET", "POST"])
def healthz() -> Response | tuple[Response, int]:
    if request.method == "POST":
        return jsonify({"error": "Method not allowed"}), 405
    return jsonify(
        {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "idea-forecast-bench-backend",
        }
    )


@app.route("/metrics", methods=["GET"])
def metrics() -> Response:
    strategies = list_strategies()
    backtest_done = sum(
        1 for strategy in strategies if strategy.get("backtest_status") == "done"
    )
    backtest_running = sum(
        1 for strategy in strategies if strategy.get("backtest_status") == "running"
    )
    generation_done = sum(
        1 for strategy in strategies if strategy.get("generation_status") == "done"
    )
    generation_running = sum(
        1 for strategy in strategies if strategy.get("generation_status") == "running"
    )
    return jsonify(
        {
            "strategies_total": len(strategies),
            "backtests_done": backtest_done,
            "backtests_running": backtest_running,
            "generations_done": generation_done,
            "generations_running": generation_running,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.route("/api/views", methods=["GET", "POST"])
def api_views() -> Response:
    views = _read_views()
    if request.method == "POST":
        views += 1
        _write_views(views)
    return jsonify({"views": views})


@app.route("/api/strategies", methods=["GET", "POST"])
def api_strategies() -> Response | tuple[Response, int]:
    if request.method == "GET":
        return jsonify(list_strategies())

    payload = request.get_json(silent=True) or {}
    strategy = create_strategy(payload)
    return jsonify(strategy), 201


@app.route("/api/strategies/<strategy_id>", methods=["GET", "DELETE"])
def api_strategy(strategy_id: str) -> Response | tuple[str, int]:
    if request.method == "GET":
        strategy = get_strategy(strategy_id)
        if strategy is None:
            raise APIError("strategy not found", status_code=404, code="not_found")
        return jsonify(strategy)

    deleted = delete_strategy(strategy_id)
    if not deleted:
        raise APIError("strategy not found", status_code=404, code="not_found")
    return "", 204


@app.route("/api/strategies/<strategy_id>/status", methods=["GET"])
def api_strategy_status(strategy_id: str) -> Response:
    strategy = get_strategy(strategy_id)
    if strategy is None:
        raise APIError("strategy not found", status_code=404, code="not_found")

    return jsonify(
        {
            "id": strategy["id"],
            "backtest_status": strategy.get("backtest_status", "pending"),
            "generation_status": strategy.get("generation_status", "pending"),
        }
    )


@app.route("/api/strategies/<strategy_id>/backtest", methods=["POST"])
def api_strategy_backtest(strategy_id: str) -> tuple[Response, int]:
    strategy = get_strategy(strategy_id)
    if strategy is None:
        raise APIError("strategy not found", status_code=404, code="not_found")

    # Persist running status synchronously before spawning thread (prevents status race)
    update_strategy(strategy_id, {"backtest_status": "running"})

    def _worker() -> None:
        run_backtest_sync(strategy_id)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return jsonify({"id": strategy_id, "backtest_status": "running"}), 202


@app.route("/api/strategies/<strategy_id>/generate", methods=["POST"])
def api_strategy_generate(strategy_id: str) -> tuple[Response, int]:
    strategy = get_strategy(strategy_id)
    if strategy is None:
        raise APIError("strategy not found", status_code=404, code="not_found")

    payload = request.get_json(silent=True) or {}
    cutoff_date_raw = payload.get("cutoff_date")
    if not isinstance(cutoff_date_raw, str) or not cutoff_date_raw.strip():
        raise APIError("cutoff_date is required (YYYY-MM-DD)", status_code=400)
    from idea_forecast_bench.papers import normalize_date

    try:
        cutoff_date = normalize_date(cutoff_date_raw.strip())
    except ValueError as exc:
        raise APIError(str(exc), status_code=400) from exc

    # Persist running status synchronously before spawning thread (prevents status race)
    update_strategy(strategy_id, {"generation_status": "running"})

    def _worker() -> None:
        run_generation_sync(strategy_id, cutoff_date=cutoff_date)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return jsonify({"id": strategy_id, "generation_status": "running"}), 202


if __name__ == "__main__":
    app.run(
        debug=_env_flag("FLASK_DEBUG", False),
        port=_env_int("PORT", 5000),
    )
