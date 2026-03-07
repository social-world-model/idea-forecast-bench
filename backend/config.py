from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LOCAL_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def env_str(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip()


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def backend_port() -> int:
    return env_int("PORT", 5000)


def cors_origins() -> list[str]:
    raw = env_str("LIVE_IDEA_CORS_ORIGINS", "")
    if not raw:
        return list(DEFAULT_LOCAL_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def admin_token() -> str:
    return env_str("LIVE_IDEA_ADMIN_TOKEN", "")


def bootstrap_backtest_enabled() -> bool:
    return env_flag("LIVE_IDEA_BOOTSTRAP_BACKTEST", True)


def data_dir_override() -> str:
    return env_str("LIVE_IDEA_BENCH_DATA_DIR", "")


def gunicorn_bind() -> str:
    return f"0.0.0.0:{backend_port()}"


def gunicorn_workers() -> int:
    return max(1, env_int("GUNICORN_WORKERS", 1))


def gunicorn_threads() -> int:
    return max(1, env_int("GUNICORN_THREADS", 4))


def gunicorn_timeout() -> int:
    return max(30, env_int("GUNICORN_TIMEOUT", 300))


def gunicorn_graceful_timeout() -> int:
    return max(15, env_int("GUNICORN_GRACEFUL_TIMEOUT", 30))


def gunicorn_keepalive() -> int:
    return max(2, env_int("GUNICORN_KEEPALIVE", 5))


def gunicorn_log_level() -> str:
    return env_str("GUNICORN_LOG_LEVEL", "info") or "info"
