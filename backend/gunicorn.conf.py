from __future__ import annotations

from backend import config as backend_config

# The repo is still single-host and file-backed, so default to one worker.
bind = backend_config.gunicorn_bind()
workers = backend_config.gunicorn_workers()
threads = backend_config.gunicorn_threads()
worker_class = "gthread"

timeout = backend_config.gunicorn_timeout()
graceful_timeout = backend_config.gunicorn_graceful_timeout()
keepalive = backend_config.gunicorn_keepalive()

accesslog = "-"
errorlog = "-"
capture_output = True
loglevel = backend_config.gunicorn_log_level()
preload_app = False
