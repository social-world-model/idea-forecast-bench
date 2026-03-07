from __future__ import annotations

from backend import config

# The repo is still single-host and file-backed, so default to one worker.
bind = config.gunicorn_bind()
workers = config.gunicorn_workers()
threads = config.gunicorn_threads()
worker_class = "gthread"

timeout = config.gunicorn_timeout()
graceful_timeout = config.gunicorn_graceful_timeout()
keepalive = config.gunicorn_keepalive()

accesslog = "-"
errorlog = "-"
capture_output = True
loglevel = config.gunicorn_log_level()
preload_app = False
