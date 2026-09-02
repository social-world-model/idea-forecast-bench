from __future__ import annotations

import itertools
import os
import threading
from collections.abc import Sequence
from typing import Any, Protocol

from idea_forecast_bench.llm import create_client, get_response_from_llm


class TextCaller(Protocol):
    """One chat completion: system + user -> assistant text."""

    @property
    def model_name(self) -> str: ...

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str: ...


class ClientCaller:
    """Wraps an already-constructed client with the repo's routing helper."""

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        content, _ = get_response_from_llm(
            msg=user,
            client=self._client,
            model=self._model,
            system_message=system,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        return content


class RoundRobinCaller:
    """Spreads calls over several callers (one per served model replica)."""

    def __init__(self, callers: Sequence[TextCaller]) -> None:
        if not callers:
            raise ValueError("RoundRobinCaller needs at least one caller")
        self._callers = tuple(callers)
        self._cycle = itertools.cycle(range(len(self._callers)))
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._callers[0].model_name

    def _next(self) -> TextCaller:
        with self._lock:
            return self._callers[next(self._cycle)]

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        return self._next().complete(
            system, user, temperature=temperature, top_p=top_p, seed=seed
        )


def caller_for_model(model: str) -> TextCaller:
    """Route through `create_client`: API keys, OPENAI_BASE_URL, local models."""
    client, resolved = create_client(model)
    return ClientCaller(client, resolved)


def callers_for_base_urls(model: str, base_urls: Sequence[str]) -> TextCaller:
    """One OpenAI-compatible client per base URL, used round-robin.

    `get_response_from_llm` decides it is talking to a local server by
    checking the OPENAI_BASE_URL environment variable, which is what makes it
    send `enable_thinking: false`; so that variable is set (to the first URL)
    when absent. Each client still carries its own base_url."""
    import openai

    urls = [u.strip().rstrip("/") for u in base_urls if u.strip()]
    if not urls:
        raise ValueError("at least one base URL is required")
    os.environ.setdefault("OPENAI_BASE_URL", urls[0])
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY") or "EMPTY"
    callers = [
        ClientCaller(openai.OpenAI(api_key=api_key, base_url=url), model)
        for url in urls
    ]
    return RoundRobinCaller(callers)
