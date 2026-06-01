from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Any

from live_idea_bench.model_refs import resolve_model_reference

try:  # google-generativeai is an optional provider dependency
    from google.generativeai.types import GenerationConfig
except ImportError:  # pragma: no cover - exercised only when the dep is absent
    GenerationConfig = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ── Batch-mode state (thread-local) ────────────────────────────────────────────
# Modes:
#   None      — live mode (default); every call goes to the real API
#   "collect" — intercept calls: record into collector, return "" placeholder
#   "replay"  — intercept calls: return from cache, fall back to live if missing
#
# In collect mode, the optional `system_prefix` filter lets you collect only
# calls whose system_message starts with that prefix (e.g. only "compress"
# calls in memory_prompting).  All other calls still return "" without hitting
# the live API.  The optional `cache` dict pre-seeds results so that already-
# known responses (e.g. compress results in round-2 of memory_prompting) are
# served from cache rather than collected again.

_batch_tls = threading.local()


def _req_key(
    model: str,
    system_message: str,
    msg: str,
    reasoning_effort: str | None,
    temperature: float,
) -> str:
    """Stable SHA-256 key for a specific LLM request."""
    content = (
        f"{model}\x00{reasoning_effort or ''}\x00{temperature}\x00"
        f"{system_message}\x00{msg}"
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def batch_set_collect(
    collector: dict,
    cache: dict | None = None,
    system_prefix: str | None = None,
) -> None:
    """Switch current thread to collect mode.

    Parameters
    ----------
    collector:
        Dict to write collected requests into.  Key = request hash,
        value = request parameter dict.  Shared across threads; Python's
        GIL makes dict item assignment atomic so no extra lock is needed.
    cache:
        Optional pre-seeded responses (key → response text).  Cache hits
        are returned directly without collecting or calling the live API.
        Use this in round-2 of memory_prompting to serve compress results.
    system_prefix:
        If set, only collect calls whose system_message starts with this
        string.  Non-matching calls still return "" (no live API call).
    """
    _batch_tls.mode = "collect"
    _batch_tls.collector = collector
    _batch_tls.cache = cache if cache is not None else {}
    _batch_tls.system_prefix = system_prefix


def batch_set_replay(cache: dict) -> None:
    """Switch current thread to replay mode (serve from cache, fall back to live)."""
    _batch_tls.mode = "replay"
    _batch_tls.cache = cache
    _batch_tls.collector = None
    _batch_tls.system_prefix = None


def batch_clear() -> None:
    """Reset current thread to live mode."""
    _batch_tls.mode = None
    _batch_tls.collector = None
    _batch_tls.cache = {}
    _batch_tls.system_prefix = None


MAX_NUM_TOKENS = 4096


def _unsupported_model_error(model: str) -> ValueError:
    return ValueError(
        "Unsupported model "
        f"'{model}'. Supported model families: claude-*, gpt-4o*, gpt-5*, *gemini*, "
        "Together AI hosted (deepseek-ai/*, Qwen/*), "
        "or a local/Hugging Face model reference such as Qwen/Qwen3.5-2B."
    )


def _require_api_key(env_var: str, model: str) -> str:
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(
            f"Missing required API key for model '{model}'. "
            f"Set environment variable {env_var}."
        )
    return api_key


def _is_openai_model(model: str) -> bool:
    return model.startswith("gpt-4o") or model.startswith("gpt-5")


def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude-")


def _is_gemini_model(model: str) -> bool:
    return "gemini" in model


# Together AI hosts the DeepSeek and Qwen API baselines we use.
# Detection rule: route to Together iff the model id is a HF-style "vendor/model"
# string AND TOGETHER_API_KEY is set in the environment. This lets the local
# (transformers) backend still pick up vendor/model ids when TOGETHER_API_KEY
# is not configured.
_TOGETHER_VENDORS = ("deepseek-ai/", "qwen/")


def _is_together_model(model: str) -> bool:
    if not os.environ.get("TOGETHER_API_KEY"):
        return False
    lower = model.lower()
    return any(lower.startswith(v) for v in _TOGETHER_VENDORS)


# DeepSeek official API uses bare ids (no vendor prefix) like
# `deepseek-chat`, `deepseek-reasoner`, `deepseek-v4-pro`.  Detection rule:
# id starts with "deepseek-" AND has no "/" (which would make it a HF id
# routed to Together or the local backend).  Gated by DEEPSEEK_API_KEY so
# `deepseek-r1:70b` style strings on a machine without the key fall through
# to the local backend.
def _is_deepseek_official_model(model: str) -> bool:
    if "/" in model:
        return False
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return False
    return model.lower().startswith("deepseek-")


def _is_local_model(model: str) -> bool:
    if (
        _is_openai_model(model)
        or _is_anthropic_model(model)
        or _is_gemini_model(model)
        or _is_together_model(model)
        or _is_deepseek_official_model(model)
    ):
        return False
    return resolve_model_reference(model) is not None


def create_client(model: str) -> tuple[Any, str]:
    if _is_anthropic_model(model):
        import anthropic

        api_key = _require_api_key("ANTHROPIC_API_KEY", model)
        return anthropic.Anthropic(api_key=api_key), model

    if _is_openai_model(model):
        import openai
        import os as _os

        # When OPENAI_BASE_URL is set, route to that endpoint (used for local
        # vLLM-served LoRA-merged models in evaluation pipelines). The
        # OPENAI_API_KEY is required by the client but vLLM ignores its
        # value, so allow any non-empty string when a base_url is provided.
        base_url = _os.environ.get("OPENAI_BASE_URL") or None
        if base_url:
            api_key = _os.environ.get("OPENAI_API_KEY", "EMPTY") or "EMPTY"
        else:
            api_key = _require_api_key("OPENAI_API_KEY", model)
        return openai.OpenAI(api_key=api_key, base_url=base_url), model

    if _is_together_model(model):
        import httpx
        import openai

        api_key = _require_api_key("TOGETHER_API_KEY", model)
        # Stream-idle hard timeout: with stream=True the OpenAI SDK never
        # surfaces an APITimeoutError on long server hangs unless we set
        # per-phase timeouts on the underlying httpx client.
        timeout = httpx.Timeout(connect=30.0, read=120.0, write=60.0, pool=60.0)
        return (
            openai.OpenAI(
                api_key=api_key,
                base_url="https://api.together.xyz/v1",
                timeout=timeout,
                max_retries=2,
            ),
            model,
        )

    if _is_deepseek_official_model(model):
        import httpx
        import openai

        api_key = _require_api_key("DEEPSEEK_API_KEY", model)
        timeout = httpx.Timeout(connect=30.0, read=120.0, write=60.0, pool=60.0)
        return (
            openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                timeout=timeout,
                max_retries=2,
            ),
            model,
        )

    if _is_gemini_model(model):
        import google.generativeai as genai

        api_key = _require_api_key("GOOGLE_API_KEY", model)
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model), model

    if _is_local_model(model):
        resolved_model = resolve_model_reference(model)
        if resolved_model is None:
            raise _unsupported_model_error(model)
        return None, resolved_model

    raise _unsupported_model_error(model)


def _sanitize_text(text: str) -> str:
    """Remove null bytes and control characters that cause JSON parse errors in API requests."""
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or (ord(ch) >= 32 and ord(ch) != 127))


def get_response_from_llm(
    msg: str,
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.75,
    top_p: float | None = None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    msg = _sanitize_text(msg)
    system_message = _sanitize_text(system_message)
    if msg_history is None:
        msg_history = []

    # ── Batch-mode interception ────────────────────────────────────────────────
    _mode = getattr(_batch_tls, "mode", None)
    if _mode in ("collect", "replay"):
        _key = _req_key(model, system_message, msg, reasoning_effort, temperature)
        _cache: dict = getattr(_batch_tls, "cache", {}) or {}

        # Cache hit: return stored response (works in both collect and replay modes)
        if _key in _cache:
            _content = _cache[_key]
            _hist = (msg_history or []) + [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": _content},
            ]
            return _content, _hist

        if _mode == "collect":
            _sys_prefix = getattr(_batch_tls, "system_prefix", None)
            if _sys_prefix is None or system_message.startswith(_sys_prefix):
                _collector: dict = getattr(_batch_tls, "collector", {})
                _collector[_key] = {
                    "model": model,
                    "system_message": system_message,
                    "msg": msg,
                    "reasoning_effort": reasoning_effort,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                }
            # Return placeholder — do NOT call live API in collect mode
            _hist = (msg_history or []) + [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": ""},
            ]
            return "", _hist

        # replay mode, key not in cache → fall through to live API (safety fallback)
        logger.warning(
            "batch replay: cache miss for key=%s model=%s — falling back to live API",
            _key[:12], model,
        )
    # ── End batch-mode interception ────────────────────────────────────────────

    if _is_anthropic_model(model):
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [{"type": "text", "text": msg}],
            }
        ]
        request_kwargs = {
            "model": model,
            "max_tokens": MAX_NUM_TOKENS,
            "temperature": temperature,
            "system": system_message,
            "messages": new_msg_history,
        }
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        response = client.messages.create(**request_kwargs)
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": content}],
            }
        ]
    elif _is_openai_model(model):
        import os as _os
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
        }
        # When routed to a local vLLM via OPENAI_BASE_URL, we serve a
        # LoRA-adapted Qwen3 that defaults to thinking mode (<think>...
        # </think>). The eval predictor expects clean JSON, so disable
        # thinking via the chat template kwarg (vLLM-specific extra body).
        # Also use temperature/top_p/seed even for "gpt-5*" aliases since
        # this is our LoRA endpoint, not real GPT-5.
        _local_route = bool(_os.environ.get("OPENAI_BASE_URL"))
        if _local_route:
            request_kwargs["max_tokens"] = MAX_NUM_TOKENS
            request_kwargs["temperature"] = temperature
            if top_p is not None:
                request_kwargs["top_p"] = top_p
            if seed is not None:
                request_kwargs["seed"] = seed
            request_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }
        elif model.startswith("gpt-5"):
            request_kwargs["max_completion_tokens"] = MAX_NUM_TOKENS
            if reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = reasoning_effort
        else:
            request_kwargs["n"] = 1
            request_kwargs["stop"] = None
            request_kwargs["seed"] = 0 if seed is None else seed
            request_kwargs["temperature"] = temperature
            request_kwargs["max_tokens"] = MAX_NUM_TOKENS
            if top_p is not None:
                request_kwargs["top_p"] = top_p

        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content or ""
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif _is_together_model(model) or _is_deepseek_official_model(model):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        request_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            "temperature": temperature,
            "max_tokens": MAX_NUM_TOKENS,
            # Together AI rejects non-streaming for some reasoning/thinking
            # models (e.g. Qwen 3.6 Plus -> 400 streaming_required). Stream
            # and reassemble; pairs with the httpx read=120s timeout set on
            # the client so server-side stream hangs raise APITimeoutError.
            "stream": True,
        }
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if seed is not None:
            request_kwargs["seed"] = seed
        stream = client.chat.completions.create(**request_kwargs)
        chunks: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = getattr(chunk.choices[0].delta, "content", None)
            if piece:
                chunks.append(piece)
        content = "".join(chunks)
        # DeepSeek-R1 / Qwen thinking variants leak chain-of-thought inside
        # <think>...</think>; strip it so downstream JSON parsing isn't fooled.
        if "<think>" in content:
            import re
            content = re.sub(
                r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE
            ).strip()
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif _is_gemini_model(model):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        gemini_contents = [{"role": "system", "parts": system_message}]
        for history_msg in new_msg_history:
            gemini_contents.append(
                {"role": history_msg["role"], "parts": history_msg["content"]}
            )

        generation_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": MAX_NUM_TOKENS,
            "candidate_count": 1,
        }
        if top_p is not None:
            generation_kwargs["top_p"] = top_p
        response = client.generate_content(
            contents=gemini_contents,
            generation_config=GenerationConfig(**generation_kwargs),
        )
        content = response.text or ""
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif _is_local_model(model):
        from forecaster.realization.local_generation import (
            _apply_chat_template,
            _load_local_model,
            _require_local_generation_stack,
        )

        resolved_model = resolve_model_reference(model)
        if resolved_model is None:
            raise _unsupported_model_error(model)

        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        model_obj, tokenizer = _load_local_model(resolved_model)
        deps = _require_local_generation_stack()
        torch = deps["torch"]

        full_prompt = f"{system_message}\n\n{msg}".strip()
        chat_prompt = _apply_chat_template(tokenizer, full_prompt, None)
        encoded = tokenizer([chat_prompt], return_tensors="pt")
        encoded = {name: value.to(model_obj.device) for name, value in encoded.items()}

        generate_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": MAX_NUM_TOKENS,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature <= 0:
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
            if top_p is not None:
                generate_kwargs["top_p"] = top_p

        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            generated = model_obj.generate(**generate_kwargs)
        output_ids = generated[0][len(encoded["input_ids"][0]):].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise _unsupported_model_error(model)

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for i, history_msg in enumerate(new_msg_history):
            print(f"Message {i}: {history_msg}")
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    logger.debug(
        "LLM | model=%s | chars=%d | preview=%s",
        model, len(content), content[:300].replace("\n", " "),
    )
    return content, new_msg_history
