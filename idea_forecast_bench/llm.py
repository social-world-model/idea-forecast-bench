from __future__ import annotations

import logging
import os
from typing import Any

from idea_forecast_bench.model_refs import resolve_model_reference

logger = logging.getLogger(__name__)


MAX_NUM_TOKENS = 4096


def _unsupported_model_error(model: str) -> ValueError:
    return ValueError(
        "Unsupported model "
        f"'{model}'. Supported model families: claude-*, gpt-4o*, gpt-4.1*, gpt-5*, "
        "*gemini*, "
        "Together AI hosted (deepseek-ai/*, Qwen/*), "
        "DashScope hosted (any id, with DASHSCOPE_API_KEY set), "
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
    # gpt-4.1 is listed explicitly: without it `gpt-4.1` fell through to
    # _is_local_model, resolved to nothing, and raised "Unsupported model" --
    # so the model could not be used as a generation backbone at all, even
    # though gpt-4.1-mini was already the default judge.
    return model.startswith(("gpt-4o", "gpt-4.1", "gpt-5"))


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


# Alibaba DashScope (Model Studio / 百炼) exposes an OpenAI-compatible endpoint
# that hosts third-party models under `vendor/model` ids, e.g.
# `vanchin/deepseek-v4-pro-0813`. Those ids collide with Hugging Face ids, so
# without this branch they fall through to the LOCAL backend and the client
# tries to download a hosted-only model.
#
# Detection rule: DASHSCOPE_API_KEY is set AND the id is not an existing local
# path (a real checkpoint on disk always wins). Everything not claimed by an
# earlier family therefore goes to DashScope while that key is exported --
# unset it when you mean to load a local model.
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _is_dashscope_model(model: str) -> bool:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        return False
    if (
        _is_openai_model(model)
        or _is_anthropic_model(model)
        or _is_gemini_model(model)
        or _is_together_model(model)
        or _is_deepseek_official_model(model)
    ):
        return False
    from pathlib import Path as _Path

    return not _Path(model).exists()


# Reasoning budgets for DashScope thinking mode. Keep them small: the reply
# still has to fit in MAX_NUM_TOKENS, and a long reasoning trace on a strict
# JSON task is how a run ends up with truncated, unparseable output.
_THINKING_BUDGETS: dict[str, int] = {"low": 512, "medium": 1024, "high": 2048}
_THINKING_OFF = {"off", "none", "disabled", "false", "0"}


def dashscope_thinking_body(reasoning_effort: str | None) -> dict[str, Any]:
    """extra_body for a DashScope call.

    Thinking is OFF unless asked for, because the two highest-volume stages
    (element extraction, the retrieve-then-judge judge) are strict-schema
    tasks where reasoning only multiplies billed output tokens -- and the
    judge's 256-token budget would be spent before it emits a score line.
    Turn it on per stage with `reasoning_effort` or DASHSCOPE_THINKING."""
    level = (reasoning_effort or os.environ.get("DASHSCOPE_THINKING") or "off").lower()
    if level in _THINKING_OFF:
        return {"enable_thinking": False}
    budget = _THINKING_BUDGETS.get(level)
    if budget is None:
        raise ValueError(
            f"Unknown thinking level {level!r}; use one of "
            f"{', '.join(sorted(_THINKING_BUDGETS))} or off."
        )
    return {"enable_thinking": True, "thinking_budget": budget}


def _is_local_model(model: str) -> bool:
    if (
        _is_openai_model(model)
        or _is_anthropic_model(model)
        or _is_gemini_model(model)
        or _is_together_model(model)
        or _is_deepseek_official_model(model)
        or _is_dashscope_model(model)
    ):
        return False
    return resolve_model_reference(model) is not None


def _stream_idle_timeout() -> Any:
    """Per-phase HTTP timeout for the OpenAI-compatible clients.

    With ``stream=True`` the OpenAI SDK does not surface an APITimeoutError on
    a long server hang unless the underlying HTTP client has per-phase
    timeouts, so a plain float is not enough.

    The Timeout class has to come from the same HTTP library the installed
    openai SDK uses: openai >= 3 vendors httpx2, older releases use httpx.
    Building it from the wrong one hands the client a foreign object.
    """
    import importlib

    for module_name in ("httpx2", "httpx"):
        try:
            http_lib = importlib.import_module(module_name)
        except ImportError:
            continue
        return http_lib.Timeout(connect=30.0, read=120.0, write=60.0, pool=60.0)
    raise ImportError("neither httpx2 nor httpx is installed; openai requires one")


def create_client(model: str) -> tuple[Any, str]:
    if _is_anthropic_model(model):
        import anthropic

        api_key = _require_api_key("ANTHROPIC_API_KEY", model)
        return anthropic.Anthropic(api_key=api_key), model

    if _is_openai_model(model):
        import os as _os

        import openai

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
        import openai

        api_key = _require_api_key("TOGETHER_API_KEY", model)
        timeout = _stream_idle_timeout()
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
        import openai

        api_key = _require_api_key("DEEPSEEK_API_KEY", model)
        timeout = _stream_idle_timeout()
        return (
            openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                timeout=timeout,
                max_retries=2,
            ),
            model,
        )

    if _is_dashscope_model(model):
        import openai

        api_key = _require_api_key("DASHSCOPE_API_KEY", model)
        base_url = os.environ.get("DASHSCOPE_BASE_URL") or DASHSCOPE_BASE_URL
        timeout = _stream_idle_timeout()
        return (
            openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=2,
            ),
            model,
        )

    if _is_gemini_model(model):
        import google.generativeai as genai

        api_key = _require_api_key("GOOGLE_API_KEY", model)
        # google-generativeai re-exports these from its __init__ without an
        # explicit `as` alias, so strict mode's no_implicit_reexport hides them.
        genai.configure(api_key=api_key)  # type: ignore[attr-defined]
        return genai.GenerativeModel(model), model  # type: ignore[attr-defined]

    if _is_local_model(model):
        resolved_model = resolve_model_reference(model)
        if resolved_model is None:
            raise _unsupported_model_error(model)
        return None, resolved_model

    raise _unsupported_model_error(model)


_USAGE_LOCK = __import__("threading").Lock()


def _log_usage(model: str, usage: Any) -> None:
    """Append one JSON line per call to IDEA_FORECAST_USAGE_LOG, when set.

    A pilot's real token counts are the only way to size a full run; reasoning
    tokens in particular are billed but never appear in the reply, so they are
    invisible without this."""
    path = os.environ.get("IDEA_FORECAST_USAGE_LOG")
    if not path or usage is None:
        return
    import json
    import time

    record = {
        "ts": round(time.time(), 3),
        "model": model,
        "stage": os.environ.get("IDEA_FORECAST_USAGE_STAGE", ""),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        record["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details is not None:
        record["cached_tokens"] = getattr(prompt_details, "cached_tokens", None)
    try:
        with _USAGE_LOCK, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError as exc:  # a ledger must never take the run down
        logger.debug("usage log write failed: %s", exc)


def _sanitize_text(text: str) -> str:
    """Remove null bytes and control characters that cause JSON parse errors in API requests."""
    return "".join(
        ch
        for ch in text
        if ch == "\n" or ch == "\t" or (ord(ch) >= 32 and ord(ch) != 127)
    )


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

    if _is_anthropic_model(model):
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [{"type": "text", "text": msg}],
            }
        ]
        request_kwargs: dict[str, Any] = {
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
        request_kwargs = {
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
    elif (
        _is_together_model(model)
        or _is_deepseek_official_model(model)
        or _is_dashscope_model(model)
    ):
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
        if _is_dashscope_model(model):
            # DashScope takes thinking control at the top level of the body,
            # and reports usage only when asked for it on a stream.
            request_kwargs["extra_body"] = dashscope_thinking_body(reasoning_effort)
            request_kwargs["stream_options"] = {"include_usage": True}
        stream = client.chat.completions.create(**request_kwargs)
        chunks: list[str] = []
        usage: Any = None
        for chunk in stream:
            usage = getattr(chunk, "usage", None) or usage
            if not chunk.choices:
                continue
            piece = getattr(chunk.choices[0].delta, "content", None)
            if piece:
                chunks.append(piece)
        _log_usage(model, usage)
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

        # Imported here, not at module scope: google-generativeai is deprecated
        # upstream and emits a FutureWarning on import, which every command in
        # the CLI was printing whether or not it used Gemini. `genai` itself is
        # already imported lazily in create_client for the same reason.
        from google.generativeai.types import GenerationConfig

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
            apply_chat_template,
            load_local_model,
            require_local_generation_stack,
        )

        resolved_model = resolve_model_reference(model)
        if resolved_model is None:
            raise _unsupported_model_error(model)

        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        model_obj, tokenizer = load_local_model(resolved_model)
        deps = require_local_generation_stack()
        torch = deps["torch"]

        full_prompt = f"{system_message}\n\n{msg}".strip()
        chat_prompt = apply_chat_template(tokenizer, full_prompt, None)
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
        output_ids = generated[0][len(encoded["input_ids"][0]) :].tolist()
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
        model,
        len(content),
        content[:300].replace("\n", " "),
    )
    return content, new_msg_history
