from __future__ import annotations

import os
from typing import Any

import anthropic
import google.generativeai as genai
import openai
from google.generativeai.types import GenerationConfig

MAX_NUM_TOKENS = 4096


def _unsupported_model_error(model: str) -> ValueError:
    return ValueError(
        "Unsupported model "
        f"'{model}'. Supported model families: claude-*, gpt-4o*, gpt-5*, *gemini*."
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


def create_client(model: str) -> tuple[Any, str]:
    if _is_anthropic_model(model):
        api_key = _require_api_key("ANTHROPIC_API_KEY", model)
        return anthropic.Anthropic(api_key=api_key), model

    if _is_openai_model(model):
        api_key = _require_api_key("OPENAI_API_KEY", model)
        return openai.OpenAI(api_key=api_key), model

    if _is_gemini_model(model):
        api_key = _require_api_key("GOOGLE_API_KEY", model)
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model), model

    raise _unsupported_model_error(model)


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
) -> tuple[str, list[dict[str, Any]]]:
    if msg_history is None:
        msg_history = []

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
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            "n": 1,
            "stop": None,
            "seed": 0 if seed is None else seed,
        }
        if model.startswith("gpt-5"):
            request_kwargs["max_completion_tokens"] = MAX_NUM_TOKENS
        else:
            request_kwargs["temperature"] = temperature
            request_kwargs["max_tokens"] = MAX_NUM_TOKENS
            if top_p is not None:
                request_kwargs["top_p"] = top_p

        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content or ""
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

    return content, new_msg_history
