from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import anthropic
import google.generativeai as genai
import openai
from google.generativeai.types import GenerationConfig

MAX_NUM_TOKENS = 4096
DEEPSEEK_TOGETHER_MODEL_ALIASES = {
    "deepseek-chat": "deepseek-ai/DeepSeek-V3.1",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
    "deepseek-reasoner": "deepseek-ai/DeepSeek-R1",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
}


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    name: str
    api_key_envs: tuple[str, ...]
    default_base_url: str | None = None
    base_url_env: str | None = None


def _unsupported_model_error(model: str) -> ValueError:
    return ValueError(
        "Unsupported model "
        f"'{model}'. Supported model families: claude-*, gpt-4o*, gpt-5*, *gemini*, "
        "deepseek*, qwen*, kimi/moonshot*, grok*, llama*."
    )


def _require_api_key(env_vars: str | tuple[str, ...], model: str) -> str:
    env_names = (env_vars,) if isinstance(env_vars, str) else env_vars
    for env_var in env_names:
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key

    joined = ", ".join(env_names)
    raise ValueError(
        f"Missing required API key for model '{model}'. "
        f"Set one of: {joined}."
    )


def _is_openai_model(model: str) -> bool:
    normalized = model.lower()
    return normalized.startswith("gpt-4o") or normalized.startswith("gpt-5")


def _is_anthropic_model(model: str) -> bool:
    return model.lower().startswith("claude-")


def _is_gemini_model(model: str) -> bool:
    return "gemini" in model.lower()


def _is_deepseek_model(model: str) -> bool:
    return model.lower().startswith("deepseek")


def _is_qwen_model(model: str) -> bool:
    return "qwen" in model.lower()


def _is_kimi_model(model: str) -> bool:
    normalized = model.lower()
    return (
        normalized.startswith("kimi")
        or normalized.startswith("moonshot")
        or "/kimi" in normalized
    )


def _is_xai_model(model: str) -> bool:
    normalized = model.lower()
    return normalized.startswith("grok") or normalized.startswith("xai/")


def _is_meta_model(model: str) -> bool:
    normalized = model.lower()
    return "llama" in normalized or normalized.startswith("meta-llama/")


def _is_openai_compatible_model(model: str) -> bool:
    return any(
        matcher(model)
        for matcher in (
            _is_openai_model,
            _is_deepseek_model,
            _is_qwen_model,
            _is_kimi_model,
            _is_xai_model,
            _is_meta_model,
        )
    )


def _provider_base_url(provider: OpenAICompatibleProvider) -> str | None:
    if provider.base_url_env:
        override = os.environ.get(provider.base_url_env)
        if override:
            return override.strip()
    return provider.default_base_url


def _resolved_provider_model_name(model: str) -> str:
    return DEEPSEEK_TOGETHER_MODEL_ALIASES.get(model.lower(), model)


def _openai_provider_for_model(model: str) -> OpenAICompatibleProvider | None:
    if _is_openai_model(model):
        return OpenAICompatibleProvider("OpenAI", ("OPENAI_API_KEY",))
    if _is_deepseek_model(model):
        return OpenAICompatibleProvider(
            "DeepSeek/Together",
            ("TOGETHER_API_KEY", "TOGETHERAI_API_KEY"),
            default_base_url="https://api.together.xyz/v1",
            base_url_env="TOGETHER_BASE_URL",
        )
    if _is_qwen_model(model):
        return OpenAICompatibleProvider(
            "Qwen",
            ("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
            default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            base_url_env="QWEN_BASE_URL",
        )
    if _is_kimi_model(model):
        return OpenAICompatibleProvider(
            "Kimi",
            ("KIMI_API_KEY", "MOONSHOT_API_KEY"),
            default_base_url="https://api.moonshot.cn/v1",
            base_url_env="KIMI_BASE_URL",
        )
    if _is_xai_model(model):
        return OpenAICompatibleProvider(
            "xAI",
            ("XAI_API_KEY",),
            default_base_url="https://api.x.ai/v1",
            base_url_env="XAI_BASE_URL",
        )
    if _is_meta_model(model):
        return OpenAICompatibleProvider(
            "Meta/Together",
            ("TOGETHER_API_KEY", "TOGETHERAI_API_KEY"),
            default_base_url="https://api.together.xyz/v1",
            base_url_env="TOGETHER_BASE_URL",
        )
    return None


def create_client(model: str) -> tuple[Any, str]:
    if _is_anthropic_model(model):
        api_key = _require_api_key(("ANTHROPIC_API_KEY",), model)
        return anthropic.Anthropic(api_key=api_key), model

    provider = _openai_provider_for_model(model)
    if provider is not None:
        resolved_model = _resolved_provider_model_name(model)
        api_key = _require_api_key(provider.api_key_envs, model)
        base_url = _provider_base_url(provider)
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        return openai.OpenAI(**client_kwargs), resolved_model

    if _is_gemini_model(model):
        api_key = _require_api_key(("GOOGLE_API_KEY",), model)
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
        response = client.messages.create(
            model=model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": content}],
            }
        ]
    elif _is_openai_compatible_model(model):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
        }
        if _is_openai_model(model):
            request_kwargs["n"] = 1
            request_kwargs["stop"] = None
            request_kwargs["seed"] = 0
            if model.lower().startswith("gpt-5"):
                request_kwargs["max_completion_tokens"] = MAX_NUM_TOKENS
            else:
                request_kwargs["temperature"] = temperature
                request_kwargs["max_tokens"] = MAX_NUM_TOKENS
        else:
            request_kwargs["temperature"] = temperature
            request_kwargs["max_tokens"] = MAX_NUM_TOKENS

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

        response = client.generate_content(
            contents=gemini_contents,
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=MAX_NUM_TOKENS,
                candidate_count=1,
            ),
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
