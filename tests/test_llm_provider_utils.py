from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import live_idea_bench.llm as llm_utils

API_KEY_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "QWEN_API_KEY",
    "DASHSCOPE_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "XAI_API_KEY",
    "TOGETHER_API_KEY",
    "TOGETHERAI_API_KEY",
]

BASE_URL_ENV_VARS = [
    "QWEN_BASE_URL",
    "KIMI_BASE_URL",
    "XAI_BASE_URL",
    "TOGETHER_BASE_URL",
]


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in [*API_KEY_ENV_VARS, *BASE_URL_ENV_VARS]:
        monkeypatch.delenv(env_var, raising=False)


def test_create_client_unsupported_model_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        llm_utils.create_client("not-a-supported-model")


@pytest.mark.parametrize(
    ("model", "expected_fragment"),
    [
        ("gpt-4o-mini", "OPENAI_API_KEY"),
        ("claude-3-5-sonnet-20241022", "ANTHROPIC_API_KEY"),
        ("gemini-2.5-flash", "GOOGLE_API_KEY"),
        ("deepseek-chat", "TOGETHER_API_KEY, TOGETHERAI_API_KEY"),
        ("qwen-plus", "QWEN_API_KEY, DASHSCOPE_API_KEY"),
        ("moonshot-v1-8k", "KIMI_API_KEY, MOONSHOT_API_KEY"),
        ("grok-2-latest", "XAI_API_KEY"),
        (
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "TOGETHER_API_KEY, TOGETHERAI_API_KEY",
        ),
    ],
)
def test_create_client_missing_api_key_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    expected_fragment: str,
) -> None:
    _clear_llm_env(monkeypatch)

    with pytest.raises(ValueError, match=expected_fragment):
        llm_utils.create_client(model)


@pytest.mark.parametrize(
    ("model", "env_values", "expected_api_key", "expected_base_url", "expected_model"),
    [
        ("gpt-4o-mini", {"OPENAI_API_KEY": "openai-key"}, "openai-key", None, "gpt-4o-mini"),
        (
            "deepseek-chat",
            {"TOGETHER_API_KEY": "together-key"},
            "together-key",
            "https://api.together.xyz/v1",
            "deepseek-ai/DeepSeek-V3.1",
        ),
        (
            "deepseek-reasoner",
            {"TOGETHERAI_API_KEY": "together-key"},
            "together-key",
            "https://api.together.xyz/v1",
            "deepseek-ai/DeepSeek-R1",
        ),
        (
            "qwen-plus",
            {"QWEN_API_KEY": "qwen-key"},
            "qwen-key",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen-plus",
        ),
        (
            "qwen-max",
            {"DASHSCOPE_API_KEY": "dashscope-key"},
            "dashscope-key",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen-max",
        ),
        (
            "moonshot-v1-8k",
            {"MOONSHOT_API_KEY": "moonshot-key"},
            "moonshot-key",
            "https://api.moonshot.cn/v1",
            "moonshot-v1-8k",
        ),
        (
            "grok-2-latest",
            {"XAI_API_KEY": "xai-key"},
            "xai-key",
            "https://api.x.ai/v1",
            "grok-2-latest",
        ),
        (
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            {"TOGETHERAI_API_KEY": "together-key"},
            "together-key",
            "https://api.together.xyz/v1",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        ),
    ],
)
def test_create_client_openai_compatible_provider_uses_expected_base_url(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    env_values: dict[str, str],
    expected_api_key: str,
    expected_base_url: str | None,
    expected_model: str,
) -> None:
    _clear_llm_env(monkeypatch)
    for env_var, value in env_values.items():
        monkeypatch.setenv(env_var, value)

    openai_ctor = Mock(return_value="client")
    monkeypatch.setattr(llm_utils.openai, "OpenAI", openai_ctor)

    client, resolved_model = llm_utils.create_client(model)

    assert client == "client"
    assert resolved_model == expected_model

    expected_kwargs = {"api_key": expected_api_key}
    if expected_base_url:
        expected_kwargs["base_url"] = expected_base_url

    openai_ctor.assert_called_once_with(**expected_kwargs)


@pytest.mark.parametrize(
    ("model", "api_env", "base_url_env", "override_url"),
    [
        (
            "deepseek-chat",
            "TOGETHER_API_KEY",
            "TOGETHER_BASE_URL",
            "https://proxy.example/together",
        ),
        (
            "qwen-plus",
            "QWEN_API_KEY",
            "QWEN_BASE_URL",
            "https://proxy.example/qwen/v1",
        ),
        (
            "moonshot-v1-8k",
            "KIMI_API_KEY",
            "KIMI_BASE_URL",
            "https://proxy.example/kimi/v1",
        ),
        ("grok-2-latest", "XAI_API_KEY", "XAI_BASE_URL", "https://proxy.example/xai/v1"),
        (
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "TOGETHER_API_KEY",
            "TOGETHER_BASE_URL",
            "https://proxy.example/together/v1",
        ),
    ],
)
def test_create_client_openai_compatible_provider_honors_base_url_override(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    api_env: str,
    base_url_env: str,
    override_url: str,
) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv(api_env, "provider-key")
    monkeypatch.setenv(base_url_env, override_url)

    openai_ctor = Mock(return_value="client")
    monkeypatch.setattr(llm_utils.openai, "OpenAI", openai_ctor)

    client, resolved_model = llm_utils.create_client(model)

    assert client == "client"
    if model == "deepseek-chat":
        assert resolved_model == "deepseek-ai/DeepSeek-V3.1"
    else:
        assert resolved_model == model
    openai_ctor.assert_called_once_with(api_key="provider-key", base_url=override_url)


def test_get_response_from_llm_openai_constructs_expected_request() -> None:
    create_mock = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="openai-output"))]
        )
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )

    text, history = llm_utils.get_response_from_llm(
        msg="Generate idea",
        client=client,
        model="gpt-4o-mini",
        system_message="You are a scientist.",
        msg_history=[{"role": "assistant", "content": "prior"}],
        temperature=0.2,
    )

    assert text == "openai-output"
    assert history[-1] == {"role": "assistant", "content": "openai-output"}

    create_mock.assert_called_once()
    assert create_mock.call_args.kwargs == {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a scientist."},
            {"role": "assistant", "content": "prior"},
            {"role": "user", "content": "Generate idea"},
        ],
        "n": 1,
        "stop": None,
        "seed": 0,
        "temperature": 0.2,
        "max_tokens": llm_utils.MAX_NUM_TOKENS,
    }


def test_get_response_from_llm_non_openai_openai_compatible_uses_minimal_request() -> None:
    create_mock = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="deepseek-output"))]
        )
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )

    text, history = llm_utils.get_response_from_llm(
        msg="Generate idea",
        client=client,
        model="deepseek-chat",
        system_message="You are a scientist.",
        msg_history=[{"role": "assistant", "content": "prior"}],
        temperature=0.2,
    )

    assert text == "deepseek-output"
    assert history[-1] == {"role": "assistant", "content": "deepseek-output"}

    create_mock.assert_called_once_with(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a scientist."},
            {"role": "assistant", "content": "prior"},
            {"role": "user", "content": "Generate idea"},
        ],
        temperature=0.2,
        max_tokens=llm_utils.MAX_NUM_TOKENS,
    )


def test_get_response_from_llm_anthropic_constructs_expected_request() -> None:
    create_mock = Mock(
        return_value=SimpleNamespace(content=[SimpleNamespace(text="anthropic-output")])
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))

    text, history = llm_utils.get_response_from_llm(
        msg="Generate idea",
        client=client,
        model="claude-3-5-sonnet-20241022",
        system_message="You are a scientist.",
        temperature=0.3,
    )

    assert text == "anthropic-output"
    assert history[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "anthropic-output"}],
    }

    create_mock.assert_called_once_with(
        model="claude-3-5-sonnet-20241022",
        max_tokens=llm_utils.MAX_NUM_TOKENS,
        temperature=0.3,
        system="You are a scientist.",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Generate idea"}],
            }
        ],
    )


def test_get_response_from_llm_gemini_constructs_expected_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_utils, "GenerationConfig", lambda **kwargs: kwargs)
    generate_mock = Mock(return_value=SimpleNamespace(text="gemini-output"))
    client = SimpleNamespace(generate_content=generate_mock)

    text, history = llm_utils.get_response_from_llm(
        msg="Generate idea",
        client=client,
        model="gemini-2.5-flash",
        system_message="You are a scientist.",
        msg_history=[{"role": "user", "content": "prior"}],
        temperature=0.4,
    )

    assert text == "gemini-output"
    assert history[-1] == {"role": "assistant", "content": "gemini-output"}

    generate_mock.assert_called_once_with(
        contents=[
            {"role": "system", "parts": "You are a scientist."},
            {"role": "user", "parts": "prior"},
            {"role": "user", "parts": "Generate idea"},
        ],
        generation_config={
            "temperature": 0.4,
            "max_output_tokens": llm_utils.MAX_NUM_TOKENS,
            "candidate_count": 1,
        },
    )
