from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend import llm_utils


def test_create_client_unsupported_model_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        llm_utils.create_client("not-a-supported-model")


@pytest.mark.parametrize(
    ("model", "env_var"),
    [
        ("gpt-4o-mini", "OPENAI_API_KEY"),
        ("claude-3-5-sonnet-20241022", "ANTHROPIC_API_KEY"),
        ("gemini-2.5-flash", "GOOGLE_API_KEY"),
    ],
)
def test_create_client_missing_api_key_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    env_var: str,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match=env_var):
        llm_utils.create_client(model)


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
