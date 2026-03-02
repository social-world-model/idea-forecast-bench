"""
Tests for backend.prompt_registry

Run via:
    conda run -n poetry-env python -m pytest tests/test_prompt_registry.py -v
"""

from __future__ import annotations

import pytest

from backend.prompt_registry import (
    get_prompt_policy,
    get_prompt_template,
    list_prompts,
)

# ---------------------------------------------------------------------------
# get_prompt_template
# ---------------------------------------------------------------------------


class TestGetPromptTemplate:
    def test_happy_path_returns_string(self) -> None:
        template = get_prompt_template("llm_baseline", "v1")
        assert isinstance(template, str)
        assert len(template) > 0

    def test_template_contains_recognizable_phrase(self) -> None:
        template = get_prompt_template("llm_baseline", "v1")
        # The v1 template instructs the LLM to generate research ideas
        assert "research" in template.lower()

    def test_template_contains_json_instruction(self) -> None:
        template = get_prompt_template("llm_baseline", "v1")
        assert "JSON" in template

    def test_unknown_prompt_id_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            get_prompt_template("nonexistent_prompt", "v1")
        assert "nonexistent_prompt" in str(exc_info.value)
        assert "v1" in str(exc_info.value)

    def test_unknown_version_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            get_prompt_template("llm_baseline", "v999")
        assert "llm_baseline" in str(exc_info.value)
        assert "v999" in str(exc_info.value)

    def test_unknown_prompt_and_version_deterministic_message(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            get_prompt_template("bad_prompt", "bad_version")
        msg = str(exc_info.value)
        assert "bad_prompt" in msg
        assert "bad_version" in msg


# ---------------------------------------------------------------------------
# get_prompt_policy
# ---------------------------------------------------------------------------


class TestGetPromptPolicy:
    def test_happy_path_returns_dict(self) -> None:
        policy = get_prompt_policy("llm_baseline", "v1")
        assert isinstance(policy, dict)

    def test_required_keys_present(self) -> None:
        policy = get_prompt_policy("llm_baseline", "v1")
        required_keys = {
            "prompt_id",
            "version",
            "template",
            "model_id",
            "temperature",
            "max_tokens",
            "timeout_seconds",
        }
        assert required_keys.issubset(policy.keys()), (
            f"Missing keys: {required_keys - policy.keys()}"
        )

    def test_prompt_id_and_version_match_inputs(self) -> None:
        policy = get_prompt_policy("llm_baseline", "v1")
        assert policy["prompt_id"] == "llm_baseline"
        assert policy["version"] == "v1"

    def test_model_id_is_string(self) -> None:
        policy = get_prompt_policy("llm_baseline", "v1")
        assert isinstance(policy["model_id"], str)
        assert len(policy["model_id"]) > 0

    def test_model_id_default_value(self) -> None:
        policy = get_prompt_policy("llm_baseline", "v1")
        assert policy["model_id"] == "gpt-4o-mini"

    def test_temperature_is_float(self) -> None:
        policy = get_prompt_policy("llm_baseline", "v1")
        assert isinstance(policy["temperature"], float)

    def test_temperature_default_value(self) -> None:
        policy = get_prompt_policy("llm_baseline", "v1")
        assert policy["temperature"] == 0.7

    def test_max_tokens_is_int(self) -> None:
        policy = get_prompt_policy("llm_baseline", "v1")
        assert isinstance(policy["max_tokens"], int)

    def test_max_tokens_default_value(self) -> None:
        policy = get_prompt_policy("llm_baseline", "v1")
        assert policy["max_tokens"] == 1024

    def test_timeout_seconds_is_int(self) -> None:
        policy = get_prompt_policy("llm_baseline", "v1")
        assert isinstance(policy["timeout_seconds"], int)

    def test_timeout_seconds_default_value(self) -> None:
        policy = get_prompt_policy("llm_baseline", "v1")
        assert policy["timeout_seconds"] == 30

    def test_template_contains_recognizable_phrase(self) -> None:
        policy = get_prompt_policy("llm_baseline", "v1")
        assert "research" in policy["template"].lower()

    def test_unknown_prompt_id_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            get_prompt_policy("unknown_prompt", "v1")
        assert "unknown_prompt" in str(exc_info.value)

    def test_unknown_version_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            get_prompt_policy("llm_baseline", "v0")
        assert "v0" in str(exc_info.value)

    def test_unknown_prompt_and_version_deterministic_message(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            get_prompt_policy("ghost_prompt", "ghost_version")
        msg = str(exc_info.value)
        assert "ghost_prompt" in msg
        assert "ghost_version" in msg

    def test_policy_is_independent_copy(self) -> None:
        """Mutating returned policy must not affect subsequent calls."""
        policy1 = get_prompt_policy("llm_baseline", "v1")
        policy1["model_id"] = "mutated-model"
        policy2 = get_prompt_policy("llm_baseline", "v1")
        assert policy2["model_id"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# list_prompts
# ---------------------------------------------------------------------------


class TestListPrompts:
    def test_returns_list(self) -> None:
        result = list_prompts()
        assert isinstance(result, list)

    def test_llm_baseline_v1_in_list(self) -> None:
        result = list_prompts()
        assert "llm_baseline@v1" in result

    def test_entries_follow_at_format(self) -> None:
        for entry in list_prompts():
            assert "@" in entry, f"Expected 'prompt_id@version', got: {entry!r}"
