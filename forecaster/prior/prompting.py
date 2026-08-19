"""Shared prompt contract for prior training, sampling, and scoring."""

from __future__ import annotations

from pathlib import Path

import yaml

_PROMPT_YAML_PATH = (
    Path(__file__).resolve().parents[2] / "forecaster" / "prompt" / "prior_sft.yaml"
)


def load_prior_prompt_config() -> dict[str, str]:
    """Load the frozen prior prompt config."""
    raw = yaml.safe_load(_PROMPT_YAML_PATH.read_text(encoding="utf-8"))
    return {
        "system_prompt": str(raw["system_prompt"]).strip(),
        "input_template": str(raw["input_template"]).strip(),
    }


def render_prior_user_prompt(
    memory_summary: str, *, input_template: str | None = None
) -> str:
    """Render the user-visible memory prompt shared by train/infer."""
    template = input_template or load_prior_prompt_config()["input_template"]
    return template.format(memory_summary=memory_summary)


def render_prior_chat_transcript(
    *,
    system_prompt: str,
    user_prompt: str,
    assistant_text: str | None = None,
) -> str:
    """Render the chat transcript used by the causal LM."""
    prefix = (
        f"<system>\n{system_prompt}\n</system>\n"
        f"<user>\n{user_prompt}\n</user>\n"
        "<assistant>\n"
    )
    if assistant_text is None:
        return prefix
    return f"{prefix}{assistant_text}\n</assistant>"
