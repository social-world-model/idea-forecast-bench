"""
Prompt Registry: versioned prompt template loader and strategy policy resolver.

Prompts live on disk as immutable artifacts under:
    backend/prompts/<prompt_id>/<version>.txt

Policy defaults:
    model_id:         gpt-4o-mini
    temperature:      0.7
    max_tokens:       1024
    timeout_seconds:  30

Usage:
    from backend.prompt_registry import get_prompt_policy, get_prompt_template, list_prompts
"""

from __future__ import annotations

import pathlib
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"

# ---------------------------------------------------------------------------
# Per-prompt policy overrides (extend here to add new prompts / versions)
# Each entry key: "<prompt_id>@<version>"
# ---------------------------------------------------------------------------

_POLICY_OVERRIDES: dict[str, dict[str, Any]] = {
    "llm_baseline@v1": {
        "model_id": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1024,
        "timeout_seconds": 30,
    },
}

# ---------------------------------------------------------------------------
# Defaults applied when a prompt_id@version is registered but has no override
# ---------------------------------------------------------------------------

_DEFAULT_POLICY: dict[str, Any] = {
    "model_id": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 1024,
    "timeout_seconds": 30,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_prompt_template(prompt_id: str, version: str) -> str:
    """Return the raw template string for *prompt_id* at *version*.

    Raises:
        ValueError: if the prompt file does not exist.
    """
    prompt_path = _PROMPTS_DIR / prompt_id / f"{version}.txt"
    if not prompt_path.is_file():
        raise ValueError(
            f"Unknown prompt: prompt_id={prompt_id!r}, version={version!r}. "
            f"Expected file at {prompt_path}"
        )
    return prompt_path.read_text(encoding="utf-8")


def get_prompt_policy(prompt_id: str, version: str) -> dict[str, Any]:
    """Return the full policy object for *prompt_id* at *version*.

    The returned dict always contains:
        prompt_id, version, template, model_id, temperature,
        max_tokens, timeout_seconds

    Raises:
        ValueError: if the prompt file does not exist.
    """
    template = get_prompt_template(prompt_id, version)

    key = f"{prompt_id}@{version}"
    overrides = _POLICY_OVERRIDES.get(key, {})

    policy: dict[str, Any] = {**_DEFAULT_POLICY, **overrides}
    policy["prompt_id"] = prompt_id
    policy["version"] = version
    policy["template"] = template
    return policy


def list_prompts() -> list[str]:
    """Return all registered prompt identifiers as 'prompt_id@version' strings.

    Scans the prompts directory on disk — useful for debugging.
    """
    results: list[str] = []
    if not _PROMPTS_DIR.is_dir():
        return results
    for txt_file in sorted(_PROMPTS_DIR.rglob("*.txt")):
        # Structure: prompts/<prompt_id>/<version>.txt
        relative = txt_file.relative_to(_PROMPTS_DIR)
        parts = relative.parts
        if len(parts) == 2:
            pid = parts[0]
            ver = parts[1].removesuffix(".txt")
            results.append(f"{pid}@{ver}")
    return results
