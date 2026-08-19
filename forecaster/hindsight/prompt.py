"""Prompt construction for hindsight innovation extraction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from live_idea_bench.models import PaperRecord

PROMPT_FILE = Path(__file__).parent.parent / "prompt" / "hindsight.yaml"

_PROMPT_CONFIG_CACHE: dict[str, Any] | None = None


def _load_prompt_config() -> dict[str, Any]:
    """Load hindsight prompt YAML (cached at module level)."""
    global _PROMPT_CONFIG_CACHE  # noqa: PLW0603
    if _PROMPT_CONFIG_CACHE is None:
        raw = PROMPT_FILE.read_text(encoding="utf-8")
        payload = yaml.safe_load(raw)
        if not isinstance(payload, dict):
            raise ValueError(
                f"hindsight.yaml must decode to a mapping, got {type(payload)}"
            )
        _PROMPT_CONFIG_CACHE = payload
    return _PROMPT_CONFIG_CACHE


def _build_context_summary(
    context_papers: list[PaperRecord],
    max_context_papers: int,
) -> str:
    """Return a compact numbered list of historical context snippets."""
    truncated = context_papers[:max_context_papers]
    if not truncated:
        return "(no historical context available)"
    lines: list[str] = []
    for i, paper in enumerate(truncated, start=1):
        keywords = ", ".join((paper.keywords or [])[:5]) or "n/a"
        summary = " ".join((paper.summary or "").split())[:240] or "(no summary available)"
        lines.append(
            f"{i}. {paper.title} ({paper.month})\n"
            f"   Keywords: {keywords}\n"
            f"   Summary: {summary}"
        )
    return "\n".join(lines)


def _format_reference_entry(entry: dict[str, Any]) -> str:
    title = str(
        entry.get("title")
        or entry.get("paper_title")
        or entry.get("name")
        or entry.get("text")
        or ""
    ).strip()
    authors_raw = entry.get("authors")
    authors = ""
    if isinstance(authors_raw, list):
        authors = ", ".join(str(author).strip() for author in authors_raw if str(author).strip())
    elif authors_raw is not None:
        authors = str(authors_raw).strip()
    year = str(entry.get("year") or entry.get("date") or "").strip()
    note = str(entry.get("context") or entry.get("note") or entry.get("reason") or "").strip()

    parts = [title or "(untitled reference)"]
    if authors:
        parts.append(f"authors={authors}")
    if year:
        parts.append(f"year={year}")
    if note:
        parts.append(f"note={note}")
    return "; ".join(parts)


def _build_future_grounding(
    future_paper: PaperRecord,
    *,
    max_reference_items: int = 12,
) -> str:
    """Render reference metadata for hindsight grounding."""
    entries: list[tuple[str, dict[str, Any]]] = []
    for reference in future_paper.references:
        entries.append(("reference", reference))
    for citation in future_paper.citations:
        entries.append(("citation", citation))

    if not entries:
        return "(no reference metadata available)"

    lines = [
        f"{idx}. [{label}] {_format_reference_entry(entry)}"
        for idx, (label, entry) in enumerate(entries[:max_reference_items], start=1)
    ]
    return "\n".join(lines)


def build_hindsight_prompt(
    future_paper: PaperRecord,
    context_papers: list[PaperRecord],
    *,
    max_context_papers: int = 15,
) -> tuple[str, str]:
    """Build (system_prompt, user_message) for hindsight extraction.

    Args:
        future_paper: The paper whose innovation to extract.
        context_papers: Historical papers available before cutoff
            (truncated to max_context_papers).
        max_context_papers: Max number of context papers to include.

    Returns:
        (system_prompt, user_message) tuple for LLM call.
    """
    config = _load_prompt_config()
    system_prompt: str = config["system_prompt"].strip()
    user_template: str = config["user_template"]

    context_summary = _build_context_summary(context_papers, max_context_papers)

    user_message = user_template.format(
        context_summary=context_summary,
        future_title=future_paper.title,
        future_abstract=future_paper.summary,
        future_grounding=_build_future_grounding(future_paper),
    )

    return system_prompt, user_message
