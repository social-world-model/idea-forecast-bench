"""Rubric data model + generation prompt scaffolding.

A rubric R is a small structured object describing the *match criteria*
that count as "an emerging research idea in topic T at this cutoff".
The rubric is consumed by:
  * the rubric-conditioned judge (forecaster/foresight/judge.py),
  * the reward function (Phase 4),
  * the Phase-2 AUC validation pass.

Generation is done by a frozen strong LLM. We keep the generation prompt
in this file so test code can inspect it without touching live_idea_bench.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Rubric:
    """Structured match criteria for one topic / cutoff slice."""

    topic_id: str
    cutoff_t: str
    criteria: tuple[str, ...]
    must_not: tuple[str, ...] = ()
    examples_positive: tuple[str, ...] = ()
    examples_negative: tuple[str, ...] = ()
    operator_focus: tuple[str, ...] = ()      # closed operator ids this rubric is tuned for, optional
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "cutoff_t": self.cutoff_t,
            "criteria": list(self.criteria),
            "must_not": list(self.must_not),
            "examples_positive": list(self.examples_positive),
            "examples_negative": list(self.examples_negative),
            "operator_focus": list(self.operator_focus),
            "version": self.version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Rubric:
        return cls(
            topic_id=str(payload["topic_id"]),
            cutoff_t=str(payload.get("cutoff_t", "")),
            criteria=tuple(str(c) for c in payload.get("criteria") or ()),
            must_not=tuple(str(c) for c in payload.get("must_not") or ()),
            examples_positive=tuple(str(c) for c in payload.get("examples_positive") or ()),
            examples_negative=tuple(str(c) for c in payload.get("examples_negative") or ()),
            operator_focus=tuple(str(c) for c in payload.get("operator_focus") or ()),
            version=int(payload.get("version", 1)),
            metadata=dict(payload.get("metadata") or {}),
        )

    def as_judge_block(self) -> str:
        """Render the rubric as a plain-text block to inline into the judge prompt."""
        lines: list[str] = [f"Topic: {self.topic_id}"]
        if self.operator_focus:
            lines.append("Operator focus: " + ", ".join(self.operator_focus))
        if self.criteria:
            lines.append("Match criteria (an idea should satisfy these to count as a positive match):")
            for c in self.criteria:
                lines.append(f"  - {c}")
        if self.must_not:
            lines.append("Disqualifying criteria (presence of any of these is a hard no-match):")
            for c in self.must_not:
                lines.append(f"  - {c}")
        return "\n".join(lines)


def save_rubric(rubric: Rubric, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rubric.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_rubric(path: str | Path) -> Rubric:
    return Rubric.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def load_rubrics_dir(rubrics_dir: str | Path) -> dict[str, Rubric]:
    """Load every {topic}.json under a directory, keyed by topic_id."""
    d = Path(rubrics_dir)
    if not d.exists():
        return {}
    out: dict[str, Rubric] = {}
    for f in sorted(d.glob("*.json")):
        try:
            r = load_rubric(f)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("failed to load rubric %s: %s", f, exc)
            continue
        out[r.topic_id] = r
    return out


# ---------------------------------------------------------------- generation prompt

RUBRIC_SYSTEM_PROMPT = (
    "You are a senior research-program officer. Your job is to write a precise rubric "
    "that an LLM judge can use to decide whether a given research idea matches the kind "
    "of work that emerged in a specific topic during a specific time window. "
    "Output strict JSON with keys: criteria (list of 3-7 short bullet points), "
    "must_not (list of 0-4 disqualifying conditions). No prose outside the JSON."
)


def build_rubric_generation_prompt(
    topic_id: str,
    cutoff_t: str,
    positive_examples: Iterable[str],
    negative_examples: Iterable[str] = (),
    operator_focus: Iterable[str] = (),
) -> str:
    """Render the user-side prompt for rubric generation."""
    pos_block = "\n".join(f"  - {e}" for e in positive_examples) or "  (none provided)"
    neg_block = "\n".join(f"  - {e}" for e in negative_examples) or "  (none provided)"
    focus = ", ".join(operator_focus) or "(none)"
    return (
        f"Topic: {topic_id}\n"
        f"Cutoff: {cutoff_t}\n"
        f"Operator focus: {focus}\n\n"
        f"Positive examples (ideas extracted from papers that emerged AFTER the cutoff):\n"
        f"{pos_block}\n\n"
        f"Negative examples (ideas describing work that already existed BEFORE the cutoff):\n"
        f"{neg_block}\n\n"
        "Write a rubric that — when read by an LLM judge — lets it tell positives from "
        "negatives. Be concrete: prefer 'must mention X-style attention' over 'should be novel'. "
        "Return JSON only, with keys `criteria` and `must_not`."
    )


def parse_rubric_response(raw_text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Parse the LLM JSON response into (criteria, must_not). Forgiving on code fences."""
    import re

    s = raw_text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    data = json.loads(s)
    if not isinstance(data, dict):
        raise ValueError(f"rubric response must be a JSON object, got {type(data).__name__}")
    criteria = tuple(str(c).strip() for c in (data.get("criteria") or ()) if str(c).strip())
    must_not = tuple(str(c).strip() for c in (data.get("must_not") or ()) if str(c).strip())
    if not criteria:
        raise ValueError("rubric must declare at least one criterion")
    return criteria, must_not


def stamp_metadata(model: str | None = None, **extra: Any) -> dict[str, Any]:
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if model:
        meta["model"] = model
    meta.update(extra)
    return meta
