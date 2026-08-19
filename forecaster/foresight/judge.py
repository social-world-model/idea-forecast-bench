"""Rubric-conditioned judge for the new reward + Phase-2 validation.

The judge has one public method:

    score(idea_text, candidate_text, rubric) -> JudgeResult

It wraps the same LLM client the benchmark already uses
(`live_idea_bench.llm.get_response_from_llm`) so the training-time reward
shares its scoring backbone with the eval-time scorer.

A pluggable `ScorerFn` is exposed so unit tests can inject a stub.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from forecaster.foresight.rubric import Rubric

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- types


@dataclass
class JudgeResult:
    score: float  # in [0, 1]
    reasoning: str
    raw_text: str
    engine: str  # e.g. "llm:gpt-4o" or "stub:fixed"


class ScorerFn(Protocol):
    """A function that takes (system_prompt, user_prompt) -> raw model text."""

    def __call__(self, system_prompt: str, user_prompt: str) -> str: ...


JUDGE_SYSTEM_PROMPT = (
    "You are a precise rubric-conditioned similarity evaluator. "
    "You read a topic-specific rubric and decide whether a proposed Research Idea "
    "matches an Academic Paper Content under the rubric's criteria. "
    "Always output your score as a single line beginning with 'Score:', "
    "followed by a one- to two-sentence rationale on a 'Reasoning:' line."
)

# Locked penalty for triggering ANY 'must_not' criterion. We deliberately
# do NOT expose this as a knob — see [[foresight-rl-plan-decisions]]: a
# tuned penalty turns M2 from a real gate into a self-fulfilling check.
MUST_NOT_PENALTY: float = 0.2


def build_judge_user_prompt(
    idea_text: str,
    candidate_text: str,
    rubric: Rubric,
    *,
    idea_max_chars: int = 2000,
    candidate_max_chars: int = 4000,
) -> str:
    """Render the rubric-conditioned judge prompt."""
    idea = (idea_text or "").strip()[:idea_max_chars]
    cand = (candidate_text or "").strip()[:candidate_max_chars]
    return (
        f"=== RUBRIC ===\n{rubric.as_judge_block()}\n\n"
        f"=== RESEARCH IDEA ===\n{idea}\n\n"
        f"=== ACADEMIC PAPER CONTENT ===\n{cand}\n\n"
        f"=== TASK ===\n"
        "Decide whether the Research Idea matches the kind of work described by the "
        "Academic Paper Content **under the rubric**. Reward criterion satisfaction, "
        "penalize criterion violations.\n\n"
        "How to score:\n"
        "  1. Score on a 0–1 scale based on how well the idea satisfies the rubric's "
        "match criteria; high coverage of criteria → high score.\n"
        f"  2. If ANY 'must_not' criterion is triggered, subtract a fixed "
        f"{MUST_NOT_PENALTY:.2f} penalty from the score. Apply this penalty AT MOST ONCE, "
        "regardless of how many 'must_not' criteria are triggered — do NOT accumulate, "
        "and do NOT treat a 'must_not' trigger as automatic disqualification.\n"
        "  3. Floor the final score at 0.0.\n\n"
        "Respond on exactly two lines:\n"
        "Score: <number in [0.0, 1.0]>\n"
        "Reasoning: <one or two sentences>\n"
    )


def parse_score(raw: str) -> tuple[float, str]:
    """Extract `Score:` and `Reasoning:` from the raw model text.

    Returns (clamped_score, reasoning). Score defaults to 0.0 if missing.
    """
    score = 0.0
    m = re.search(r"(?im)\**\s*score\s*\**\s*:?\s*\**\s*(0?\.\d+|[01](?:\.\d+)?)", raw)
    if m:
        try:
            score = float(m.group(1))
        except ValueError:
            score = 0.0
    else:
        # This is the training-reward / M2 path: a hard raise here would
        # crash the GRPO reward loop on a single malformed judge reply.
        # Warn + count via the log instead of silently passing, but keep
        # the 0.0 default (the documented contract; see
        # test_parse_score_extracts_clamped).
        logger.warning(
            "judge.parse_score: no parseable 'Score:' line in model output; "
            "defaulting to 0.0. Raw: %r",
            raw[:500],
        )
    score = max(0.0, min(1.0, score))
    reasoning = ""
    r = re.search(r"Reasoning:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
    if r:
        reasoning = r.group(1).strip().splitlines()[0].strip()
    return score, reasoning


# ---------------------------------------------------------------- live + stub scorers


def make_live_scorer(
    model_name: str | None = None,
    temperature: float = 0.0,
    *,
    base_url: str | None = None,
) -> ScorerFn:
    """Return a ScorerFn backed by an LLM API.

    Routing:
      * If `base_url` (or env `JUDGE_BASE_URL`) is set → talk to an
        OpenAI-compatible endpoint directly (used to point M2 + reward
        at the same self-hosted judge the trainer will use, e.g. SGLang
        serving Qwen3.5-9B-Instruct on http://localhost:30000/v1).
        Model is taken from `model_name` arg or env `JUDGE_MODEL`.
        API key from env `JUDGE_API_KEY` (defaults to "EMPTY" so local
        servers without auth still work).
      * Otherwise → use live_idea_bench.llm routing (gpt-4o / claude / gemini).

    The judge prompt itself is shared across backends so the M2 AUC
    numbers are directly comparable across judges.
    """
    import os

    resolved_base = (base_url or os.environ.get("JUDGE_BASE_URL", "")).strip() or None

    if resolved_base:
        import openai

        resolved_model = (
            model_name
            or os.environ.get("JUDGE_MODEL", "").strip()
            or "qwen3.5-9b-instruct"
        )
        api_key = os.environ.get("JUDGE_API_KEY", "EMPTY").strip() or "EMPTY"
        client = openai.OpenAI(api_key=api_key, base_url=resolved_base)

        def _scorer(system_prompt: str, user_prompt: str) -> str:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=256,
                # Reasoning judges (Qwen3.5) would otherwise spend the budget on
                # a <think> block; disabling keeps Score/Reasoning parseable+fast.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return response.choices[0].message.content or ""

        _scorer.__name__ = f"live_scorer_local:{resolved_model}@{resolved_base}"
        return _scorer

    # Fallback: live_idea_bench routing (OpenAI / Anthropic / Gemini hosted models).
    from live_idea_bench.config import load_runtime_config
    from live_idea_bench.llm import create_client, get_response_from_llm

    runtime_cfg = load_runtime_config()
    resolved = model_name or runtime_cfg.model_name
    client, resolved = create_client(resolved)

    # Same name as the local-endpoint scorer above, but the two definitions sit
    # on mutually exclusive branches (the one above returns before this point).
    def _scorer(system_prompt: str, user_prompt: str) -> str:  # type: ignore[no-redef]
        raw, _ = get_response_from_llm(
            msg=user_prompt,
            client=client,
            model=resolved,
            system_message=system_prompt,
            temperature=temperature,
        )
        return raw

    _scorer.__name__ = f"live_scorer:{resolved}"
    return _scorer


@dataclass
class StubScorer:
    """Deterministic scorer for tests. Returns a score derived from a function."""

    fn: Callable[[str, str], float]
    name: str = "stub"

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        # Pull the idea + candidate out of the rendered prompt for the test fn.
        idea = _extract_block(user_prompt, "=== RESEARCH IDEA ===")
        candidate = _extract_block(user_prompt, "=== ACADEMIC PAPER CONTENT ===")
        score = max(0.0, min(1.0, float(self.fn(idea, candidate))))
        return f"Score: {score:.4f}\nReasoning: stubbed"


def _extract_block(text: str, header: str) -> str:
    idx = text.find(header)
    if idx < 0:
        return ""
    rest = text[idx + len(header) :]
    next_idx = rest.find("\n===")
    return (rest[:next_idx] if next_idx >= 0 else rest).strip()


# ---------------------------------------------------------------- public API


class RubricJudge:
    """Stateless rubric-conditioned judge."""

    def __init__(self, scorer: ScorerFn, *, engine_label: str = "llm"):
        self.scorer = scorer
        self.engine_label = engine_label

    def score(
        self,
        idea_text: str,
        candidate_text: str,
        rubric: Rubric,
    ) -> JudgeResult:
        user_prompt = build_judge_user_prompt(idea_text, candidate_text, rubric)
        raw = self.scorer(JUDGE_SYSTEM_PROMPT, user_prompt)
        score, reasoning = parse_score(raw)
        engine_name = getattr(self.scorer, "__name__", None) or self.engine_label
        return JudgeResult(
            score=score, reasoning=reasoning, raw_text=raw, engine=engine_name
        )

    def score_batch(
        self,
        pairs: list[tuple[str, str]],
        rubric: Rubric,
    ) -> list[JudgeResult]:
        return [self.score(idea, cand, rubric) for idea, cand in pairs]


__all__ = [
    "JudgeResult",
    "ScorerFn",
    "RubricJudge",
    "StubScorer",
    "make_live_scorer",
    "build_judge_user_prompt",
    "parse_score",
    "JUDGE_SYSTEM_PROMPT",
]
