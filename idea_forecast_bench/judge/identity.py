from __future__ import annotations

import hashlib

from idea_forecast_bench.judge.config import (
    JUDGE_MAX_TOKENS,
    JUDGE_SYSTEM,
    JUDGE_TEMPERATURE,
)
from idea_forecast_bench.models import IdeaPrediction
from idea_forecast_bench.similarity import _sanitize, idea_text


def pred_text(p: IdeaPrediction) -> str:
    # Canonical prediction serialization shared with the benchmark matcher.
    return _sanitize(idea_text(p))


def pred_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def judge_enable_thinking(judge_model: str) -> bool:
    """Whether the judge call disables 'thinking' mode for this model.

    Mirrors the per-model branch in call_judge (Qwen judges run with thinking
    disabled). Folded into the fingerprint because a state file produced with
    thinking on is not comparable to one with it off."""
    return "qwen" not in judge_model.lower()


def judge_fingerprint(judge_model: str) -> str:
    """12-hex fingerprint of the judge config that affects decisions.

    Namespaces the judge-decision cache so that changing --judge-model, the
    JUDGE_SYSTEM rubric, or the decode config (max_tokens / temperature /
    thinking) does not silently reuse decisions made under the old config when
    an existing state file is resumed."""
    h = hashlib.sha256()
    h.update(judge_model.encode())
    h.update(b"\x00")
    h.update(JUDGE_SYSTEM.encode())
    h.update(b"\x00")
    # Decode config: different max_tokens / temperature / thinking produce
    # non-comparable decisions, so they must change the cache namespace.
    h.update(
        repr(
            (JUDGE_MAX_TOKENS, JUDGE_TEMPERATURE, judge_enable_thinking(judge_model))
        ).encode()
    )
    return h.hexdigest()[:12]


def embed_fingerprint(embed_model: str) -> str:
    """12-hex fingerprint of the embedding model.

    Namespaces the prediction/paper vector caches so a state file embedded with
    one model is never mixed with vectors from another (incomparable geometry)."""
    return hashlib.sha256(embed_model.encode()).hexdigest()[:12]
