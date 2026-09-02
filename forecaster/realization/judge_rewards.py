from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from idea_forecast_bench.models import IdeaPrediction, PaperRecord
from idea_forecast_bench.similarity import idea_text, paper_text

if TYPE_CHECKING:  # pragma: no cover - typing-only import, kept off the hot path
    import openai


class SupportsEmbedding(Protocol):
    """Structural type of the injected embedder (see forecaster.embedding)."""

    def embed(self, texts: Iterable[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...


logger = logging.getLogger(__name__)

# Reuse the exact constants from examples/judge_eval.py so the GRPO
# training reward and the offline eval stay numerically aligned.
DEFAULT_TOP_R = 10
DEFAULT_CLUSTER_K = 5
MATCH_PM_THRESHOLD = 5
MATCH_S_THRESHOLD = 2
COVERAGE_SIM_GATE = 0.5

JUDGE_SYSTEM = (
    "You score how strongly a research-idea prediction matches a "
    "real paper along three axes. Return ONLY the four labeled lines "
    "in order: PROBLEM_MATCH: <0-3>, METHOD_MATCH: <0-3>, "
    "SPECIFICITY: <0-3>, REASONING: <one short sentence>."
)
JUDGE_USER_TMPL = """Prediction:
Title: {pred_title}
Rationale: {pred_rationale}
Approach: {pred_approach}
Key Terms: {pred_terms}

Paper:
Title: {paper_title}
Abstract: {paper_abstract}

Score PROBLEM_MATCH (same core problem? 0=unrelated, 3=identical),
METHOD_MATCH (same mechanism? 0=different, 3=identical),
SPECIFICITY (precise novel claim vs vague keywords? 0=keyword-only, 3=precise).
"""

SCORE_RE = re.compile(r"^\s*(PROBLEM_MATCH|METHOD_MATCH|SPECIFICITY)\s*:\s*(\d+)", re.M)


@dataclass
class JudgeScores:
    problem: int
    method: int
    specificity: int

    @property
    def is_match(self) -> bool:
        return (
            self.problem + self.method >= MATCH_PM_THRESHOLD
            and self.specificity >= MATCH_S_THRESHOLD
        )

    @property
    def soft(self) -> float:
        return round((self.problem + self.method + self.specificity) / 9.0, 4)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _embed_prediction(
    prediction: IdeaPrediction, embedder: SupportsEmbedding
) -> list[float]:
    return embedder.embed_one(idea_text(prediction))


def _embed_papers(
    papers: Iterable[PaperRecord], embedder: SupportsEmbedding
) -> list[list[float]]:
    texts = [paper_text(p) for p in papers]
    return embedder.embed(texts) if texts else []


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------
def compute_novelty_reward(
    prediction: IdeaPrediction,
    train_papers: list[PaperRecord],
    *,
    embedder: SupportsEmbedding,
) -> float:
    """1 - max cosine similarity to any training paper. Higher = more novel."""
    if not train_papers:
        return 1.0
    pred_vec = _embed_prediction(prediction, embedder)
    train_vecs = _embed_papers(train_papers, embedder)
    if not train_vecs:
        return 1.0
    max_sim = max(_cosine(pred_vec, tv) for tv in train_vecs)
    return round(max(0.0, min(1.0, 1.0 - max_sim)), 4)


# ---------------------------------------------------------------------------
# Cluster coverage (per-rollout proxy)
# ---------------------------------------------------------------------------
def compute_coverage_reward(
    prediction: IdeaPrediction,
    future_papers: list[PaperRecord],
    *,
    embedder: SupportsEmbedding,
    k: int = DEFAULT_CLUSTER_K,
    sim_gate: float = COVERAGE_SIM_GATE,
) -> float:
    """Per-rollout proxy for cluster coverage.

    For one prediction, partition the future papers into ``k`` KMeans
    clusters, compute the prediction's max cosine to each cluster, and
    average the contributions where similarity exceeds ``sim_gate``.

    This preserves the original metric's "did the prediction reach this
    cluster?" semantic at the single-prediction granularity GRPO needs.
    """
    if not future_papers:
        return 0.0
    future_vecs = _embed_papers(future_papers, embedder)
    if not future_vecs:
        return 0.0
    pred_vec = _embed_prediction(prediction, embedder)

    k_eff = min(k, len(future_vecs))
    try:
        import numpy as np
        from sklearn.cluster import KMeans

        X = np.asarray(future_vecs, dtype=np.float32)
        labels = KMeans(n_clusters=k_eff, n_init=5, random_state=0).fit_predict(X)
    except Exception:
        # Fallback: one cluster per paper.
        labels = list(range(len(future_vecs)))
        k_eff = len(future_vecs)

    by_cluster: dict[int, list[float]] = {}
    for i, vec in enumerate(future_vecs):
        by_cluster.setdefault(int(labels[i]), []).append(_cosine(pred_vec, vec))
    contributions = []
    for _cid, sims in by_cluster.items():
        max_sim = max(sims)
        contributions.append(max_sim if max_sim >= sim_gate else 0.0)
    if not contributions:
        return 0.0
    return round(sum(contributions) / k_eff, 4)


# ---------------------------------------------------------------------------
# Soft reward (local LLM judge over retrieved candidates)
# ---------------------------------------------------------------------------
def _parse_judge_response(content: str) -> JudgeScores:
    scores: dict[str, int] = {}
    for m in SCORE_RE.finditer(content or ""):
        try:
            scores[m.group(1).upper()] = int(m.group(2))
        except ValueError:
            continue
    return JudgeScores(
        problem=max(0, min(3, scores.get("PROBLEM_MATCH", 0))),
        method=max(0, min(3, scores.get("METHOD_MATCH", 0))),
        specificity=max(0, min(3, scores.get("SPECIFICITY", 0))),
    )


def _call_judge(
    prediction: IdeaPrediction,
    paper: PaperRecord,
    *,
    # OpenAI-compatible chat client; duck-typed so tests can inject a fake
    # without depending on the concrete openai.OpenAI resource classes.
    judge_client: Any,
    judge_model: str,
    max_retries: int = 2,
) -> JudgeScores | None:
    user_msg = JUDGE_USER_TMPL.format(
        pred_title=prediction.title,
        pred_rationale=(prediction.rationale or "")[:600],
        pred_approach=(prediction.approach or "")[:400],
        pred_terms=", ".join(prediction.key_terms or []),
        paper_title=paper.title,
        paper_abstract=(paper.summary or "")[:800],
    )
    last_exc: Exception | None = None
    for _ in range(max_retries):
        try:
            resp = judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=120,
                # Reasoning judges (e.g. Qwen3.5) would otherwise spend the whole
                # token budget on a <think> block and emit no parseable scores.
                # No-op for non-reasoning judges (Qwen2.5).
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = resp.choices[0].message.content or ""
            return _parse_judge_response(content)
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        logger.warning("Judge call failed after retries: %s", last_exc)
    return None


def compute_soft_reward(
    prediction: IdeaPrediction,
    future_papers: list[PaperRecord],
    *,
    embedder: SupportsEmbedding,
    judge_client: Any,
    judge_model: str,
    top_r: int = DEFAULT_TOP_R,
) -> float:
    """Retrieve top-R future papers, judge each, return best matched score.

    Returns the soft score of the highest-scoring matched candidate, or 0.0
    if no candidate passes the match thresholds. This is the per-prediction
    analogue of ``examples/judge_eval.py``'s aggregate ``soft_score``.
    """
    if not future_papers:
        return 0.0
    future_vecs = _embed_papers(future_papers, embedder)
    if not future_vecs:
        return 0.0
    pred_vec = _embed_prediction(prediction, embedder)

    scored = [(i, _cosine(pred_vec, vec)) for i, vec in enumerate(future_vecs)]
    scored.sort(key=lambda x: -x[1])
    candidates = [future_papers[i] for i, _ in scored[:top_r]]

    best_soft = 0.0
    for cand in candidates:
        result = _call_judge(
            prediction,
            cand,
            judge_client=judge_client,
            judge_model=judge_model,
        )
        if result is None:
            continue
        if result.is_match:
            best_soft = max(best_soft, result.soft)
    return round(best_soft, 4)


# ---------------------------------------------------------------------------
# Lazy judge-client singleton driven by env vars.
# ---------------------------------------------------------------------------
_JUDGE_CLIENT: openai.OpenAI | None = None


def get_default_judge_client() -> openai.OpenAI:
    """Return a process-wide OpenAI-compatible client built from env vars.

    Reads ``JUDGE_BASE_URL`` (default ``http://localhost:8765/v1``) and
    ``JUDGE_API_KEY`` (default ``EMPTY``) so the same code path works
    against a local vLLM OpenAI-compatible server.
    """
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is not None:
        return _JUDGE_CLIENT
    import openai

    base_url = os.environ.get("JUDGE_BASE_URL", "http://localhost:8765/v1")
    api_key = os.environ.get("JUDGE_API_KEY", "EMPTY")
    _JUDGE_CLIENT = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)
    return _JUDGE_CLIENT


def get_default_judge_model() -> str:
    return os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
