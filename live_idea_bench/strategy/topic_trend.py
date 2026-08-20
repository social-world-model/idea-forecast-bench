"""Topic Trend baseline strategy.

Groups historical papers into topic clusters via keyword co-occurrence, tracks
each cluster's trajectory, and uses an LLM to generate one prediction per
top-trending cluster.

Operates at the
*cluster* level — a topic is a coherent group of co-occurring keywords that
collectively describe a research direction.
"""

from __future__ import annotations

import contextlib
import json
from collections import Counter, defaultdict
from typing import Any

from live_idea_bench.llm import create_client, get_response_from_llm
from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.papers import add_months, month_to_index
from live_idea_bench.similarity import _sanitize
from live_idea_bench.strategy.base import IdeaStrategy

_STOP = {
    "cs",
    "ml",
    "ai",
    "arxiv",
    "machine learning",
    "deep learning",
    "neural network",
    "neural networks",
    "learning",
    "model",
    "models",
    "paper",
    "method",
    "methods",
    "approach",
    "based",
    "using",
    "new",
    "large",
    "language",
    "llm",
}

_DEFAULT_MODEL = "gpt-4o"
_MAX_CLUSTERS = 12
_COOCCUR_WINDOW = 1  # keywords within the same paper co-occur


def _clean_keyword(kw: str) -> str:
    return kw.lower().strip()


def _build_clusters(
    train_papers: list[PaperRecord],
    min_freq: int,
) -> list[list[str]]:
    """Build topic clusters from keyword co-occurrence.

    Each cluster is a group of keywords that frequently appear together.
    We use a greedy union-find approach: two keywords that co-occur in at
    least `min_freq` papers are merged into the same cluster.
    """
    keyword_freq: Counter[str] = Counter()
    cooccur: Counter[tuple[str, str]] = Counter()

    for paper in train_papers:
        kws = [
            _clean_keyword(k)
            for k in paper.keywords
            if _clean_keyword(k) not in _STOP and len(_clean_keyword(k)) > 2
        ]
        kws = list(dict.fromkeys(kws))  # deduplicate order-preserving
        for kw in kws:
            keyword_freq[kw] += 1
        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                pair = (min(kws[i], kws[j]), max(kws[i], kws[j]))
                cooccur[pair] += 1

    # Union-Find
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent.get(x, x)
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    vocab = sorted(kw for kw, cnt in keyword_freq.items() if cnt >= min_freq)
    vocab_set = set(vocab)
    for kw in vocab:
        parent.setdefault(kw, kw)
    for (a, b), cnt in sorted(cooccur.items()):
        if a in vocab_set and b in vocab_set and cnt >= min_freq:
            union(a, b)

    # Group by root
    groups: dict[str, list[str]] = defaultdict(list)
    for kw in vocab:
        groups[find(kw)].append(kw)

    # Sort members by frequency descending
    clusters = [
        sorted(members, key=lambda k: -keyword_freq[k]) for members in groups.values()
    ]
    # Sort clusters by total frequency
    clusters.sort(key=lambda c: -sum(keyword_freq[k] for k in c))
    return clusters[:_MAX_CLUSTERS]


def _score_cluster(
    cluster: list[str],
    train_papers: list[PaperRecord],
    cutoff_month: str,
    recent_months: int,
) -> float:
    """Score a cluster by its recent momentum (recent / overall ratio)."""
    cutoff_idx = month_to_index(cutoff_month)
    recent_start_idx = month_to_index(add_months(cutoff_month, -(recent_months - 1)))
    cluster_set = set(cluster)

    total = 0
    recent = 0
    for paper in train_papers:
        kws = {_clean_keyword(k) for k in paper.keywords}
        if kws & cluster_set:
            total += 1
            m = month_to_index(paper.month)
            if recent_start_idx <= m <= cutoff_idx:
                recent += 1

    if total == 0:
        return 0.0
    expected_recent = max(1, total // max(1, recent_months))
    return float(recent * 2 + (recent - expected_recent))


def _llm_predict_for_cluster(
    cluster: list[str],
    cluster_score: float,
    cutoff_month: str,
    top_k: int,
    rank_offset: int,
    client: object,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None = None,
) -> list[IdeaPrediction]:
    """Ask the LLM to generate `top_k` predictions for a topic cluster."""
    kw_list = ", ".join(_sanitize(k) for k in cluster[:8])
    prompt = (
        f"You are a research trend forecaster. The following keywords define a growing research topic cluster "
        f"(literature cutoff: {cutoff_month}):\n\n  Keywords: {kw_list}\n\n"
        f"Based on the trajectory of this topic, generate {top_k} concrete research idea(s) likely to appear "
        f"in the months following {cutoff_month}. Do NOT use knowledge from after {cutoff_month}.\n\n"
        f"Return JSON only:\n"
        f'[{{"title": "...", "rationale": "...", "approach": "...", "confidence": 0.0, "key_terms": []}}]'
    )
    system = (
        "You are a precise research forecasting assistant. "
        "Return only valid JSON matching the requested schema."
    )
    raw, _ = get_response_from_llm(
        msg=prompt,
        client=client,
        model=model,
        system_message=system,
        temperature=temperature if temperature is not None else 0.4,
        reasoning_effort=reasoning_effort,
    )

    # Parse
    items: list[dict[str, Any]] = []
    try:
        payload = json.loads(raw.strip())
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and "ideas" in payload:
            items = payload["ideas"]
    except json.JSONDecodeError:
        import re

        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            with contextlib.suppress(json.JSONDecodeError):
                items = json.loads(m.group())

    predictions: list[IdeaPrediction] = []
    for item in items[:top_k]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        predictions.append(
            IdeaPrediction(
                rank=rank_offset + len(predictions) + 1,
                title=title,
                rationale=str(item.get("rationale") or "").strip(),
                approach=str(item.get("approach") or "").strip(),
                score=round(min(1.0, max(0.0, cluster_score / 10.0)), 4),
                confidence=float(item.get("confidence") or 0.5),
                key_terms=list(item.get("key_terms") or cluster[:3]),
            )
        )
    return predictions


class TopicTrendStrategy(IdeaStrategy):
    """Topic Trend baseline.

    Clusters keywords by co-occurrence, ranks clusters by recent momentum,
    and uses an LLM to generate predictions for the top-k trending clusters.
    """

    name = "topic_trend"

    def __init__(
        self,
        model_name: str | None = None,
        recent_months: int = 3,
        min_keyword_freq: int = 2,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model_name = model_name or _DEFAULT_MODEL
        self.recent_months = recent_months
        self.min_keyword_freq = min_keyword_freq
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def generate(
        self,
        train_papers: list[PaperRecord],
        cutoff_month: str,
        top_k: int,
    ) -> list[IdeaPrediction]:
        if not train_papers:
            return []

        clusters = _build_clusters(train_papers, self.min_keyword_freq)
        if not clusters:
            return []

        # Score and rank clusters
        scored = [
            (
                cluster,
                _score_cluster(cluster, train_papers, cutoff_month, self.recent_months),
            )
            for cluster in clusters
        ]
        scored.sort(key=lambda x: -x[1])

        client, resolved_model = create_client(self.model_name)

        predictions: list[IdeaPrediction] = []
        # One prediction per cluster until we have top_k
        for cluster, score in scored:
            if len(predictions) >= top_k:
                break
            new_preds = _llm_predict_for_cluster(
                cluster=cluster,
                cluster_score=score,
                cutoff_month=cutoff_month,
                top_k=1,
                rank_offset=len(predictions),
                client=client,
                model=resolved_model,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
            )
            predictions.extend(new_preds)

        # Re-rank
        for idx, pred in enumerate(predictions[:top_k], start=1):
            pred.rank = idx
        return predictions[:top_k]
