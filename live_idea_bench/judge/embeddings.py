"""Embedding calls and cosine retrieval for the judge pipeline."""

from __future__ import annotations

import math
import time

import openai

from live_idea_bench.judge.config import (
    BATCH_SIZE,
    EMBED_MODEL,
    MAX_EMBED_RETRY,
)


def embed_batch(
    texts: list[str],
    client: openai.OpenAI,
    model: str = EMBED_MODEL,
) -> list[list[float]]:
    results: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for attempt in range(MAX_EMBED_RETRY):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vecs = [
                    item.embedding for item in sorted(resp.data, key=lambda x: x.index)
                ]
                results.extend(vecs)
                break
            except Exception as exc:
                wait = 2**attempt
                if attempt < MAX_EMBED_RETRY - 1:
                    print(f"\n  [embed retry {attempt + 1}] {exc}", flush=True)
                    time.sleep(wait)
                else:
                    raise
    return results


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return max(0.0, min(1.0, dot / (na * nb))) if na and nb else 0.0


def top_r_candidates(
    pred_vec: list[float],
    paper_vecs: dict[str, list[float]],
    future_paper_ids: list[str],
    top_r: int,
) -> list[tuple[str, float]]:
    """Return top-R (paper_id, cosine_score) sorted descending."""
    scores = [
        (pid, cosine(pred_vec, paper_vecs[pid]))
        for pid in future_paper_ids
        if pid in paper_vecs
    ]
    scores.sort(key=lambda x: -x[1])
    return scores[:top_r]
