"""Run every window of one topic, then summarise the topic."""

from __future__ import annotations

from typing import Any

import openai

from live_idea_bench.backtest import (
    split_train_future_by_cutoff,
)
from live_idea_bench.judge.config import MAX_CHARS
from live_idea_bench.judge.embeddings import embed_batch
from live_idea_bench.judge.state import RunState
from live_idea_bench.judge.windows import process_window
from live_idea_bench.similarity import _sanitize, paper_text


def process_topic(
    topic_id: str,
    saved_topic: dict[str, Any],
    scoped_papers: list[Any],
    horizon_months: int,
    top_k: int,
    top_r: int,
    cluster_k: int,
    embed_client: openai.OpenAI,
    judge_client: openai.OpenAI,
    judge_model: str,
    state: RunState,
    workers: int,
    max_windows: int | None,
) -> tuple[str, dict[str, Any]]:
    saved_bt = saved_topic.get("backtest")
    if not saved_bt or not saved_bt.get("windows"):
        return topic_id, {
            "topic_name": saved_topic.get("topic_name", ""),
            "paper_count": 0,
            "backtest": None,
        }

    # Batch-embed all topic papers once (use state cache)
    missing_ids = [
        p.paper_id for p in scoped_papers if state.get_paper_vec(p.paper_id) is None
    ]
    missing_papers = [p for p in scoped_papers if p.paper_id in set(missing_ids)]
    if missing_papers:
        print(f"  [{topic_id}] embedding {len(missing_papers)} papers ...", flush=True)
        texts = [_sanitize(paper_text(p)[:MAX_CHARS]) for p in missing_papers]
        vecs = embed_batch(texts, embed_client)
        state.set_paper_vecs(
            list(zip([p.paper_id for p in missing_papers], vecs, strict=False))
        )

    # One lookup per paper: the comprehension this replaces called
    # get_paper_vec twice for every paper, once to test and once to store.
    paper_vecs: dict[str, list[float]] = {}
    for paper in scoped_papers:
        vec = state.get_paper_vec(paper.paper_id)
        if vec is not None:
            paper_vecs[paper.paper_id] = vec

    saved_windows = saved_bt["windows"]
    if max_windows is not None:
        saved_windows = saved_windows[:max_windows]

    windows_out: list[dict[str, Any]] = []
    for wi, w in enumerate(saved_windows):
        cutoff = w["cutoff_month"]

        if state.is_window_done(topic_id, cutoff):
            cached = state.get_window_output(topic_id, cutoff)
            if cached is not None:
                windows_out.append(cached)
            print(f"  [{topic_id}] window {cutoff} already done, skipping", flush=True)
            continue

        train, future, _, _ = split_train_future_by_cutoff(
            papers=scoped_papers,
            cutoff_month=cutoff,
            horizon_months=horizon_months,
            cutoff_date=w.get("cutoff_date"),
        )
        if not future:
            continue

        print(
            f"  [{topic_id}] window {wi + 1}/{len(saved_windows)} cutoff={cutoff} "
            f"future={len(future)} ...",
            flush=True,
        )

        window_result = process_window(
            window_data=w,
            future_papers=future,
            train_paper_ids=[p.paper_id for p in train],
            paper_vecs=paper_vecs,
            embed_client=embed_client,
            judge_client=judge_client,
            judge_model=judge_model,
            top_k=top_k,
            top_r=top_r,
            cluster_k=cluster_k,
            state=state,
            workers=workers,
        )
        windows_out.append(window_result)
        state.mark_window_done(topic_id, cutoff, window_result)

    if not windows_out:
        return topic_id, {
            "topic_name": saved_topic.get("topic_name", ""),
            "paper_count": len(scoped_papers),
            "backtest": None,
        }

    def _avg(metric: str) -> float:
        vals = [w["evaluation"][metric] for w in windows_out]
        mean: float = round(sum(vals) / len(vals), 4)
        return mean

    summary = {
        "windows": len(windows_out),
        "avg_hit_at_k": _avg("hit_at_k"),
        "avg_mrr": _avg("mrr"),
        "avg_precision_at_k": _avg("precision_at_k"),
        "avg_soft_score": _avg("soft_score"),
        "avg_cluster_coverage": _avg("cluster_coverage"),
        "avg_novelty": _avg("avg_novelty"),
    }
    print(
        f"  [{topic_id}] done — hit@k={summary['avg_hit_at_k']:.4f}  "
        f"mrr={summary['avg_mrr']:.4f}  coverage={summary['avg_cluster_coverage']:.4f}",
        flush=True,
    )
    return topic_id, {
        "topic_name": saved_topic.get("topic_name", ""),
        "paper_count": len(scoped_papers),
        "backtest": {"summary": summary, "windows": windows_out},
    }
