"""Score one (topic, cutoff) window: retrieve, judge, then aggregate."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import openai

from idea_forecast_bench.judge.config import MAX_CHARS
from idea_forecast_bench.judge.embeddings import embed_batch, top_r_candidates
from idea_forecast_bench.judge.identity import pred_hash, pred_text
from idea_forecast_bench.judge.metrics import cluster_coverage, novelty_score
from idea_forecast_bench.judge.protocol import call_judge
from idea_forecast_bench.judge.state import RunState
from idea_forecast_bench.models import IdeaPrediction
from idea_forecast_bench.similarity import _sanitize


def load_predictions(raw: list[dict[str, Any]]) -> list[IdeaPrediction]:
    return [
        IdeaPrediction(
            rank=p["rank"],
            title=p["title"],
            rationale=p.get("rationale", ""),
            approach=p.get("approach", ""),
            score=p.get("score", 0.0),
            confidence=p.get("confidence", 0.0),
            key_terms=p.get("key_terms", []),
        )
        for p in raw
    ]


def process_window(
    window_data: dict[str, Any],
    future_papers: list[Any],
    train_paper_ids: list[str],
    paper_vecs: dict[str, list[float]],
    embed_client: openai.OpenAI,
    judge_client: openai.OpenAI,
    judge_model: str,
    top_k: int,
    top_r: int,
    cluster_k: int,
    state: RunState,
    workers: int,
) -> dict[str, Any]:
    cutoff = window_data["cutoff_month"]
    predictions = load_predictions(window_data.get("predictions", []))[:top_k]
    future_paper_ids = [p.paper_id for p in future_papers]
    paper_lookup = {p.paper_id: p for p in future_papers}

    train_vecs = [paper_vecs[pid] for pid in train_paper_ids if pid in paper_vecs]

    per_pred_out: list[dict[str, Any]] = []
    used_paper_ids: set[str] = set()
    judge_calls = 0
    judge_parse_failures = 0

    for pred in predictions:
        pt = _sanitize(pred_text(pred))[:MAX_CHARS]
        ph = pred_hash(pt)

        # Embed prediction (cached)
        pred_vec = state.get_pred_vec(ph)
        if pred_vec is None:
            pred_vec = embed_batch([pt], embed_client)[0]
            state.set_pred_vec(ph, pred_vec)

        # Retrieve top-R candidates
        candidates = top_r_candidates(pred_vec, paper_vecs, future_paper_ids, top_r)

        # Judge all candidates in parallel
        # Loop variables bound as defaults: see the note in
        # idea_forecast_bench/similarity.py. The executor is drained within the
        # iteration, so this makes an existing guarantee explicit.
        def _judge_one(
            pid_score: tuple[str, float],
            _ph: str = ph,
            _pred: IdeaPrediction = pred,
        ) -> tuple[str, float, dict[str, Any]]:
            pid, score = pid_score
            cached = state.get_decision(_ph, pid)
            if cached is not None:
                return pid, score, cached
            paper = paper_lookup.get(pid)
            if paper is None:
                d = {
                    "match": False,
                    "problem_score": 0,
                    "method_score": 0,
                    "specificity_score": 0,
                    "reasoning": "paper not found",
                    "raw": "",
                }
            else:
                d = call_judge(
                    pred=_pred,
                    paper_title=getattr(paper, "title", ""),
                    paper_abstract=getattr(paper, "summary", ""),
                    judge_client=judge_client,
                    judge_model=judge_model,
                )
            state.set_decision(_ph, pid, d)
            return pid, score, d

        judge_results: dict[str, tuple[float, dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_judge_one, c): c for c in candidates}
            for fut in as_completed(futs):
                pid, score, decision = fut.result()
                judge_results[pid] = (score, decision)
                judge_calls += 1
                if decision.get("parse_failed"):
                    judge_parse_failures += 1

        # Find first non-duplicate match (process in rank order)
        matched_paper_id = None
        matched_reasoning = None
        matched_decision: dict[str, Any] = {}
        for pid, _embed_score in candidates:
            score, decision = judge_results[pid]
            if decision["match"] and pid not in used_paper_ids:
                matched_paper_id = pid
                matched_reasoning = decision["reasoning"]
                matched_decision = decision
                used_paper_ids.add(pid)
                break

        # Always record scores: from matched candidate if hit, else from top-1 candidate
        if matched_decision:
            rep_decision = matched_decision
        elif candidates and candidates[0][0] in judge_results:
            rep_decision = judge_results[candidates[0][0]][1]
        else:
            rep_decision = {
                "problem_score": 0,
                "method_score": 0,
                "specificity_score": 0,
            }

        novelty = novelty_score(pred_vec, train_vecs)

        per_pred_out.append(
            {
                "rank": pred.rank,
                "title": pred.title,
                "is_match": matched_paper_id is not None,
                "matched_paper_id": matched_paper_id,
                "matched_reasoning": matched_reasoning,
                "problem_score": rep_decision["problem_score"],
                "method_score": rep_decision["method_score"],
                "specificity_score": rep_decision["specificity_score"],
                "novelty": novelty,
                "top_candidates": [
                    {
                        "paper_id": pid,
                        "embed_score": round(judge_results[pid][0], 4),
                        "llm_match": judge_results[pid][1]["match"],
                        "problem_score": judge_results[pid][1]["problem_score"],
                        "method_score": judge_results[pid][1]["method_score"],
                        "specificity_score": judge_results[pid][1]["specificity_score"],
                        "reasoning": judge_results[pid][1]["reasoning"],
                    }
                    for pid, _ in candidates
                    if pid in judge_results
                ],
            }
        )

    matched_ranks = [p["rank"] for p in per_pred_out if p["is_match"]]
    hit_at_k = 1.0 if matched_ranks else 0.0
    mrr = 1.0 / matched_ranks[0] if matched_ranks else 0.0
    precision = len(matched_ranks) / top_k if top_k else 0.0

    # Soft score: average (problem + method + specificity) / 9 across matched
    # predictions. A matched prediction always has integer scores (parse_failed
    # decisions are match=False), but coerce None->0 defensively.
    matched_preds = [p for p in per_pred_out if p["is_match"]]
    soft_score = 0.0
    if matched_preds:
        soft_score = round(
            sum(
                (
                    (p.get("problem_score") or 0)
                    + (p.get("method_score") or 0)
                    + (p.get("specificity_score") or 0)
                )
                / 9.0
                for p in matched_preds
            )
            / len(matched_preds),
            4,
        )

    avg_novelty = (
        round(sum(p["novelty"] for p in per_pred_out) / len(per_pred_out), 4)
        if per_pred_out
        else 0.0
    )

    # Cluster coverage: how many future-paper clusters did any hit prediction cover?
    matched_ids = {p["matched_paper_id"] for p in per_pred_out if p["matched_paper_id"]}
    future_vecs = [paper_vecs[pid] for pid in future_paper_ids if pid in paper_vecs]
    coverage = cluster_coverage(future_vecs, matched_ids, future_paper_ids, cluster_k)

    return {
        "cutoff_month": cutoff,
        "cutoff_date": window_data.get("cutoff_date", ""),
        "future_end_month": window_data.get("future_end_month", ""),
        "train_papers": window_data.get("train_papers", 0),
        # arXiv IDs of the training-window papers, so the citation/coauthor
        # validity analyses can target the train community (not a global union).
        "train_paper_ids": list(train_paper_ids),
        "future_papers": len(future_papers),
        # Telemetry: fraction of judge calls whose score lines could not be
        # parsed (and were recorded as parse_failed rather than silently scored).
        # A high value invalidates the window — surfaced so it isn't hidden.
        "judge_calls": judge_calls,
        "judge_parse_failures": judge_parse_failures,
        "judge_parse_failure_rate": round(judge_parse_failures / judge_calls, 4)
        if judge_calls
        else 0.0,
        "evaluation": {
            "hit_at_k": round(hit_at_k, 4),
            "mrr": round(mrr, 4),
            "precision_at_k": round(precision, 4),
            "soft_score": soft_score,
            "cluster_coverage": coverage,
            "avg_novelty": avg_novelty,
        },
        "per_prediction": per_pred_out,
    }
