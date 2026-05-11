#!/usr/bin/env python3
"""LLM-as-Judge evaluation for LiveIdeaBench predictions.

Implements a retrieve → embed → LLM-judge pipeline:
  1. Embed predictions + future papers with voyage-3-large
  2. Retrieve top-R candidates per prediction (cosine similarity)
  3. Call LLM judge for each candidate (binary match/no-match)
  4. A prediction "hits" if any candidate passes the LLM judgment

Usage:
    python examples/llm_judge_eval.py \\
        --input-json logs/baselines/memory_prompting_raw.json \\
        --papers-dir data/csml_v2/raw_markdown \\
        --output logs/baselines/memory_prompting_llmjudge.json

    # Quick test on one topic (3 windows):
    python examples/llm_judge_eval.py \\
        --input-json logs/baselines/memory_prompting_raw.json \\
        --papers-dir data/csml_v2/raw_markdown \\
        --output /tmp/mp_judge_test.json \\
        --topics llm_pretraining --max-windows 3
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import openai

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from live_idea_bench.backtest import split_train_future_by_cutoff  # noqa: E402
from live_idea_bench.config import load_topics  # noqa: E402
from live_idea_bench.models import IdeaPrediction  # noqa: E402
from live_idea_bench.papers import load_papers_from_markdown  # noqa: E402
from live_idea_bench.similarity import paper_text, _sanitize  # noqa: E402
from live_idea_bench.topics import classify_papers_by_topic  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOYAGE_BASE_URL  = "https://api.voyageai.com/v1"
EMBED_MODEL      = "voyage-3-large"
DEFAULT_JUDGE    = "gpt-4.1-mini"
DEFAULT_TOP_R    = 10
BATCH_SIZE       = 128
MAX_CHARS        = 4000
MAX_EMBED_RETRY  = 12
MAX_JUDGE_RETRY  = 3

JUDGE_SYSTEM = """\
You are an expert scientific reviewer. You are given a PREDICTED research direction — a forecast made before the paper was published — and a PUBLISHED paper. Your task is to judge whether the published paper is a realization of that prediction.

A MATCH requires alignment on ALL FOUR facets (Shahid et al., 2025):
(1) PURPOSE   — the core research problem / objective being addressed.
(2) MECHANISM — the specific methodological approach (the technical "how"). The mechanism stated in the prediction must be a concrete technical method, not a generic meta-process.
(3) EVALUATION — the validation strategy: what is measured, on what data, to demonstrate the contribution.
(4) APPLICATION — the application domain or scope of use.

CRITICAL RULES — apply BEFORE assessing similarity:
- A. If the prediction is only a topic keyword, a phrase like "Next-stage direction: X", or describes a meta-analytic process (e.g., "track papers", "cluster trends", "benchmark recent methodology shifts"), respond MATCH: NO. Such predictions do not specify a mechanism — alignment on facets (2)–(4) cannot be verified, and shared TOPIC alone is not a realization.
- B. Do NOT match based on shared topic or keyword overlap alone. A paper being "in the same area" as the prediction's keyword is NOT sufficient. Two works can both be about "long-context LLMs" or "quantization" without one being a realization of the other.
- C. Do NOT justify a YES with "the paper realizes the broader direction" or "the prediction is a meta-analysis but the paper is a concrete realization within that area." Each facet must align independently and concretely.

Below are three reference examples.

--- EXAMPLE 1 (MATCH: NO — divergent mechanism) ---
Predicted Research Direction:
Title: Hierarchical Segment-Level Memory Routing with Learned Boundary Detection for Infinite-Context LLMs
Rationale: Standard attention scales quadratically with sequence length, making truly long-context modeling infeasible. We propose a memory system that learns where natural segment boundaries lie in a document, and routes each segment to a different compression level based on access frequency.
Approach: A boundary detector segments the input; high-salience segments are kept at full resolution in a cross-segment attention layer, while low-salience segments are compressed into summary vectors. This enables sub-quadratic attention while preserving long-range dependencies.
Key Terms: hierarchical memory, learned segmentation, cross-segment attention, infinite context, KV compression

Published Paper:
Title: SnapKV: LLM Knows What You are Looking for Before Generation
Abstract: SnapKV observes which KV positions receive attention in a fixed observation window before generation, then retains only those important KV entries for the remainder of the sequence. It achieves significant memory reduction without fine-tuning.

MATCH: NO
REASONING: PURPOSE roughly aligns (long-context efficiency via KV compression). MECHANISM diverges fundamentally: prediction relies on learned segment boundaries, hierarchical salience routing, and cross-segment attention; SnapKV uses attention-pattern observation in a fixed prefix window with no segmentation or routing.

--- EXAMPLE 2 (MATCH: YES — all four facets align) ---
Predicted Research Direction:
Title: Distilling Explicit Chain-of-Thought into Implicit Latent Reasoning Without Token Generation
Rationale: Explicit CoT inference is slow and verbose. If the reasoning steps can be internalized into the model's hidden states via distillation, we can reason without generating visible intermediate tokens.
Approach: A teacher model generates full CoT traces; a student model is trained via knowledge distillation to reproduce the final answer while encoding the reasoning process in its latent activations, never outputting reasoning tokens at inference time.
Key Terms: implicit reasoning, knowledge distillation, chain-of-thought compression, latent states, inference efficiency

Published Paper:
Title: Implicit Chain of Thought Reasoning via Knowledge Distillation
Abstract: We propose to have LLMs reason implicitly. Rather than producing explicit reasoning steps, the model internalizes them as hidden states via a teacher-student distillation framework. At inference time, no intermediate tokens are generated.

MATCH: YES
REASONING: PURPOSE (skipping explicit CoT tokens at inference), MECHANISM (teacher-student distillation of explicit CoT into latent activations), EVALUATION (downstream answer accuracy without intermediate tokens), and APPLICATION (LLM reasoning) all align.

--- EXAMPLE 3 (MATCH: NO — prediction is keyword-only / meta-analytic) ---
Predicted Research Direction:
Title: Next-stage direction: Quantization
Rationale: Keyword "quantization" appears 12 times in the recent 3 month(s), 8 times overall before 2024-08.
Approach: Track papers centered on quantization, cluster the recent methodology shifts, and benchmark whether the trend remains predictive over the next few months.
Key Terms: ['quantization']

Published Paper:
Title: A Scaling Law for Low-Bit Quantization in LLMs
Abstract: We benchmark 1500 LLM checkpoints under 2-bit and 4-bit post-training quantization and derive a scaling law that predicts downstream accuracy as a function of bit-width and model size.

MATCH: NO
REASONING: Triggers Rule A. The prediction names only a topic keyword and a meta-analytic process (track / cluster / benchmark trends). MECHANISM, EVALUATION, and APPLICATION cannot be verified because the prediction has no specific technical method — shared topic ("quantization") is not enough.
---

Respond with exactly two lines:
MATCH: YES  or  MATCH: NO
REASONING: <one to two sentences naming which facets aligned or which failed; cite Rule A/B/C if invoked>\
"""

JUDGE_USER_TMPL = """\
## Predicted Research Direction
Title: {pred_title}
Rationale: {pred_rationale}
Approach: {pred_approach}
Key Terms: {pred_terms}

## Published Paper
Title: {paper_title}
Abstract: {paper_abstract}

Apply the rules in order: (1) check Rule A — is the prediction only a topic keyword or a meta-analytic process? If yes, MATCH: NO. (2) Otherwise, verify alignment on PURPOSE, MECHANISM, EVALUATION, and APPLICATION; all four must align for MATCH: YES.
"""

MATCH_RE = re.compile(r"MATCH\s*:\s*(YES|NO)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# State file (checkpointing)
# ---------------------------------------------------------------------------
class RunState:
    """Persists embeddings and judge decisions across runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # Embeddings are persisted in a sidecar so the main state stays small
        # (~25MB: judge_decisions + window_outputs) and flushes fast on NFS.
        self.embeddings_path = path.with_name(path.stem + ".embeddings.json")
        self._lock = threading.Lock()
        self._last_flush_time = 0.0
        self._last_emb_flush_time = 0.0
        self._dirty = False
        self._emb_dirty = False
        if path.exists():
            self._data: dict = json.loads(path.read_text(encoding="utf-8"))
        else:
            self._data = {"version": 1}
        # Forward-compat: ensure all expected keys exist regardless of which
        # schema version produced the loaded file.
        self._data.setdefault("version", 1)
        self._data.setdefault("paper_embeddings", {})
        self._data.setdefault("pred_embeddings", {})
        self._data.setdefault("judge_decisions", {})
        self._data.setdefault("completed_windows", [])
        self._data.setdefault("window_outputs", {})
        # If a sidecar embeddings file exists, merge it in. The sidecar is
        # authoritative when both sources have the same key (it's written more
        # recently in steady state).
        if self.embeddings_path.exists():
            emb = json.loads(self.embeddings_path.read_text(encoding="utf-8"))
            self._data["paper_embeddings"].update(emb.get("paper_embeddings", {}))
            self._data["pred_embeddings"].update(emb.get("pred_embeddings", {}))

    # ---- embeddings --------------------------------------------------------
    def get_paper_vec(self, paper_id: str) -> list[float] | None:
        with self._lock:
            return self._data["paper_embeddings"].get(paper_id)

    def set_paper_vecs(self, id_vec_pairs: list[tuple[str, list[float]]]) -> None:
        """Batch-set paper embeddings under a single lock acquisition."""
        with self._lock:
            for paper_id, vec in id_vec_pairs:
                self._data["paper_embeddings"][paper_id] = vec
            self._flush_embeddings()

    def get_pred_vec(self, pred_hash: str) -> list[float] | None:
        with self._lock:
            return self._data["pred_embeddings"].get(pred_hash)

    def set_pred_vec(self, pred_hash: str, vec: list[float]) -> None:
        with self._lock:
            self._data["pred_embeddings"][pred_hash] = vec
            self._flush_embeddings()

    # ---- judge decisions ---------------------------------------------------
    def get_decision(self, pred_hash: str, paper_id: str) -> dict | None:
        key = f"{pred_hash}__{paper_id}"
        return self._data["judge_decisions"].get(key)

    def set_decision(self, pred_hash: str, paper_id: str, decision: dict) -> None:
        key = f"{pred_hash}__{paper_id}"
        with self._lock:
            self._data["judge_decisions"][key] = decision
            self._flush()

    # ---- window completion -------------------------------------------------
    def is_window_done(self, topic_id: str, cutoff: str) -> bool:
        return f"{topic_id}__{cutoff}" in self._data["completed_windows"]

    def get_window_output(self, topic_id: str, cutoff: str) -> dict | None:
        key = f"{topic_id}__{cutoff}"
        with self._lock:
            return self._data.get("window_outputs", {}).get(key)

    def mark_window_done(self, topic_id: str, cutoff: str, window_result: dict) -> None:
        key = f"{topic_id}__{cutoff}"
        with self._lock:
            if key not in self._data["completed_windows"]:
                self._data["completed_windows"].append(key)
            if "window_outputs" not in self._data:
                self._data["window_outputs"] = {}
            self._data["window_outputs"][key] = window_result
            self._flush()

    # ---- persistence -------------------------------------------------------
    def _flush(self, *, force: bool = False) -> None:
        """Must be called with self._lock held. Writes only the main state
        (judge_decisions + window_outputs + completed_windows, ~25 MB at full
        scale); embeddings live in a sidecar. Throttled to 1 write per 30s."""
        self._dirty = True
        now = time.time()
        if not force and now - self._last_flush_time < 30:
            return
        main = {
            "version": self._data.get("version", 1),
            "judge_decisions": self._data["judge_decisions"],
            "completed_windows": self._data["completed_windows"],
            "window_outputs": self._data["window_outputs"],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(main, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
        self._last_flush_time = now
        self._dirty = False

    def _flush_embeddings(self, *, force: bool = False) -> None:
        """Must be called with self._lock held. Writes the embeddings sidecar
        (paper_embeddings + pred_embeddings, ~150-250 MB JSON). Throttled to
        1 write per 300s — embeddings are append-mostly and stabilize early."""
        self._emb_dirty = True
        now = time.time()
        if not force and now - self._last_emb_flush_time < 300:
            return
        emb = {
            "paper_embeddings": self._data["paper_embeddings"],
            "pred_embeddings": self._data["pred_embeddings"],
        }
        tmp = self.embeddings_path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(emb, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.embeddings_path)
        self._last_emb_flush_time = now
        self._emb_dirty = False

    def force_flush(self) -> None:
        """Force-write both the main state and the embeddings sidecar."""
        with self._lock:
            self._flush(force=True)
            self._flush_embeddings(force=True)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _pred_text(p: IdeaPrediction) -> str:
    parts = [p.title]
    if p.rationale:
        parts.append(p.rationale)
    if p.approach:
        parts.append(p.approach)
    if p.key_terms:
        parts.append(", ".join(p.key_terms))
    return _sanitize(" ".join(parts))


def _pred_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _embed_batch(
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
                vecs = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
                results.extend(vecs)
                break
            except Exception as exc:
                wait = 2 ** attempt
                if attempt < MAX_EMBED_RETRY - 1:
                    print(f"\n  [embed retry {attempt+1}] {exc}", flush=True)
                    time.sleep(wait)
                else:
                    raise
    return results


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return max(0.0, min(1.0, dot / (na * nb))) if na and nb else 0.0


def _top_r_candidates(
    pred_vec: list[float],
    paper_vecs: dict[str, list[float]],
    future_paper_ids: list[str],
    top_r: int,
) -> list[tuple[str, float]]:
    """Return top-R (paper_id, cosine_score) sorted descending."""
    scores = [
        (pid, _cosine(pred_vec, paper_vecs[pid]))
        for pid in future_paper_ids
        if pid in paper_vecs
    ]
    scores.sort(key=lambda x: -x[1])
    return scores[:top_r]


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------
def _call_judge(
    pred: IdeaPrediction,
    paper_title: str,
    paper_abstract: str,
    judge_client: openai.OpenAI,
    judge_model: str,
) -> dict[str, Any]:
    user_msg = JUDGE_USER_TMPL.format(
        pred_title    = pred.title,
        pred_rationale= pred.rationale[:600],
        pred_approach = pred.approach[:400],
        pred_terms    = ", ".join(pred.key_terms),
        paper_title   = paper_title,
        paper_abstract= paper_abstract[:800],
    )
    for attempt in range(MAX_JUDGE_RETRY):
        try:
            resp = judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=120,
            )
            content = resp.choices[0].message.content or ""
            m = MATCH_RE.search(content)
            match_val = (m.group(1).upper() == "YES") if m else False
            # extract reasoning line
            reasoning = ""
            for line in content.splitlines():
                if line.strip().upper().startswith("REASONING"):
                    reasoning = line.split(":", 1)[-1].strip()
                    break
            return {"match": match_val, "reasoning": reasoning, "raw": content}
        except Exception as exc:
            wait = 2 ** attempt
            if attempt < MAX_JUDGE_RETRY - 1:
                print(f"\n  [judge retry {attempt+1}] {exc}", flush=True)
                time.sleep(wait)
            else:
                print(f"\n  [judge FAILED] {exc} — treating as NO", flush=True)
                return {"match": False, "reasoning": f"error: {exc}", "raw": ""}
    return {"match": False, "reasoning": "max retries exhausted", "raw": ""}


# ---------------------------------------------------------------------------
# Window processing
# ---------------------------------------------------------------------------
def _load_predictions(raw: list[dict]) -> list[IdeaPrediction]:
    return [
        IdeaPrediction(
            rank=p["rank"], title=p["title"],
            rationale=p.get("rationale", ""),
            approach=p.get("approach", ""),
            score=p.get("score", 0.0),
            confidence=p.get("confidence", 0.0),
            key_terms=p.get("key_terms", []),
        )
        for p in raw
    ]


def _process_window(
    window_data: dict,
    future_papers: list,
    paper_vecs: dict[str, list[float]],
    embed_client: openai.OpenAI,
    judge_client: openai.OpenAI,
    judge_model: str,
    top_k: int,
    top_r: int,
    state: RunState,
    workers: int,
) -> dict[str, Any]:
    cutoff = window_data["cutoff_month"]
    predictions = _load_predictions(window_data.get("predictions", []))[:top_k]
    future_paper_ids = [p.paper_id for p in future_papers]
    paper_lookup = {p.paper_id: p for p in future_papers}

    per_pred_out: list[dict] = []
    used_paper_ids: set[str] = set()

    for pred in predictions:
        pt = _sanitize(_pred_text(pred))[:MAX_CHARS]
        ph = _pred_hash(pt)

        # Embed prediction (cached)
        pred_vec = state.get_pred_vec(ph)
        if pred_vec is None:
            pred_vec = _embed_batch([pt], embed_client)[0]
            state.set_pred_vec(ph, pred_vec)

        # Retrieve top-R candidates
        candidates = _top_r_candidates(pred_vec, paper_vecs, future_paper_ids, top_r)

        # Judge all candidates in parallel
        def _judge_one(pid_score: tuple[str, float]) -> tuple[str, float, dict]:
            pid, score = pid_score
            cached = state.get_decision(ph, pid)
            if cached is not None:
                return pid, score, cached
            paper = paper_lookup.get(pid)
            if paper is None:
                d = {"match": False, "reasoning": "paper not found", "raw": ""}
            else:
                d = _call_judge(
                    pred=pred,
                    paper_title=getattr(paper, "title", ""),
                    paper_abstract=getattr(paper, "summary", ""),
                    judge_client=judge_client,
                    judge_model=judge_model,
                )
            state.set_decision(ph, pid, d)
            return pid, score, d

        judge_results: dict[str, tuple[float, dict]] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_judge_one, c): c for c in candidates}
            for fut in as_completed(futs):
                pid, score, decision = fut.result()
                judge_results[pid] = (score, decision)

        # Find first non-duplicate match (process in rank order)
        matched_paper_id = None
        matched_reasoning = None
        for pid, embed_score in candidates:
            score, decision = judge_results[pid]
            if decision["match"] and pid not in used_paper_ids:
                matched_paper_id = pid
                matched_reasoning = decision["reasoning"]
                used_paper_ids.add(pid)
                break

        per_pred_out.append({
            "rank": pred.rank,
            "title": pred.title,
            "is_match": matched_paper_id is not None,
            "matched_paper_id": matched_paper_id,
            "matched_reasoning": matched_reasoning,
            "top_candidates": [
                {
                    "paper_id": pid,
                    "embed_score": round(judge_results[pid][0], 4),
                    "llm_match": judge_results[pid][1]["match"],
                    "reasoning": judge_results[pid][1]["reasoning"],
                }
                for pid, _ in candidates
                if pid in judge_results
            ],
        })

    matched_ranks = [p["rank"] for p in per_pred_out if p["is_match"]]
    hit_at_k  = 1.0 if matched_ranks else 0.0
    mrr       = 1.0 / matched_ranks[0] if matched_ranks else 0.0
    precision = len(matched_ranks) / top_k if top_k else 0.0

    return {
        "cutoff_month":    cutoff,
        "cutoff_date":     window_data.get("cutoff_date", ""),
        "future_end_month": window_data.get("future_end_month", ""),
        "train_papers":    window_data.get("train_papers", 0),
        "future_papers":   len(future_papers),
        "evaluation": {
            "hit_at_k":       round(hit_at_k, 4),
            "mrr":            round(mrr, 4),
            "precision_at_k": round(precision, 4),
        },
        "per_prediction": per_pred_out,
    }


# ---------------------------------------------------------------------------
# Topic processing
# ---------------------------------------------------------------------------
def _process_topic(
    topic_id: str,
    saved_topic: dict,
    scoped_papers: list,
    horizon_months: int,
    top_k: int,
    top_r: int,
    embed_client: openai.OpenAI,
    judge_client: openai.OpenAI,
    judge_model: str,
    state: RunState,
    workers: int,
    max_windows: int | None,
) -> tuple[str, dict]:
    saved_bt = saved_topic.get("backtest")
    if not saved_bt or not saved_bt.get("windows"):
        return topic_id, {
            "topic_name": saved_topic.get("topic_name", ""),
            "paper_count": 0,
            "backtest": None,
        }

    # Batch-embed all topic papers once (use state cache)
    missing_ids   = [p.paper_id for p in scoped_papers if state.get_paper_vec(p.paper_id) is None]
    missing_papers = [p for p in scoped_papers if p.paper_id in set(missing_ids)]
    if missing_papers:
        print(f"  [{topic_id}] embedding {len(missing_papers)} papers ...", flush=True)
        texts = [_sanitize(paper_text(p)[:MAX_CHARS]) for p in missing_papers]
        vecs  = _embed_batch(texts, embed_client)
        state.set_paper_vecs(list(zip([p.paper_id for p in missing_papers], vecs)))

    paper_vecs: dict[str, list[float]] = {
        p.paper_id: state.get_paper_vec(p.paper_id)
        for p in scoped_papers
        if state.get_paper_vec(p.paper_id) is not None
    }

    saved_windows = saved_bt["windows"]
    if max_windows is not None:
        saved_windows = saved_windows[:max_windows]

    windows_out: list[dict] = []
    for wi, w in enumerate(saved_windows):
        cutoff = w["cutoff_month"]

        if state.is_window_done(topic_id, cutoff):
            cached = state.get_window_output(topic_id, cutoff)
            if cached is not None:
                windows_out.append(cached)
            print(f"  [{topic_id}] window {cutoff} already done, skipping", flush=True)
            continue

        _, future, _, _ = split_train_future_by_cutoff(
            papers=scoped_papers,
            cutoff_month=cutoff,
            horizon_months=horizon_months,
            cutoff_date=w.get("cutoff_date"),
        )
        if not future:
            continue

        print(
            f"  [{topic_id}] window {wi+1}/{len(saved_windows)} cutoff={cutoff} "
            f"future={len(future)} ...",
            flush=True,
        )

        window_result = _process_window(
            window_data=w,
            future_papers=future,
            paper_vecs=paper_vecs,
            embed_client=embed_client,
            judge_client=judge_client,
            judge_model=judge_model,
            top_k=top_k,
            top_r=top_r,
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
        return round(sum(vals) / len(vals), 4)

    summary = {
        "windows":          len(windows_out),
        "avg_hit_at_k":     _avg("hit_at_k"),
        "avg_mrr":          _avg("mrr"),
        "avg_precision_at_k": _avg("precision_at_k"),
    }
    print(
        f"  [{topic_id}] done — hit@k={summary['avg_hit_at_k']:.4f}  mrr={summary['avg_mrr']:.4f}",
        flush=True,
    )
    return topic_id, {
        "topic_name": saved_topic.get("topic_name", ""),
        "paper_count": len(scoped_papers),
        "backtest": {"summary": summary, "windows": windows_out},
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions with retrieve+LLM-judge pipeline."
    )
    parser.add_argument("--input-json",  required=True)
    parser.add_argument("--papers-dir",  required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--state-file",  default=None,
                        help="Checkpoint file (default: <output>.state.json)")
    parser.add_argument("--top-r",       type=int, default=DEFAULT_TOP_R,
                        help="Number of candidates to retrieve per prediction")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--judge-base-url", default=None,
                        help="Base URL for judge model API (e.g. http://localhost:8000/v1 for local models)")
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--workers",      type=int, default=8,
                        help="Parallel LLM judge threads per window")
    parser.add_argument("--topic-workers", type=int, default=4,
                        help="Parallel topics processed simultaneously")
    parser.add_argument("--topics",      default=None,
                        help="Comma-separated topic IDs to process (default: all)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Max windows per topic (for testing)")
    parser.add_argument("--topics-config", default=None)
    args = parser.parse_args()

    voyage_key = os.environ.get("VOYAGE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not voyage_key:
        print("ERROR: Set VOYAGE_API_KEY env var", file=sys.stderr)
        return 1
    if not openai_key:
        print("ERROR: Set OPENAI_API_KEY env var", file=sys.stderr)
        return 1

    embed_client = openai.OpenAI(api_key=voyage_key, base_url=VOYAGE_BASE_URL)
    judge_client = openai.OpenAI(
        api_key=openai_key,
        base_url=args.judge_base_url or None,
        timeout=30.0,
    )

    out_path   = Path(args.output)
    state_path = Path(args.state_file) if args.state_file else out_path.with_suffix(".state.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state = RunState(state_path)
    atexit.register(state.force_flush)

    saved = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    cfg   = saved.get("config", {})
    start_month    = cfg.get("start_month",    "2023-01")
    end_month      = cfg.get("end_month",      "2025-06")
    horizon_months = cfg.get("horizon_months", 3)
    top_k          = cfg.get("top_k",          5)

    print(f"Loading papers from {args.papers_dir} ...", flush=True)
    papers = load_papers_from_markdown(
        Path(args.papers_dir), start_month=start_month, end_month=end_month
    )
    print(f"Loaded {len(papers)} papers", flush=True)

    topics   = load_topics(args.topics_config)
    grouped  = classify_papers_by_topic(papers, topics)

    # Filter topics if requested
    topic_filter: set[str] | None = None
    if args.topics:
        topic_filter = set(t.strip() for t in args.topics.split(","))
        topics = [t for t in topics if t.id in topic_filter]
        print(f"Running on {len(topics)} topic(s): {[t.id for t in topics]}", flush=True)

    topic_results: dict[str, Any] = {}
    total_windows = 0

    def _run_topic(topic):
        saved_topic = saved.get("topic_results", {}).get(topic.id, {})
        scoped      = grouped.get(topic.id, [])
        if not scoped or not saved_topic.get("backtest"):
            return topic.id, {
                "topic_name": topic.name, "paper_count": len(scoped), "backtest": None
            }
        return _process_topic(
            topic_id=topic.id,
            saved_topic=saved_topic,
            scoped_papers=scoped,
            horizon_months=horizon_months,
            top_k=top_k,
            top_r=args.top_r,
            embed_client=embed_client,
            judge_client=judge_client,
            judge_model=args.judge_model,
            state=state,
            workers=args.workers,
            max_windows=args.max_windows,
        )

    results_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.topic_workers) as pool:
        futs = {pool.submit(_run_topic, t): t for t in topics}
        for fut in as_completed(futs):
            tid, result = fut.result()
            with results_lock:
                topic_results[tid] = result
                bt = result.get("backtest")
                if bt:
                    total_windows += bt["summary"]["windows"]
                _write_output(out_path, args, cfg, topic_results, total_windows)

    # Final weighted aggregate
    aggregate = _compute_aggregate(topic_results)

    print(f"\n{'='*60}")
    print(f"Aggregate: {total_windows} windows | judge={args.judge_model}")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.4f}")

    _write_output(out_path, args, cfg, topic_results, total_windows, aggregate)
    state.force_flush()
    print(f"\nSaved → {out_path}")
    print(f"State  → {state_path}")
    return 0


def _compute_aggregate(topic_results: dict) -> dict[str, float]:
    metrics = ("avg_hit_at_k", "avg_mrr", "avg_precision_at_k")
    out: dict[str, float] = {}
    for metric in metrics:
        num, den = 0.0, 0
        for tr in topic_results.values():
            bt = tr.get("backtest")
            if not bt:
                continue
            s = bt["summary"]
            w = s.get("windows", 0)
            if w > 0:
                num += s[metric] * w
                den += w
        out[metric] = round(num / den, 4) if den else 0.0
    return out


def _write_output(
    out_path: Path,
    args: argparse.Namespace,
    cfg: dict,
    topic_results: dict,
    total_windows: int,
    aggregate: dict | None = None,
) -> None:
    payload = {
        "mode":           "llm_judge_eval",
        "source_json":    args.input_json,
        "embed_model":    args.embed_model,
        "judge_model":    args.judge_model,
        "top_r":          args.top_r,
        "config":         cfg,
        "total_windows":  total_windows,
        "aggregate_summary": aggregate or _compute_aggregate(topic_results),
        "topic_results":  topic_results,
    }
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
