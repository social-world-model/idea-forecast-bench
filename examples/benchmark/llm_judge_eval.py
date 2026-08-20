#!/usr/bin/env python3
"""LLM-as-Judge evaluation for LiveIdeaBench predictions.

Implements a retrieve → embed → LLM-judge pipeline:
  1. Embed predictions + future papers with voyage-3-large
  2. Retrieve top-R candidates per prediction (cosine similarity)
  3. Call LLM judge for each candidate (multi-dimensional scoring)
  4. A prediction "hits" if (P+M >= 5) AND (S >= 2), scale 0-3
  5. Compute diversity (cluster coverage) and novelty (embedding distance)

Usage:
    python examples/llm_judge_eval.py \\
        --input-json logs/baselines/memory_prompting_raw.json \\
        --papers-dir data/csml/raw_markdown \\
        --output logs/baselines/memory_prompting_llmjudge.json

    # Quick test on one topic (2 windows):
    python examples/llm_judge_eval.py \\
        --input-json logs/baselines/memory_prompting_raw.json \\
        --papers-dir data/csml/raw_markdown \\
        --output /tmp/mp_judge_test.json \\
        --topics llm_pretraining --max-windows 2
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

from live_idea_bench.backtest import (  # noqa: E402
    split_train_future_by_cutoff,
    weighted_mean_over_topics,
)
from live_idea_bench.config import load_topics  # noqa: E402
from live_idea_bench.models import IdeaPrediction  # noqa: E402
from live_idea_bench.papers import load_papers_from_markdown  # noqa: E402
from live_idea_bench.similarity import _sanitize, idea_text, paper_text  # noqa: E402
from live_idea_bench.topics import classify_papers_by_topic  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOYAGE_BASE_URL = "https://api.voyageai.com/v1"
EMBED_MODEL = "voyage-3-large"
DEFAULT_JUDGE = "gpt-4.1-mini"
DEFAULT_TOP_R = 10
DEFAULT_CLUSTER_K = 5
BATCH_SIZE = 128
MAX_CHARS = 4000
MAX_EMBED_RETRY = 12
MAX_JUDGE_RETRY = 3

# Match: P + M >= MATCH_PM_THRESHOLD AND S >= MATCH_S_THRESHOLD
MATCH_PM_THRESHOLD = 5  # out of 6
MATCH_S_THRESHOLD = 2  # specificity must be meaningful

JUDGE_SYSTEM = """\
You are an expert scientific reviewer. You are given a PREDICTED research direction -- a forecast made before the paper was published -- and a PUBLISHED paper. Your task is to judge whether the published paper is a realization of that prediction.

Score the prediction-paper pair on three dimensions using a 0-3 scale:

PROBLEM_MATCH -- Does the paper address the same core research problem and goal?
  3 = Same core problem and goal -- the prediction and paper tackle the exact same question
  2 = Very similar problem: same domain challenge, minor difference in scope or framing
  1 = Adjacent problem: overlapping concern but different primary objective
  0 = Unrelated or only superficially connected problem

METHOD_MATCH -- Does the paper employ a similar technical mechanism or approach?
  3 = Same or near-identical mechanism -- the prediction's approach is directly implemented
  2 = Closely related approach: shares key technical ideas, differs in implementation detail
  1 = Broadly similar paradigm: same general category of method, substantially different specifics
  0 = Different technical approach entirely

SPECIFICITY -- Does the paper realize the specific novelty described in the prediction?
  3 = Prediction's specific novelty is precisely present in the paper
  2 = Core specific novelty partially realized; paper simplifies or focuses on a subset
  1 = Prediction is generic enough to loosely fit, or paper addresses adjacent specifics
  0 = Prediction is keyword-only or meta-analytic, or paper entirely ignores the predicted novelty

Score each dimension independently on its own merits; do not infer or optimize toward any overall verdict.
Do NOT score based on shared topic or keyword overlap alone.

Here are four reference examples ordered from clear non-match to clear match:

--- EXAMPLE 1 (does NOT match -- paradigm overlap only) ---
Predicted Research Direction:
Title: Multiscale Generative Models for Long-Form Audio and Music with Hybrid Token-Spectrogram Representations
Rationale: Diffusion models show promise for long high-fidelity music, and multiscale autoregressive architectures have proven effective for very long sequences. A likely near-term direction is combining hierarchical temporal structure with efficient long-context generation specifically for raw or near-raw audio.
Approach: Use a two-level model where a global transformer predicts coarse musical structure over long spans, and local modules generate fine-grained spectrogram patches or audio tokens. Explore diffusion or autoregressive local decoders with recurrent memory for minute-scale coherence.
Key Terms: music generation, audio modeling, multiscale transformer, diffusion, hierarchical generation, long-range coherence

Published Paper:
Title: MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers
Abstract: We propose MEGABYTE, a multiscale decoder architecture for end-to-end modeling of sequences over one million bytes. It segments sequences into patches, using a local submodel within patches and a global model between patches. Experiments show competitive performance on long-context language modeling, image density estimation, and audio from raw files.

PROBLEM_MATCH: 1
METHOD_MATCH: 2
SPECIFICITY: 1
REASONING: The prediction targets audio and music generation as its primary objective, while MEGABYTE is a general-purpose byte-sequence architecture whose audio experiments are incidental -- different primary objectives. Both share a global-local hierarchy, so the method is broadly similar, but the prediction's specific novelties (spectrogram tokens, diffusion decoders, music-specific design) are entirely absent from MEGABYTE's domain-agnostic patch approach.

--- EXAMPLE 2 (does NOT match -- mechanism diverges) ---
Predicted Research Direction:
Title: Hierarchical Segment-Level Memory Routing with Learned Boundary Detection for Infinite-Context LLMs
Rationale: Standard attention scales quadratically with sequence length, making truly long-context modeling infeasible. We propose a memory system that learns where natural segment boundaries lie in a document, and routes each segment to a different compression level based on access frequency.
Approach: A boundary detector segments the input; high-salience segments are kept at full resolution in a cross-segment attention layer, while low-salience segments are compressed into summary vectors. This enables sub-quadratic attention while preserving long-range dependencies.
Key Terms: hierarchical memory, learned segmentation, cross-segment attention, infinite context, KV compression

Published Paper:
Title: SnapKV: LLM Knows What You are Looking for Before Generation
Abstract: SnapKV observes which KV positions receive attention in a fixed observation window before generation, then retains only those important KV entries for the remainder of the sequence. It achieves significant memory reduction without fine-tuning.

PROBLEM_MATCH: 2
METHOD_MATCH: 1
SPECIFICITY: 0
REASONING: Both address long-context efficiency via selective KV retention, but the mechanisms diverge fundamentally -- the prediction requires a learned boundary detector, hierarchical routing, and cross-segment attention, while SnapKV simply observes attention over a fixed prefix window with no segmentation, routing, or learned components. None of the prediction's specific technical contributions appear in the paper.

--- EXAMPLE 3 (MATCHES -- core idea realized, details differ) ---
Predicted Research Direction:
Title: Adaptive KV Cache Compression via Importance Prediction and Mixed Precision
Rationale: Scissorhands and similar work show that token importance persists across decoding steps, and KV cache size is a major inference bottleneck. A likely next step is more adaptive cache management combining importance-based eviction, head-wise or layer-wise policies, and quantization rather than a single fixed heuristic.
Approach: Fit a lightweight importance predictor from attention statistics and recency signals to decide per-step which cached entries to keep, merge, or quantize. Use mixed precision so high-salience tokens stay at full resolution while others are compressed.
Key Terms: KV cache, inference optimization, token pruning, quantization, attention importance, serving efficiency

Published Paper:
Title: Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time
Abstract: We hypothesize the persistence of importance: only pivotal tokens that had substantial influence at one step will significantly influence future generation. Based on this, we propose SCISSORHANDS, a system that maintains KV cache under a fixed memory budget without finetuning, by evicting non-pivotal tokens. It reduces KV cache memory by up to 5x and can be combined with 4-bit quantization for further compression.

PROBLEM_MATCH: 3
METHOD_MATCH: 2
SPECIFICITY: 2
REASONING: Both address KV cache memory reduction via importance-based token eviction, and Scissorhands directly realizes this by identifying pivotal tokens from attention history. However, the prediction's specific contributions -- a trained importance predictor with per-head granularity and mixed-precision storage -- are absent; Scissorhands uses a simpler fixed-budget eviction heuristic, so the core idea is present but the predicted implementation details are not.

--- EXAMPLE 4 (MATCHES -- exact realization) ---
Predicted Research Direction:
Title: Distilling Explicit Chain-of-Thought into Implicit Latent Reasoning Without Token Generation
Rationale: Explicit CoT inference is slow and verbose. If the reasoning steps can be internalized into the model's hidden states via distillation, we can reason without generating visible intermediate tokens.
Approach: A teacher model generates full CoT traces; a student model is trained via knowledge distillation to reproduce the final answer while encoding the reasoning process in its latent activations, never outputting reasoning tokens at inference time.
Key Terms: implicit reasoning, knowledge distillation, chain-of-thought compression, latent states, inference efficiency

Published Paper:
Title: Implicit Chain of Thought Reasoning via Knowledge Distillation
Abstract: We propose to have LLMs reason implicitly. Rather than producing explicit reasoning steps, the model internalizes them as hidden states via a teacher-student distillation framework. At inference time, no intermediate tokens are generated.

PROBLEM_MATCH: 3
METHOD_MATCH: 3
SPECIFICITY: 3
REASONING: Both propose to internalize chain-of-thought reasoning into latent states via teacher-student knowledge distillation, such that no intermediate reasoning tokens are produced at inference time -- the core problem, mechanism, and implementation goal are identical.
---

Respond with exactly four lines:
PROBLEM_MATCH: <0, 1, 2, or 3>
METHOD_MATCH: <0, 1, 2, or 3>
SPECIFICITY: <0, 1, 2, or 3>
REASONING: <one to two sentences explaining your scores>\
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

Score this prediction-paper pair on PROBLEM_MATCH, METHOD_MATCH, and SPECIFICITY (0-3 each), then give REASONING.
"""

# Decode config for the judge. Lifted to module constants so they can be folded
# into the judge fingerprint (a state file produced with thinking on / a smaller
# max_tokens is NOT comparable to one without).
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 256

# Anchored to start-of-line + word boundary so a stray "...: 3" inside REASONING
# prose (or an injected line from the abstract) is not captured, and so
# "METHOD_MATCH: 2/3" is not misread as 2. The three score lines are required;
# a partial parse is treated as a failure (see _call_judge), never silently
# backfilled to 1.
SCORE_RE = re.compile(
    r"^\s*(PROBLEM_MATCH|METHOD_MATCH|SPECIFICITY)\s*:\s*([0-3])(?![0-9/.])",
    re.IGNORECASE | re.MULTILINE,
)
_REQUIRED_DIMS = ("PROBLEM_MATCH", "METHOD_MATCH", "SPECIFICITY")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# State file (checkpointing)
# ---------------------------------------------------------------------------
class RunState:
    """Persists embeddings and judge decisions across runs."""

    def __init__(
        self,
        path: Path,
        *,
        judge_fingerprint: str,
        embed_fingerprint: str,
    ) -> None:
        self.path = path
        self.judge_fp = judge_fingerprint
        self.embed_fp = embed_fingerprint
        # Split-flush: embeddings live in a sidecar file because they grow to
        # ~99% of state bytes but only change during the embedding phase. The
        # main state (decisions + windows, ~30MB) flushes every 30s; the
        # sidecar (~1.5GB) flushes every 300s. Without this, a single 866MB
        # flush takes ~10s under lock and drops throughput from ~12 c/s to
        # under 2 c/s at scale.
        self.embeddings_path = path.with_name(path.stem + ".embeddings.json")
        self._lock = threading.Lock()
        self._last_flush_time = 0.0
        self._last_emb_flush_time = 0.0
        if path.exists():
            self._data: dict = json.loads(path.read_text(encoding="utf-8"))
        else:
            self._data = {"version": 2}
        # Forward-compat: backfill expected keys regardless of which schema
        # version produced the loaded file.
        self._data.setdefault("version", 2)
        self._data.setdefault("paper_embeddings", {})
        self._data.setdefault("pred_embeddings", {})
        self._data.setdefault("judge_decisions", {})
        self._data.setdefault("completed_windows", [])
        self._data.setdefault("window_outputs", {})
        # Fingerprint guard: a resumed state file must have been produced under
        # the same judge config (model + rubric) and embedding model, otherwise
        # cached decisions / vectors are stale and incomparable. Fail loud
        # rather than silently serving them. A fresh file (no fingerprints key)
        # is stamped with the current fingerprints.
        stored_fp = self._data.get("fingerprints")
        if stored_fp is None:
            self._data["fingerprints"] = {
                "judge": self.judge_fp,
                "embed": self.embed_fp,
            }
        else:
            mismatches = []
            if stored_fp.get("judge") != self.judge_fp:
                mismatches.append(
                    f"judge ({stored_fp.get('judge')} != {self.judge_fp}): "
                    f"--judge-model or JUDGE_SYSTEM changed"
                )
            if stored_fp.get("embed") != self.embed_fp:
                mismatches.append(
                    f"embed ({stored_fp.get('embed')} != {self.embed_fp}): "
                    f"--embed-model changed"
                )
            if mismatches:
                raise ValueError(
                    "State file "
                    f"{path} was produced under a different config:\n  - "
                    + "\n  - ".join(mismatches)
                    + "\nUse a fresh --state-file (or delete the existing one) "
                    "so decisions/embeddings are not silently reused."
                )
        # If a sidecar exists, merge its embeddings in (authoritative when
        # both sources have the same key — sidecar is written more recently).
        if self.embeddings_path.exists():
            emb = json.loads(self.embeddings_path.read_text(encoding="utf-8"))
            self._data["paper_embeddings"].update(emb.get("paper_embeddings", {}))
            self._data["pred_embeddings"].update(emb.get("pred_embeddings", {}))

    # ---- embeddings --------------------------------------------------------
    def get_paper_vec(self, paper_id: str) -> list[float] | None:
        with self._lock:
            return self._data["paper_embeddings"].get(f"{self.embed_fp}__{paper_id}")

    def set_paper_vecs(self, id_vec_pairs: list[tuple[str, list[float]]]) -> None:
        with self._lock:
            for paper_id, vec in id_vec_pairs:
                self._data["paper_embeddings"][f"{self.embed_fp}__{paper_id}"] = vec
            self._flush_embeddings()

    def get_pred_vec(self, pred_hash: str) -> list[float] | None:
        with self._lock:
            return self._data["pred_embeddings"].get(f"{self.embed_fp}__{pred_hash}")

    def set_pred_vec(self, pred_hash: str, vec: list[float]) -> None:
        with self._lock:
            self._data["pred_embeddings"][f"{self.embed_fp}__{pred_hash}"] = vec
            self._flush_embeddings()

    # ---- judge decisions ---------------------------------------------------
    def get_decision(self, pred_hash: str, paper_id: str) -> dict | None:
        key = f"{self.judge_fp}__{pred_hash}__{paper_id}"
        return self._data["judge_decisions"].get(key)

    def set_decision(self, pred_hash: str, paper_id: str, decision: dict) -> None:
        key = f"{self.judge_fp}__{pred_hash}__{paper_id}"
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
            # Force a flush at window boundaries so the completed_windows
            # marker is durable for resume.
            self._flush(force=True)

    # ---- persistence -------------------------------------------------------
    def _flush(self, *, force: bool = False) -> None:
        """Must be called with self._lock held. Writes only the main state
        (decisions + windows + version, ~30MB at full scale). Embeddings
        live in a sidecar with its own throttle. Throttled to 1 write per
        30s unless force=True."""
        now = time.time()
        if not force and now - self._last_flush_time < 30:
            return
        main = {
            "version": self._data.get("version", 2),
            "fingerprints": self._data.get("fingerprints", {}),
            "judge_decisions": self._data["judge_decisions"],
            "completed_windows": self._data["completed_windows"],
            "window_outputs": self._data["window_outputs"],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(main, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
        self._last_flush_time = now

    def _flush_embeddings(self, *, force: bool = False) -> None:
        """Must be called with self._lock held. Writes the embeddings sidecar
        (paper_embeddings + pred_embeddings, ~1.5GB at full scale). Throttled
        to 1 write per 300s — embeddings are append-mostly and stabilize early
        in the run."""
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

    def force_flush(self) -> None:
        """Force-write both the main state and the embeddings sidecar."""
        with self._lock:
            self._flush(force=True)
            self._flush_embeddings(force=True)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _pred_text(p: IdeaPrediction) -> str:
    # Canonical prediction serialization shared with the benchmark matcher.
    return _sanitize(idea_text(p))


def _pred_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _judge_enable_thinking(judge_model: str) -> bool:
    """Whether the judge call disables 'thinking' mode for this model.

    Mirrors the per-model branch in _call_judge (Qwen judges run with thinking
    disabled). Folded into the fingerprint because a state file produced with
    thinking on is not comparable to one with it off."""
    return "qwen" not in judge_model.lower()


def _judge_fingerprint(judge_model: str) -> str:
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
            (JUDGE_MAX_TOKENS, JUDGE_TEMPERATURE, _judge_enable_thinking(judge_model))
        ).encode()
    )
    return h.hexdigest()[:12]


def _embed_fingerprint(embed_model: str) -> str:
    """12-hex fingerprint of the embedding model.

    Namespaces the prediction/paper vector caches so a state file embedded with
    one model is never mixed with vectors from another (incomparable geometry)."""
    return hashlib.sha256(embed_model.encode()).hexdigest()[:12]


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


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
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
# Diversity & Novelty metrics
# ---------------------------------------------------------------------------
def _cluster_coverage(
    future_vecs: list[list[float]],
    matched_ids: set[str],
    future_ids: list[str],
    k: int,
) -> float:
    """Fraction of future-paper clusters covered by at least one matched prediction."""
    n = len(future_vecs)
    if n == 0:
        return 0.0
    k = min(k, n)

    # No silent fallback. The old one put every paper in its own cluster, which
    # turns cluster_coverage from "fraction of topical clusters hit" into
    # "fraction of papers matched" -- a different quantity reported under the
    # same name, differing between machines depending on whether scikit-learn
    # happened to be installed.
    import numpy as np
    from sklearn.cluster import KMeans

    X = np.array(future_vecs)
    labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(X)

    matched_clusters = {
        labels[i] for i, pid in enumerate(future_ids) if pid in matched_ids
    }
    return round(len(matched_clusters) / k, 4)


def _novelty_score(pred_vec: list[float], train_vecs: list[list[float]]) -> float:
    """1 - max cosine similarity to any training paper. Higher = more novel."""
    if not train_vecs:
        return 1.0
    return round(1.0 - max(_cosine(pred_vec, tv) for tv in train_vecs), 4)


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
        pred_title=pred.title,
        pred_rationale=pred.rationale[:600],
        pred_approach=pred.approach[:400],
        pred_terms=", ".join(pred.key_terms),
        paper_title=paper_title,
        paper_abstract=paper_abstract[:800],
    )
    last_problem = ""
    for attempt in range(MAX_JUDGE_RETRY):
        try:
            create_kwargs = {
                "model": judge_model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": JUDGE_TEMPERATURE,
                "max_tokens": JUDGE_MAX_TOKENS,
            }
            # Qwen3.5 judges default to "thinking" mode, which burns the token
            # budget on <think> tokens before the PROBLEM_MATCH/... lines.
            if "qwen" in judge_model.lower():
                create_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            resp = judge_client.chat.completions.create(**create_kwargs)
            content = resp.choices[0].message.content or ""
            # Strip any chain-of-thought block before parsing so a reasoning
            # model's <think> ... </think> can't shadow the real score lines.
            parse_target = _THINK_RE.sub("", content)
            scores: dict[str, int] = {}
            for m in SCORE_RE.finditer(parse_target):
                scores[m.group(1).upper()] = int(m.group(2))

            missing = [d for d in _REQUIRED_DIMS if d not in scores]
            if missing:
                # Do NOT silently backfill to 1 — that would mark a truncated or
                # malformed response as a real low score. Treat as a parse
                # failure and retry; only after retries give up explicitly.
                last_problem = f"missing dims {missing}"
                if attempt < MAX_JUDGE_RETRY - 1:
                    print(
                        f"\n  [judge parse-retry {attempt + 1}] {last_problem}",
                        flush=True,
                    )
                    continue
                print(
                    f"\n  [judge PARSE-FAILED] {last_problem} — recording parse_failed",
                    flush=True,
                )
                return {
                    "match": False,
                    "problem_score": None,
                    "method_score": None,
                    "specificity_score": None,
                    "reasoning": "",
                    "raw": content,
                    "parse_failed": True,
                }

            problem = scores["PROBLEM_MATCH"]
            method = scores["METHOD_MATCH"]
            specificity = scores["SPECIFICITY"]
            match_val = (problem + method >= MATCH_PM_THRESHOLD) and (
                specificity >= MATCH_S_THRESHOLD
            )

            reasoning = ""
            for line in parse_target.splitlines():
                if line.strip().upper().startswith("REASONING"):
                    reasoning = line.split(":", 1)[-1].strip()
                    break

            return {
                "match": match_val,
                "problem_score": problem,
                "method_score": method,
                "specificity_score": specificity,
                "reasoning": reasoning,
                "raw": content,
                "parse_failed": False,
            }
        except Exception as exc:
            wait = 2**attempt
            if attempt < MAX_JUDGE_RETRY - 1:
                print(f"\n  [judge retry {attempt + 1}] {exc}", flush=True)
                time.sleep(wait)
            else:
                print(f"\n  [judge FAILED] {exc} — treating as NO", flush=True)
                return {
                    "match": False,
                    "problem_score": 0,
                    "method_score": 0,
                    "specificity_score": 0,
                    "reasoning": f"error: {exc}",
                    "raw": "",
                    "parse_failed": True,
                }
    return {
        "match": False,
        "problem_score": None,
        "method_score": None,
        "specificity_score": None,
        "reasoning": "max retries exhausted",
        "raw": "",
        "parse_failed": True,
    }


# ---------------------------------------------------------------------------
# Window processing
# ---------------------------------------------------------------------------
def _load_predictions(raw: list[dict]) -> list[IdeaPrediction]:
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


def _process_window(
    window_data: dict,
    future_papers: list,
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
    predictions = _load_predictions(window_data.get("predictions", []))[:top_k]
    future_paper_ids = [p.paper_id for p in future_papers]
    paper_lookup = {p.paper_id: p for p in future_papers}

    train_vecs = [paper_vecs[pid] for pid in train_paper_ids if pid in paper_vecs]

    per_pred_out: list[dict] = []
    used_paper_ids: set[str] = set()
    pred_vecs_for_novelty: list[tuple[list[float], bool]] = []  # (vec, is_match)
    judge_calls = 0
    judge_parse_failures = 0

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
        # Loop variables bound as defaults: see the note in
        # live_idea_bench/similarity.py. The executor is drained within the
        # iteration, so this makes an existing guarantee explicit.
        def _judge_one(
            pid_score: tuple[str, float],
            _ph: str = ph,
            _pred: str = pred,
        ) -> tuple[str, float, dict]:
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
                d = _call_judge(
                    pred=_pred,
                    paper_title=getattr(paper, "title", ""),
                    paper_abstract=getattr(paper, "summary", ""),
                    judge_client=judge_client,
                    judge_model=judge_model,
                )
            state.set_decision(_ph, pid, d)
            return pid, score, d

        judge_results: dict[str, tuple[float, dict]] = {}
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
        matched_decision: dict = {}
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

        novelty = _novelty_score(pred_vec, train_vecs)
        pred_vecs_for_novelty.append(pred_vec)

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
    coverage = _cluster_coverage(future_vecs, matched_ids, future_paper_ids, cluster_k)

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
    cluster_k: int,
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
    missing_ids = [
        p.paper_id for p in scoped_papers if state.get_paper_vec(p.paper_id) is None
    ]
    missing_papers = [p for p in scoped_papers if p.paper_id in set(missing_ids)]
    if missing_papers:
        print(f"  [{topic_id}] embedding {len(missing_papers)} papers ...", flush=True)
        texts = [_sanitize(paper_text(p)[:MAX_CHARS]) for p in missing_papers]
        vecs = _embed_batch(texts, embed_client)
        state.set_paper_vecs(
            list(zip([p.paper_id for p in missing_papers], vecs, strict=False))
        )

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

        window_result = _process_window(
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
        return round(sum(vals) / len(vals), 4)

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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions with retrieve+LLM-judge pipeline."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--papers-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--state-file",
        default=None,
        help="Checkpoint file (default: <output>.state.json)",
    )
    parser.add_argument(
        "--top-r",
        type=int,
        default=DEFAULT_TOP_R,
        help="Number of candidates to retrieve per prediction",
    )
    parser.add_argument(
        "--cluster-k",
        type=int,
        default=DEFAULT_CLUSTER_K,
        help="Number of clusters for future-paper diversity coverage",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="Base URL for judge model API (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel LLM judge threads per window"
    )
    parser.add_argument(
        "--topic-workers",
        type=int,
        default=4,
        help="Parallel topics processed simultaneously",
    )
    parser.add_argument(
        "--topics",
        default=None,
        help="Comma-separated topic IDs to process (default: all)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Max windows per topic (for testing)",
    )
    parser.add_argument("--topics-config", default=None)
    args = parser.parse_args()

    # Embedding backend: Voyage-only, no fallback. A missing key fails loud
    # rather than silently swapping in a different embedding geometry (which
    # would make scores incomparable to Voyage-embedded runs).
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    if not voyage_key:
        print(
            "ERROR: Set VOYAGE_API_KEY — the judge embeds with Voyage and has no "
            "local fallback (mixing embedding models corrupts score comparability).",
            file=sys.stderr,
        )
        return 1
    embed_client = openai.OpenAI(api_key=voyage_key, base_url=VOYAGE_BASE_URL)

    # Judge client: prefer env-var endpoint so we can target a local
    # vLLM OpenAI-compatible server when OPENAI_API_KEY is absent.
    judge_base_url = args.judge_base_url or os.environ.get("JUDGE_BASE_URL")
    openai_key = os.environ.get("OPENAI_API_KEY")
    judge_api_key = os.environ.get("JUDGE_API_KEY", openai_key or "EMPTY")
    if not judge_base_url and not openai_key:
        print(
            "ERROR: Set OPENAI_API_KEY (for hosted judge) or "
            "JUDGE_BASE_URL/--judge-base-url (for local vLLM judge).",
            file=sys.stderr,
        )
        return 1
    judge_client = openai.OpenAI(
        api_key=judge_api_key,
        base_url=judge_base_url or None,
        timeout=60.0,
    )

    out_path = Path(args.output)
    state_path = (
        Path(args.state_file)
        if args.state_file
        else out_path.with_suffix(".state.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    judge_fp = _judge_fingerprint(args.judge_model)
    embed_fp = _embed_fingerprint(args.embed_model)
    state = RunState(state_path, judge_fingerprint=judge_fp, embed_fingerprint=embed_fp)
    atexit.register(state.force_flush)

    saved = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    cfg = saved.get("config", {})
    start_month = cfg.get("start_month", "2023-01")
    end_month = cfg.get("end_month", "2025-06")
    horizon_months = cfg.get("horizon_months", 3)
    top_k = cfg.get("top_k", 5)

    print(f"Loading papers from {args.papers_dir} ...", flush=True)
    papers = load_papers_from_markdown(
        Path(args.papers_dir), start_month=start_month, end_month=end_month
    )
    print(f"Loaded {len(papers)} papers", flush=True)

    topics = load_topics(args.topics_config)
    grouped = classify_papers_by_topic(papers, topics)

    if args.topics:
        topic_filter = {t.strip() for t in args.topics.split(",")}
        topics = [t for t in topics if t.id in topic_filter]
        print(
            f"Running on {len(topics)} topic(s): {[t.id for t in topics]}", flush=True
        )

    topic_results: dict[str, Any] = {}
    total_windows = 0

    def _run_topic(topic):
        saved_topic = saved.get("topic_results", {}).get(topic.id, {})
        scoped = grouped.get(topic.id, [])
        if not scoped or not saved_topic.get("backtest"):
            return topic.id, {
                "topic_name": topic.name,
                "paper_count": len(scoped),
                "backtest": None,
            }
        return _process_topic(
            topic_id=topic.id,
            saved_topic=saved_topic,
            scoped_papers=scoped,
            horizon_months=horizon_months,
            top_k=top_k,
            top_r=args.top_r,
            cluster_k=args.cluster_k,
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

    aggregate = _compute_aggregate(topic_results)

    print(f"\n{'=' * 60}")
    print(f"Aggregate: {total_windows} windows | judge={args.judge_model}")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.4f}")

    _write_output(out_path, args, cfg, topic_results, total_windows, aggregate)
    state.force_flush()
    print(f"\nSaved → {out_path}")
    print(f"State  → {state_path}")
    return 0


def _compute_aggregate(topic_results: dict) -> dict[str, float]:
    return weighted_mean_over_topics(
        topic_results,
        (
            "avg_hit_at_k",
            "avg_mrr",
            "avg_precision_at_k",
            "avg_soft_score",
            "avg_cluster_coverage",
            "avg_novelty",
        ),
    )


def _write_output(
    out_path: Path,
    args: argparse.Namespace,
    cfg: dict,
    topic_results: dict,
    total_windows: int,
    aggregate: dict | None = None,
) -> None:
    payload = {
        "mode": "llm_judge_eval",
        "source_json": args.input_json,
        "embed_model": args.embed_model,
        "judge_model": args.judge_model,
        "top_r": args.top_r,
        "cluster_k": args.cluster_k,
        "match_pm_threshold": MATCH_PM_THRESHOLD,
        "match_s_threshold": MATCH_S_THRESHOLD,
        "config": cfg,
        "total_windows": total_windows,
        "aggregate_summary": aggregate or _compute_aggregate(topic_results),
        "topic_results": topic_results,
    }
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
