"""Constants and prompts that define the retrieve-then-judge protocol.

Changing anything here changes what a score means, so these values are also
folded into the judge fingerprint that guards the checkpoint file.
"""

from __future__ import annotations

import re

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
# a partial parse is treated as a failure (see call_judge), never silently
# backfilled to 1.
SCORE_RE = re.compile(
    r"^\s*(PROBLEM_MATCH|METHOD_MATCH|SPECIFICITY)\s*:\s*([0-3])(?![0-9/.])",
    re.IGNORECASE | re.MULTILINE,
)
REQUIRED_DIMS = ("PROBLEM_MATCH", "METHOD_MATCH", "SPECIFICITY")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
