"""Phase-4 reward: retrieve-then-judge against the per-cutoff future index.

Drop-in replacement for the old composite paper-faithful reward. Exposed
two ways:

  1. `compute_foresight_reward(rollout_text, payload, ctx) -> float`
     — pure function for unit tests + ablation runners.
  2. Routed inside the live TRL reward callback via `ForesightContext`
     installation (forecaster/foresight/trainer_wiring.py, added next).

Gates (any failure ⇒ 0.0):
  * format_ok
  * grounded (history index over X_<=t)
  * operator_consistent (rollout-vs-z.operator)
Retrieve from future_index[cutoff_t] (top-R) → rubric-conditioned judge
(`forecaster.foresight.judge`). Return max of the judge scores.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forecaster.foresight.gates import format_ok, grounded, operator_consistent
from forecaster.foresight.indices import (
    CutoffIndexBundle,
    Embedder,
    FutureIndex,
    HistoryIndex,
)
from forecaster.foresight.judge import RubricJudge
from forecaster.foresight.operators import (
    OperatorInventory,
    load_operator_inventory,
    map_free_text_operator,
)
from forecaster.foresight.rubric import Rubric, load_rubrics_dir
from forecaster.models import Innovation

logger = logging.getLogger(__name__)

_DBG_N = 0  # TEMP: throttle counter for gate diagnostics


# --------------------------------------------------------------------------- context


@dataclass
class ForesightRewardConfig:
    """Tuning knobs for compute_foresight_reward.

    operator_threshold: minimum fraction of operator-keyword hits (the
    underlying scorer in realization_reward.compute_operator_adherence
    divides by the keyword-list length, which is 7-8 per operator, so
    0.10 ≈ "at least one keyword present"). Tune up if rollouts produce
    too many false positives.
    """
    retrieval_top_k: int = 5
    grounding_threshold: float = 0.45
    grounding_top_k: int = 5
    grounding_require_citations: bool = False
    operator_threshold: float = 0.10
    fail_open_on_index_miss: bool = False  # if True, return 0.0 with WARN; else strict fail


@dataclass
class ForesightContext:
    """Shared state passed into the reward (one per training run)."""

    embedder: Embedder
    judge: RubricJudge
    # Per-cutoff indices keyed by `cutoff_date` (YYYY-MM-DD).
    future_indices: dict[str, FutureIndex]
    history_indices: dict[str, HistoryIndex]
    # Topic-keyed rubric library (Phase 2 outputs).
    rubrics: dict[str, Rubric]
    # future_paper_id -> topic_id, used to recover topic_id when the dataset's
    # extra_info doesn't carry it (HindsightSample drops topic_id upstream).
    paper_to_topic: dict[str, str] = field(default_factory=dict)
    inventory: OperatorInventory = field(default_factory=load_operator_inventory)
    config: ForesightRewardConfig = field(default_factory=ForesightRewardConfig)

    # -------- factory --------

    @classmethod
    def from_bundles(
        cls,
        bundles: dict[str, CutoffIndexBundle],
        rubrics_dir: str | Path,
        embedder: Embedder,
        judge: RubricJudge,
        *,
        inventory: OperatorInventory | None = None,
        config: ForesightRewardConfig | None = None,
    ) -> ForesightContext:
        future = {k: b.future for k, b in bundles.items()}
        history = {k: b.history for k, b in bundles.items()}
        return cls(
            embedder=embedder,
            judge=judge,
            future_indices=future,
            history_indices=history,
            rubrics=load_rubrics_dir(rubrics_dir),
            inventory=inventory or load_operator_inventory(),
            config=config or ForesightRewardConfig(),
        )

    # -------- lookups --------

    def rubric_for(self, topic_id: str, *, operator_closed: str | None = None) -> Rubric | None:
        r = self.rubrics.get(topic_id)
        if r is None:
            return None
        if operator_closed and r.operator_focus and operator_closed not in r.operator_focus:
            # The rubric is tuned for a different operator focus — caller can
            # still use it, but log so a refresh can be triggered upstream.
            logger.debug(
                "rubric for topic=%s does not target operator=%s",
                topic_id, operator_closed,
            )
        return r


# --------------------------------------------------------------------------- payload schema


@dataclass
class RewardPayload:
    """Normalized per-rollout payload (a slim view of extra_info)."""

    rollout_text: str
    cutoff_date: str
    topic_id: str
    innovation: Innovation
    operator_closed: str
    prompt_mode: str = "z_conditioned_realization"

    @classmethod
    def from_extra_info(
        cls,
        rollout_text: str,
        extra_info: dict[str, Any],
        inventory: OperatorInventory,
        paper_to_topic: dict[str, str] | None = None,
    ) -> RewardPayload:
        inno_raw = extra_info.get("innovation") or {}
        innovation = Innovation(
            base_direction=str(inno_raw.get("base_direction") or ""),
            operator=str(inno_raw.get("operator") or ""),
            gap=str(inno_raw.get("gap") or ""),
        )
        op_closed = map_free_text_operator(innovation.operator, inventory)
        # `topic_id` may be present at top-level or under metadata; otherwise
        # recover it from target_future_paper_id via the context's map (the
        # dataset's extra_info drops topic_id because HindsightSample does).
        topic_id = (
            str(extra_info.get("topic_id") or "")
            or str(extra_info.get("topic") or "")
        )
        if not topic_id and paper_to_topic:
            tfpid = str(extra_info.get("target_future_paper_id") or "")
            topic_id = str(paper_to_topic.get(tfpid) or "")
        return cls(
            rollout_text=rollout_text,
            cutoff_date=str(extra_info.get("cutoff_date") or ""),
            topic_id=topic_id,
            innovation=innovation,
            operator_closed=op_closed,
            prompt_mode=str(extra_info.get("prompt_mode") or "z_conditioned_realization"),
        )


# --------------------------------------------------------------------------- the reward


def _candidate_text_from_index(idx: FutureIndex, paper_id: str) -> str:
    """Look up paper title/abstract from the index's meta blob, if available."""
    bank = idx.meta.get("paper_texts") if isinstance(idx.meta, dict) else None
    if isinstance(bank, dict):
        return str(bank.get(paper_id) or paper_id)
    return paper_id


def compute_foresight_reward(
    payload: RewardPayload,
    ctx: ForesightContext,
) -> tuple[float, dict[str, Any]]:
    """Pure reward function. Returns (reward, diagnostics).

    Diagnostics is a small dict explaining the score (which gate failed,
    retrieval hits, judge scores). The trainer wiring discards it; tests
    + ablations consume it.
    """
    diag: dict[str, Any] = {"gate": None, "reason": None}

    # ------------------- format gate -------------------
    if not format_ok(
        payload.rollout_text,
        prompt_mode=payload.prompt_mode,
        innovation=payload.innovation,
    ):
        diag["gate"] = "format"
        diag["reason"] = "rollout failed to parse into IdeaPrediction"
        return 0.0, diag

    # ------------------- grounding gate -------------------
    history = ctx.history_indices.get(payload.cutoff_date)
    if history is None:
        if not ctx.config.fail_open_on_index_miss:
            diag["gate"] = "grounding"
            diag["reason"] = f"history_index missing for cutoff={payload.cutoff_date}"
            return 0.0, diag
        logger.warning("no history index for cutoff=%s; failing open", payload.cutoff_date)
    elif not grounded(
        payload.rollout_text,
        history,
        ctx.embedder,
        threshold=ctx.config.grounding_threshold,
        top_k=ctx.config.grounding_top_k,
        require_citations=ctx.config.grounding_require_citations,
    ):
        diag["gate"] = "grounding"
        diag["reason"] = "cited evidence does not retrieve any close match in X_<=t"
        return 0.0, diag

    # ------------------- operator gate -------------------
    if not operator_consistent(
        payload.rollout_text,
        payload.innovation.operator,
        inventory=ctx.inventory,
        threshold=ctx.config.operator_threshold,
    ):
        diag["gate"] = "operator"
        diag["reason"] = f"rollout did not exhibit operator={payload.innovation.operator}"
        return 0.0, diag

    # ------------------- retrieve from future index -------------------
    future = ctx.future_indices.get(payload.cutoff_date)
    if future is None or future.size == 0:
        if not ctx.config.fail_open_on_index_miss:
            diag["gate"] = "future"
            diag["reason"] = f"future_index missing/empty for cutoff={payload.cutoff_date}"
            return 0.0, diag
        logger.warning("no future index for cutoff=%s; failing open", payload.cutoff_date)
        diag["gate"] = "future"
        diag["reason"] = "future index unavailable; failing open"
        return 0.0, diag
    q = ctx.embedder.encode([payload.rollout_text])[0]
    hits = future.search(q, top_k=ctx.config.retrieval_top_k)
    if not hits:
        diag["gate"] = "future"
        diag["reason"] = "future index returned 0 hits"
        return 0.0, diag

    # ------------------- rubric-conditioned judge -------------------
    rubric = ctx.rubric_for(payload.topic_id, operator_closed=payload.operator_closed)
    if rubric is None:
        diag["gate"] = "rubric"
        diag["reason"] = f"no rubric for topic={payload.topic_id!r}"
        return 0.0, diag

    judge_scores: list[float] = []
    for paper_id, _retrieval_score in hits:
        cand_text = _candidate_text_from_index(future, paper_id)
        res = ctx.judge.score(payload.rollout_text, cand_text, rubric)
        judge_scores.append(res.score)
    best_judge = float(max(judge_scores))
    # Dense shaping (env REWARD_SIM_SHAPING; default 0.0 = original judge-only).
    # The retrieval cosine-sim to the nearest future paper is a continuous signal
    # available even when the judge scores 0 (idea matched no future paper). The
    # rubric judge is near-binary (0 or 0.6-1.0), so all-zero groups have no
    # advantage variance and GRPO learns nothing. Adding lambda*sim gives
    # within-group gradient toward the future-idea distribution. hits are sorted
    # desc, so hits[0][1] is the closest cosine sim.
    import os
    _sim_w = float(os.environ.get("REWARD_SIM_SHAPING", "0.0") or 0.0)
    best_sim = float(hits[0][1]) if hits else 0.0
    reward = best_judge + _sim_w * max(0.0, best_sim)
    diag["retrieval_hits"] = hits
    diag["judge_scores"] = judge_scores
    diag["retrieval_sim"] = best_sim
    diag["sim_shaping_w"] = _sim_w
    diag["gate"] = "passed"
    return float(reward), diag


# --------------------------------------------------------------------------- compatibility wrapper


def compute_score_v2(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | str | None,
    *,
    ctx: ForesightContext,
) -> float:
    """Same signature as forecaster/realization/verl/reward_fn.py:compute_score.

    Used by the TRL runner once `--reward foresight` is selected.
    """
    if isinstance(extra_info, str):
        import json
        try:
            extra = json.loads(extra_info) if extra_info else {}
        except json.JSONDecodeError:
            extra = {}
    elif isinstance(extra_info, dict):
        extra = extra_info
    else:
        extra = {}
    payload = RewardPayload.from_extra_info(
        rollout_text=solution_str or "",
        extra_info=extra,
        inventory=ctx.inventory,
        paper_to_topic=ctx.paper_to_topic,
    )
    reward, _diag = compute_foresight_reward(payload, ctx)
    # --- TEMP DEBUG: surface which gate fired (throttled to first N calls) ---
    global _DBG_N
    if _DBG_N < 24:
        _DBG_N += 1
        logger.warning(
            "[foresight-diag #%d] reward=%.3f gate=%s reason=%s | cutoff=%s topic=%r op=%r len=%d head=%r",
            _DBG_N, reward, _diag.get("gate"), _diag.get("reason"),
            payload.cutoff_date, payload.topic_id, payload.operator_closed,
            len(payload.rollout_text), payload.rollout_text[:200],
        )
    return float(reward)


__all__ = [
    "ForesightContext",
    "ForesightRewardConfig",
    "RewardPayload",
    "compute_foresight_reward",
    "compute_score_v2",
]
