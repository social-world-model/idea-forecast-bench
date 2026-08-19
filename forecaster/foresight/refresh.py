"""Phase-6 rubric refresh (co-evolution).

Every K updates, snapshot the policy's recent rollouts, label them by
reward (high → fresh positives, low → fresh negatives) within each topic,
regenerate the rubric, re-validate AUC, and hot-swap. Rubric versions
are bumped and logged so a per-step audit can detect reward hacking
(inflated training reward + flat held-out metrics).

Disabled by default — opt in via `rubric_refresh_every > 0` on the config.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from forecaster.foresight.judge import RubricJudge
from forecaster.foresight.rubric import Rubric, save_rubric, stamp_metadata
from forecaster.foresight.rubric_validation import (
    LabeledPair,
    validate_rubric,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- types


@dataclass
class RolloutSnapshot:
    topic_id: str
    rollout_text: str
    candidate_text: str
    reward: float


@dataclass
class RefreshOutcome:
    topic_id: str
    old_version: int
    new_version: int
    candidate_rubric: Rubric
    auc: float
    leakage_hits: int
    accepted: bool
    reason: str = ""


# ---------------------------------------------------------------- helpers


def _split_by_reward(
    rollouts: Iterable[RolloutSnapshot],
    *,
    pos_quantile: float = 0.75,
    neg_quantile: float = 0.25,
    min_per_class: int = 4,
) -> tuple[list[RolloutSnapshot], list[RolloutSnapshot]]:
    """Quantile-split into freshly-labeled positives + negatives."""
    rs = list(rollouts)
    if len(rs) < (min_per_class * 2):
        return [], []
    sorted_rs = sorted(rs, key=lambda x: x.reward)
    n = len(sorted_rs)
    neg = sorted_rs[: max(int(n * neg_quantile), min_per_class)]
    pos = sorted_rs[-max(int(n * (1.0 - pos_quantile)), min_per_class):]
    return pos, neg


def _build_pairs(
    pos: Iterable[RolloutSnapshot],
    neg: Iterable[RolloutSnapshot],
) -> list[LabeledPair]:
    pairs: list[LabeledPair] = []
    for r in pos:
        pairs.append(LabeledPair(
            idea_text=r.rollout_text, candidate_text=r.candidate_text, label=1,
            meta={"refresh_source": "policy_high_reward"},
        ))
    for r in neg:
        pairs.append(LabeledPair(
            idea_text=r.rollout_text, candidate_text=r.candidate_text, label=0,
            meta={"refresh_source": "policy_low_reward"},
        ))
    return pairs


# ---------------------------------------------------------------- main entrypoints


GenerateRubricFn = Callable[[str, str, list[str], list[str]], Rubric]
"""Pluggable rubric generator: (topic_id, cutoff_t, pos_examples, neg_examples) -> Rubric."""


def refresh_one_topic(
    topic_id: str,
    rollouts_for_topic: list[RolloutSnapshot],
    *,
    current_rubric: Rubric,
    generate_rubric: GenerateRubricFn,
    judge: RubricJudge,
    auc_threshold: float = 0.70,
) -> RefreshOutcome:
    """Run a single topic's rubric refresh + validation cycle."""
    pos, neg = _split_by_reward(rollouts_for_topic)
    if not pos or not neg:
        return RefreshOutcome(
            topic_id=topic_id,
            old_version=current_rubric.version,
            new_version=current_rubric.version,
            candidate_rubric=current_rubric,
            auc=0.0,
            leakage_hits=0,
            accepted=False,
            reason=f"insufficient rollouts (pos={len(pos)} neg={len(neg)})",
        )

    cand = generate_rubric(
        topic_id, current_rubric.cutoff_t,
        [p.rollout_text for p in pos[:3]],
        [n.rollout_text for n in neg[:3]],
    )
    # Persist the temporal pedigree: bump version, stamp ancestry.
    bumped = Rubric(
        topic_id=topic_id,
        cutoff_t=current_rubric.cutoff_t,
        criteria=cand.criteria,
        must_not=cand.must_not,
        examples_positive=cand.examples_positive,
        examples_negative=cand.examples_negative,
        operator_focus=cand.operator_focus or current_rubric.operator_focus,
        version=current_rubric.version + 1,
        metadata={
            **stamp_metadata(model=cand.metadata.get("model", "")),
            "ancestor_version": current_rubric.version,
            "refresh_pos_count": len(pos),
            "refresh_neg_count": len(neg),
        },
    )

    pairs = _build_pairs(pos, neg)
    report, _scored = validate_rubric(
        bumped, pairs, judge=judge, threshold=auc_threshold,
    )
    accepted = report.passed
    return RefreshOutcome(
        topic_id=topic_id,
        old_version=current_rubric.version,
        new_version=bumped.version if accepted else current_rubric.version,
        candidate_rubric=bumped,
        auc=report.auc,
        leakage_hits=report.leakage_hits,
        accepted=accepted,
        reason=("auc and leakage check passed" if accepted else
                f"rejected: auc={report.auc:.3f} leakage={report.leakage_hits}"),
    )


@dataclass
class RubricRefreshState:
    """Per-step bookkeeper. Held by the trainer wiring across batches."""
    every: int = 0
    step: int = 0
    rollout_buffer: list[RolloutSnapshot] = field(default_factory=list)
    auc_threshold: float = 0.70
    rubrics_dir: Path | None = None
    versions: dict[str, int] = field(default_factory=dict)

    def record(self, snapshot: RolloutSnapshot) -> None:
        self.rollout_buffer.append(snapshot)

    def should_refresh(self) -> bool:
        return self.every > 0 and self.step > 0 and (self.step % self.every == 0)

    def reset_buffer(self) -> None:
        self.rollout_buffer.clear()


def maybe_refresh(
    state: RubricRefreshState,
    rubrics: dict[str, Rubric],
    *,
    generate_rubric: GenerateRubricFn,
    judge: RubricJudge,
) -> list[RefreshOutcome]:
    """Run one refresh cycle if `state.should_refresh()`; mutate `rubrics` in place.

    Returns one RefreshOutcome per topic visited (whether accepted or not).
    Always clears the snapshot buffer at the end of a cycle.
    """
    if not state.should_refresh():
        return []
    outcomes: list[RefreshOutcome] = []
    grouped: dict[str, list[RolloutSnapshot]] = defaultdict(list)
    for snap in state.rollout_buffer:
        if snap.topic_id:
            grouped[snap.topic_id].append(snap)

    for topic_id, rollouts in grouped.items():
        current = rubrics.get(topic_id)
        if current is None:
            continue
        outcome = refresh_one_topic(
            topic_id, rollouts,
            current_rubric=current,
            generate_rubric=generate_rubric,
            judge=judge,
            auc_threshold=state.auc_threshold,
        )
        outcomes.append(outcome)
        if outcome.accepted:
            rubrics[topic_id] = outcome.candidate_rubric
            state.versions[topic_id] = outcome.new_version
            if state.rubrics_dir is not None:
                save_rubric(outcome.candidate_rubric, state.rubrics_dir / f"{topic_id}.json")
            logger.info(
                "rubric refresh accepted: topic=%s v%d -> v%d auc=%.3f",
                topic_id, outcome.old_version, outcome.new_version, outcome.auc,
            )
        else:
            logger.warning(
                "rubric refresh rejected: topic=%s %s",
                topic_id, outcome.reason,
            )
    state.reset_buffer()
    return outcomes


__all__ = [
    "RolloutSnapshot",
    "RefreshOutcome",
    "RubricRefreshState",
    "refresh_one_topic",
    "maybe_refresh",
    "GenerateRubricFn",
]
