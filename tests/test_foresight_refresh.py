"""Tests for the rubric refresh state machine."""
from __future__ import annotations

import pytest

from forecaster.foresight.judge import RubricJudge, StubScorer
from forecaster.foresight.refresh import (
    RolloutSnapshot,
    RubricRefreshState,
    maybe_refresh,
    refresh_one_topic,
)
from forecaster.foresight.rubric import Rubric


def _good_judge() -> RubricJudge:
    # Score by simple cue presence — so refresh tests run deterministically.
    def fn(idea: str, cand: str) -> float:
        return 0.9 if "novel" in idea.lower() or "extend" in idea.lower() else 0.1
    return RubricJudge(scorer=StubScorer(fn=fn))


def _gen_rubric(topic_id: str, cutoff_t: str, pos, neg) -> Rubric:
    return Rubric(
        topic_id=topic_id, cutoff_t=cutoff_t,
        criteria=("must extend or introduce novelty",),
        must_not=("legacy framing",),
        examples_positive=tuple(pos),
        examples_negative=tuple(neg),
        version=1,
    )


def test_refresh_accepts_when_auc_passes():
    current = Rubric(topic_id="rag", cutoff_t="2024-06-30",
                     criteria=("v0 criterion",), version=2)
    rollouts = [
        RolloutSnapshot("rag", "we extend retrieval with a novel layer", "cand", 0.9),
        RolloutSnapshot("rag", "introduce novel retrieval extension", "cand", 0.92),
        RolloutSnapshot("rag", "another novel extend on retrieval", "cand", 0.88),
        RolloutSnapshot("rag", "fresh extend, novel", "cand", 0.95),
        RolloutSnapshot("rag", "legacy framing of old retrieval", "cand", 0.1),
        RolloutSnapshot("rag", "long-standing approach", "cand", 0.05),
        RolloutSnapshot("rag", "established methods only", "cand", 0.08),
        RolloutSnapshot("rag", "no operator at all", "cand", 0.1),
    ]
    outcome = refresh_one_topic(
        "rag", rollouts,
        current_rubric=current,
        generate_rubric=_gen_rubric,
        judge=_good_judge(),
        auc_threshold=0.70,
    )
    assert outcome.accepted
    assert outcome.new_version == current.version + 1
    assert outcome.candidate_rubric.metadata["ancestor_version"] == current.version


def test_refresh_rejects_on_insufficient_rollouts():
    current = Rubric(topic_id="rag", cutoff_t="2024-06-30", criteria=("v0",))
    outcome = refresh_one_topic(
        "rag", [RolloutSnapshot("rag", "x", "y", 0.5)],
        current_rubric=current,
        generate_rubric=_gen_rubric,
        judge=_good_judge(),
    )
    assert not outcome.accepted
    assert "insufficient" in outcome.reason
    # Version not bumped.
    assert outcome.new_version == current.version


def test_state_should_refresh_only_at_interval():
    s = RubricRefreshState(every=5, step=0)
    assert not s.should_refresh()
    s.step = 4
    assert not s.should_refresh()
    s.step = 5
    assert s.should_refresh()
    s.step = 10
    assert s.should_refresh()


def test_maybe_refresh_hotswaps_topic_rubric():
    current_rubrics = {
        "rag": Rubric(topic_id="rag", cutoff_t="2024-06-30",
                      criteria=("v0",), version=1),
    }
    state = RubricRefreshState(every=2, step=2, auc_threshold=0.7)
    for txt, r in [
        ("extend novel rag idea", 0.9),
        ("novel extend twice", 0.95),
        ("introduce novel extension here", 0.92),
        ("extend novel pipeline", 0.88),
        ("legacy framing", 0.05),
        ("long-standing approach", 0.05),
        ("established methods only", 0.05),
        ("no operator", 0.05),
    ]:
        state.record(RolloutSnapshot("rag", txt, "cand", r))

    outcomes = maybe_refresh(state, current_rubrics,
                             generate_rubric=_gen_rubric,
                             judge=_good_judge())
    assert outcomes and outcomes[0].accepted
    assert current_rubrics["rag"].version == 2
    assert state.rollout_buffer == []


def test_maybe_refresh_no_op_when_interval_not_reached():
    rubrics = {"rag": Rubric(topic_id="rag", cutoff_t="2024-06-30", criteria=("v0",))}
    state = RubricRefreshState(every=5, step=2)
    state.record(RolloutSnapshot("rag", "x", "y", 0.9))
    outcomes = maybe_refresh(state, rubrics, generate_rubric=_gen_rubric, judge=_good_judge())
    assert outcomes == []
    assert rubrics["rag"].version == 1


def test_maybe_refresh_skips_topics_without_existing_rubric():
    rubrics: dict[str, Rubric] = {}
    state = RubricRefreshState(every=1, step=1)
    for txt in ["extend novel"] * 4 + ["legacy"] * 4:
        state.record(RolloutSnapshot("unknown_topic", txt, "cand",
                                     0.9 if "extend" in txt else 0.1))
    outcomes = maybe_refresh(state, rubrics, generate_rubric=_gen_rubric, judge=_good_judge())
    assert outcomes == []
