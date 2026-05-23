"""Tests for rubric schema + judge prompt + validation AUC."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forecaster.foresight.judge import (
    JUDGE_SYSTEM_PROMPT,
    MUST_NOT_PENALTY,
    RubricJudge,
    StubScorer,
    build_judge_user_prompt,
    parse_score,
)
from forecaster.foresight.rubric import (
    Rubric,
    build_rubric_generation_prompt,
    load_rubric,
    parse_rubric_response,
    save_rubric,
)
from forecaster.foresight.rubric_validation import (
    LabeledPair,
    compute_auc,
    validate_rubric,
)


# ----------------------------------------------------------------- rubric schema


def test_rubric_round_trip(tmp_path: Path):
    r = Rubric(
        topic_id="rag",
        cutoff_t="2024-06-30",
        criteria=("must use retrieval", "must be inference-time"),
        must_not=("no fine-tuning",),
        examples_positive=("hyde-style retrieval",),
        examples_negative=("plain finetune",),
        operator_focus=("limitation_extension",),
        version=1,
    )
    p = save_rubric(r, tmp_path / "rag.json")
    loaded = load_rubric(p)
    assert loaded == r


def test_rubric_as_judge_block_contains_keywords():
    r = Rubric(
        topic_id="rag",
        cutoff_t="2024-06-30",
        criteria=("uses retrieval",),
        must_not=("no fine-tuning",),
    )
    block = r.as_judge_block()
    assert "uses retrieval" in block
    assert "no fine-tuning" in block
    assert "rag" in block


def test_generation_prompt_renders_examples():
    prompt = build_rubric_generation_prompt(
        topic_id="rag",
        cutoff_t="2024-06-30",
        positive_examples=["retrieval-augmented forecasting"],
        negative_examples=["plain transformer"],
        operator_focus=["limitation_extension"],
    )
    assert "retrieval-augmented forecasting" in prompt
    assert "plain transformer" in prompt
    assert "limitation_extension" in prompt


def test_parse_rubric_response_strict():
    raw = '{"criteria": ["uses retrieval", "is inference-time"], "must_not": ["finetunes the model"]}'
    crit, must = parse_rubric_response(raw)
    assert crit == ("uses retrieval", "is inference-time")
    assert must == ("finetunes the model",)


def test_parse_rubric_response_handles_code_fence():
    raw = "```json\n{\"criteria\": [\"x\"], \"must_not\": []}\n```"
    crit, _ = parse_rubric_response(raw)
    assert crit == ("x",)


def test_parse_rubric_response_rejects_empty_criteria():
    with pytest.raises(ValueError):
        parse_rubric_response('{"criteria": [], "must_not": []}')


# ----------------------------------------------------------------- judge prompt + parser


def test_judge_user_prompt_includes_rubric_and_inputs():
    r = Rubric(
        topic_id="rag", cutoff_t="2024-06-30",
        criteria=("uses retrieval",),
    )
    prompt = build_judge_user_prompt("my idea", "paper content", r)
    assert "my idea" in prompt
    assert "paper content" in prompt
    assert "uses retrieval" in prompt
    assert "Score:" in prompt  # tells the model what to emit


def test_judge_prompt_uses_soft_must_not_penalty_not_hard_ceiling():
    """Option-2 invariants: prompt must subtract a fixed penalty, cap once, floor at 0.

    These are load-bearing per the Phase-2 design decision — a hard
    'must_not → ≤0.2' ceiling turned the M2 gate into a self-fulfilling
    check on rubrics whose must_not language overlapped with the
    must-have features of real positives. The new wording must say:
      1) subtract (not 'at most'),
      2) at most once / no accumulation,
      3) floor at 0.
    """
    r = Rubric(topic_id="rag", cutoff_t="2024-06-30",
               criteria=("uses retrieval",), must_not=("no fine-tuning",))
    prompt = build_judge_user_prompt("idea", "candidate", r)
    lower = prompt.lower()
    assert "subtract" in lower
    assert "at most once" in lower
    assert "do not accumulate" in lower
    assert "floor" in lower
    # The locked penalty value must appear verbatim.
    assert f"{MUST_NOT_PENALTY:.2f}" in prompt
    # Anti-regression: the old hard ceiling phrasing must NOT reappear.
    assert "at most 0.2" not in lower
    assert "must be at most" not in lower


def test_must_not_penalty_is_locked_at_two_tenths():
    """If you find yourself changing this number, re-read [[foresight-rl-plan-decisions]]
    — the penalty is a principled constant, not a tunable knob."""
    assert MUST_NOT_PENALTY == 0.2


def test_parse_score_extracts_clamped():
    s, _ = parse_score("Score: 0.83\nReasoning: it's good")
    assert s == pytest.approx(0.83)
    s, _ = parse_score("Score: 1.5\nReasoning: too high")
    assert s == 1.0
    s, _ = parse_score("no score here")
    assert s == 0.0


def test_stub_scorer_round_trip():
    # Fake scorer that returns 0.9 if idea overlaps with candidate, else 0.1.
    def f(idea: str, candidate: str) -> float:
        return 0.9 if any(w in candidate for w in idea.split()) else 0.1

    scorer = StubScorer(fn=f, name="overlap-stub")
    judge = RubricJudge(scorer=scorer)
    r = Rubric(topic_id="t", cutoff_t="2024-06-30", criteria=("x",))
    hit = judge.score("hyde retrieval", "a paper about hyde retrieval", r)
    miss = judge.score("hyde retrieval", "completely unrelated content", r)
    assert hit.score > miss.score
    assert hit.score == pytest.approx(0.9)
    assert miss.score == pytest.approx(0.1)


# ----------------------------------------------------------------- AUC


def test_auc_perfect_separation():
    assert compute_auc([0.9, 0.95, 1.0], [0.1, 0.2, 0.3]) == pytest.approx(1.0)


def test_auc_random_at_half():
    assert compute_auc([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)


def test_auc_inverted_below_half():
    assert compute_auc([0.1, 0.2], [0.9, 0.95]) == pytest.approx(0.0)


def test_auc_empty_returns_half():
    assert compute_auc([], [0.1, 0.2]) == 0.5
    assert compute_auc([0.5], []) == 0.5


# ----------------------------------------------------------------- end-to-end validation


def test_validate_rubric_pass(tmp_path: Path):
    r = Rubric(topic_id="rag", cutoff_t="2024-06-30", criteria=("uses retrieval",))

    # Stub scorer: positives have "future" in candidate, negatives don't.
    def scoring(idea: str, candidate: str) -> float:
        if "future" in candidate:
            return 0.85
        return 0.15

    pairs = [
        LabeledPair("retrieval idea", "future paper about retrieval", 1),
        LabeledPair("retrieval idea v2", "future paper, fresh", 1),
        LabeledPair("retrieval idea v3", "future paper extra", 1),
        LabeledPair("legacy retrieval", "old paper", 0),
        LabeledPair("legacy retrieval", "old paper too", 0),
        LabeledPair("legacy retrieval", "stale paper", 0),
    ]
    judge = RubricJudge(scorer=StubScorer(fn=scoring))
    report, scored = validate_rubric(r, pairs, judge=judge, threshold=0.70)
    assert report.passed
    assert report.auc == pytest.approx(1.0)
    assert report.leakage_hits == 0
    assert len(scored) == 6


def test_validate_rubric_fails_on_low_auc():
    r = Rubric(topic_id="rag", cutoff_t="2024-06-30", criteria=("x",))

    def scoring(idea: str, candidate: str) -> float:
        return 0.5     # everything tied → AUC = 0.5

    judge = RubricJudge(scorer=StubScorer(fn=scoring))
    pairs = [
        LabeledPair("i", "c1", 1),
        LabeledPair("i", "c2", 1),
        LabeledPair("i", "c3", 0),
        LabeledPair("i", "c4", 0),
    ]
    report, _ = validate_rubric(r, pairs, judge=judge, threshold=0.70)
    assert not report.passed
    assert report.auc == pytest.approx(0.5)


def test_validate_rubric_flags_leakage_even_with_high_auc():
    """A single negative scoring as high as the positive median is a leakage hit."""
    r = Rubric(topic_id="rag", cutoff_t="2024-06-30", criteria=("x",))

    def scoring(idea: str, candidate: str) -> float:
        # Most positives high, most negatives low — but one negative leaks.
        if "leak" in candidate:
            return 0.95
        if "pos" in candidate:
            return 0.9
        return 0.1

    judge = RubricJudge(scorer=StubScorer(fn=scoring))
    pairs = [
        LabeledPair("i", "pos1", 1),
        LabeledPair("i", "pos2", 1),
        LabeledPair("i", "pos3", 1),
        LabeledPair("i", "pos4", 1),
        LabeledPair("i", "pos5", 1),
        LabeledPair("i", "neg_a", 0),
        LabeledPair("i", "neg_b", 0),
        LabeledPair("i", "neg_c", 0),
        LabeledPair("i", "neg_d", 0),
        LabeledPair("i", "neg_leak_paper", 0),  # one leaked negative scoring above pos median
    ]
    report, _ = validate_rubric(r, pairs, judge=judge, threshold=0.70)
    assert report.auc >= 0.70          # mostly separates despite the one leak
    assert report.leakage_hits == 1
    assert not report.passed           # leakage trumps high AUC
    assert report.leakage_examples[0]["score"] == pytest.approx(0.95)
