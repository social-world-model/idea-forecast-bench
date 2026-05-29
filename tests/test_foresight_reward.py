"""Tests for the Phase-4 reward gates + retrieve-then-judge."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from live_idea_bench.models import PaperRecord
from forecaster.foresight.gates import (
    extract_citation_candidates,
    format_ok,
    grounded,
    operator_consistent,
)
from forecaster.foresight.indices import (
    HashingEmbedder,
    build_future_index,
    build_history_index,
)
from forecaster.foresight.judge import RubricJudge, StubScorer
from forecaster.foresight.operators import load_operator_inventory
from forecaster.foresight.reward import (
    ForesightContext,
    ForesightRewardConfig,
    RewardPayload,
    compute_foresight_reward,
    compute_score_v2,
)
from forecaster.foresight.rubric import Rubric
from forecaster.models import Innovation


# --------------------------------------------------------------------------- shared fixtures


def _hist_paper(pid: str, date: str, text: str) -> PaperRecord:
    return PaperRecord(
        paper_id=pid, title=text, month=date[:7], summary=text,
        keywords=["rag"], source_path="", published_date=date,
    )


def _fut_paper(pid: str, date: str, text: str) -> PaperRecord:
    return PaperRecord(
        paper_id=pid, title=text, month=date[:7], summary=text,
        keywords=["rag"], source_path="", published_date=date,
    )


def _make_context(rubric_criteria=("must extend retrieval",)) -> ForesightContext:
    emb = HashingEmbedder(dim=128, seed=42)
    history_papers = [
        _hist_paper("2305.04321", "2024-04-15",
                    "Dense passage retrieval for rag baselines"),
        _hist_paper("2305.04322", "2024-05-20",
                    "Hybrid sparse and dense retrievers"),
    ]
    future_papers = [
        _fut_paper("p_future_1", "2024-08-15",
                   "RAG meets time series with retrieval extension"),
        _fut_paper("p_future_2", "2024-08-20",
                   "Plan composition for retrieval-augmented agents"),
    ]
    history_idx = build_history_index(history_papers, emb, cutoff_date="2024-06-30")
    future_idx = build_future_index(future_papers, emb, cutoff_date="2024-06-30")

    # Stub scorer: positive cues lift score; legacy framing tanks it.
    def scorer(idea: str, cand: str) -> float:
        s = idea.lower()
        if "long-standing" in s or "established" in s:
            return 0.05
        combined = (idea + " " + cand).lower()
        cues = sum(
            1 for w in ("extend", "new", "novel", "introduce", "retrieval", "extension")
            if w in combined
        )
        return min(0.95, 0.4 + 0.08 * cues)

    judge = RubricJudge(scorer=StubScorer(fn=scorer))
    rubric = Rubric(
        topic_id="rag", cutoff_t="2024-06-30",
        criteria=rubric_criteria,
        operator_focus=("limitation_extension",),
    )

    ctx = ForesightContext(
        embedder=emb,
        judge=judge,
        future_indices={"2024-06-30": future_idx},
        history_indices={"2024-06-30": history_idx},
        rubrics={"rag": rubric},
        inventory=load_operator_inventory(),
        config=ForesightRewardConfig(
            retrieval_top_k=2,
            grounding_threshold=0.30,
            operator_threshold=0.10,
        ),
    )
    return ctx


def _payload(
    rollout_text: str,
    *,
    cutoff: str = "2024-06-30",
    topic: str = "rag",
    op: str = "extend",
    base: str = "rag",
    gap: str = "extends retrieval to a new setting",
) -> RewardPayload:
    return RewardPayload(
        rollout_text=rollout_text,
        cutoff_date=cutoff,
        topic_id=topic,
        innovation=Innovation(base_direction=base, operator=op, gap=gap),
        operator_closed=("limitation_extension" if op == "extend" else op),
    )


# --------------------------------------------------------------------------- gate-level


def test_format_ok_accepts_proposal_text():
    inno = Innovation("rag", "extend", "new gap")
    assert format_ok("This idea extends rag with a new retrieval layer.",
                     prompt_mode="z_conditioned_realization", innovation=inno)


def test_format_ok_rejects_empty_completion():
    assert not format_ok("", prompt_mode="z_conditioned_realization",
                         innovation=Innovation("rag", "extend", "x"))


def test_extract_citation_candidates_finds_arxiv_ids():
    out = extract_citation_candidates(
        "We build on arxiv:2305.04321 and (Smith, 2024) and 2306.10000 again."
    )
    assert "2305.04321" in out
    assert "2306.10000" in out


def test_grounded_rejects_nonexistent_citation():
    ctx = _make_context()
    history = ctx.history_indices["2024-06-30"]
    # Make up an arxiv id that has no semantic neighbor in the history index.
    rollout = "We cite arxiv:9999.99999 which is not in history."
    assert not grounded(rollout, history, ctx.embedder, threshold=0.95)


def test_grounded_passes_with_no_explicit_citations_by_default():
    ctx = _make_context()
    history = ctx.history_indices["2024-06-30"]
    assert grounded("Soft pass: no explicit cite.", history, ctx.embedder)


def test_grounded_can_require_citations():
    ctx = _make_context()
    history = ctx.history_indices["2024-06-30"]
    assert not grounded(
        "No citations here.", history, ctx.embedder,
        require_citations=True,
    )


def test_operator_consistent_accepts_matching_text():
    assert operator_consistent(
        "We extend the existing baseline to overcome its context limit.",
        expected_operator="extend",
    )


def test_operator_consistent_rejects_mismatch():
    assert not operator_consistent(
        "We benchmark prior work on a fresh suite.",
        expected_operator="transfer",
    )


def test_operator_consistent_passes_unmappable():
    assert operator_consistent("anything goes", expected_operator="other")


# --------------------------------------------------------------------------- end-to-end reward


def test_emerged_idea_scores_high():
    ctx = _make_context()
    rollout = (
        "Our proposal extends retrieval with a new time-series adaptation. "
        "Building on arxiv:2305.04321, we introduce a novel retrieval-extension layer."
    )
    reward, diag = compute_foresight_reward(_payload(rollout), ctx)
    assert diag["gate"] == "passed", diag
    assert reward > 0.5, (reward, diag)


def test_random_legacy_idea_scores_low():
    ctx = _make_context()
    rollout = (
        "Long-standing line of established work in retrieval-augmented generation "
        "with no new operator. Built on arxiv:2305.04321."
    )
    reward, diag = compute_foresight_reward(_payload(rollout), ctx)
    # Either operator gate trips it, or the judge scores it low.
    if diag["gate"] == "passed":
        assert reward < 0.30, (reward, diag)
    else:
        assert diag["gate"] in {"operator", "format"}


def test_rollout_citing_nonexistent_paper_gets_zero():
    ctx = _make_context()
    rollout = (
        "We extend retrieval with a clever new trick — see arxiv:9999.99999."
    )
    ctx.config.grounding_threshold = 0.95  # force strictness
    reward, diag = compute_foresight_reward(_payload(rollout), ctx)
    assert reward == 0.0
    assert diag["gate"] == "grounding"


def test_rollout_with_wrong_operator_gets_zero():
    ctx = _make_context()
    # Idea text only talks about benchmarks; z.operator is `transfer`.
    rollout = (
        "This work proposes a new benchmark suite for retrieval-augmented "
        "agents and reports baseline numbers."
    )
    payload = _payload(rollout, op="transfer")
    reward, diag = compute_foresight_reward(payload, ctx)
    assert reward == 0.0
    assert diag["gate"] == "operator"


def test_missing_cutoff_indices_strict_fail():
    ctx = _make_context()
    payload = _payload("extend rag", cutoff="2099-01-01")
    reward, diag = compute_foresight_reward(payload, ctx)
    assert reward == 0.0
    assert diag["gate"] in {"grounding", "future"}


def test_missing_rubric_strict_fail():
    ctx = _make_context()
    ctx.rubrics = {}  # remove all rubrics
    rollout = (
        "Our proposal extends retrieval with a new time-series adaptation. "
        "Building on arxiv:2305.04321, we introduce a novel retrieval extension."
    )
    reward, diag = compute_foresight_reward(_payload(rollout), ctx)
    assert reward == 0.0
    assert diag["gate"] == "rubric"


# --------------------------------------------------------------------------- compatibility wrapper


def test_compute_score_v2_accepts_extra_info_as_string():
    ctx = _make_context()
    rollout = (
        "We extend retrieval with a new long-context retrieval-extension. "
        "Built on arxiv:2305.04321 with novel introduction."
    )
    extra = {
        "innovation": {"base_direction": "rag", "operator": "extend", "gap": "x"},
        "cutoff_date": "2024-06-30",
        "topic_id": "rag",
        "prompt_mode": "z_conditioned_realization",
    }
    reward = compute_score_v2(
        data_source="t", solution_str=rollout, ground_truth="",
        extra_info=json.dumps(extra), ctx=ctx,
    )
    assert reward > 0.0
