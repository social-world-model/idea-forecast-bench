"""Offline lock-in checks for one built :class:`Vocabulary`.

Everything here runs against records and vectors already on disk -- no LLM
or embedding calls. ``run_checks`` answers the question "does this
vocabulary still describe the future?" by assigning future-paper terms back
onto the pre-cutoff concepts (via :func:`idea_forecast_bench.vocab.build.
assign_record`) and comparing pre- vs. post-cutoff term traffic.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from scipy.stats import spearmanr

from idea_forecast_bench.vocab.build import assign_record
from idea_forecast_bench.vocab.config import VocabConfig
from idea_forecast_bench.vocab.types import (
    RECORD_OK,
    SLOTS,
    Concept,
    ConceptRecord,
    Slot,
    Term,
    Vocabulary,
    concept_key,
)

#: Most frequent unmapped future terms / fastest-growing concepts kept for
#: the human report. Not configurable: they bound report size, not science.
_MAX_UNMAPPED_TERMS = 40
_MAX_GROWTH_CONCEPTS = 20


@dataclass(frozen=True)
class CheckResult:
    """Flat metrics plus the longer lists the report renders as tables."""

    values: Mapping[str, float]
    details: Mapping[str, object]


def _lookup_concept(vocab: Vocabulary, text: str) -> Concept | None:
    """Find the concept for ``text`` by trying every slot's key, since a
    term's own recorded slot can be the losing side of a slot conflict."""
    for slot in SLOTS:
        concept_id = vocab.member_of.get(concept_key(slot, text))
        if concept_id is None:
            continue
        concept = vocab.concepts.get(concept_id)
        if concept is not None:
            return concept
    return None


def _mid_layer_share(
    vocab: Vocabulary, train_records: Sequence[ConceptRecord], min_papers: int
) -> float:
    denom = 0
    numer = 0
    for record in train_records:
        for _slot, term in record.terms():
            concept = _lookup_concept(vocab, term.text)
            if concept is not None and concept.background:
                continue  # background occurrences are excluded entirely
            denom += 1
            if concept is not None and concept.count >= min_papers:
                numer += 1
    return numer / denom if denom else math.nan


def _safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom else math.nan


def _extras_get(vocab: Vocabulary, key: str) -> float:
    value = vocab.extras.get(key)
    return math.nan if value is None else float(value)


@dataclass
class _FutureScan:
    object_hits: int = 0
    mechanism_hits: int = 0
    both_hits: int = 0
    n_ok_future: int = 0
    any_term_total: int = 0
    any_term_hit: int = 0
    post_paper_ids: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    text_counts: Counter[str] = field(default_factory=Counter)
    text_mapped: dict[str, bool] = field(default_factory=dict)


def _terms_for_slot(record: ConceptRecord, slot: Slot) -> tuple[Term, ...]:
    if slot == "object":
        return record.objects
    if slot == "mechanism":
        return record.mechanisms
    return record.problems


def _scan_future(
    *,
    vocab: Vocabulary,
    future_records: Sequence[ConceptRecord],
    vectors: Mapping[str, Sequence[float]],
    threshold: float,
) -> _FutureScan:
    scan = _FutureScan()
    for record in future_records:
        if record.ok:
            scan.n_ok_future += 1
        assignment = assign_record(record, vocab, vectors, threshold)
        record_slot_hit: dict[Slot, bool] = dict.fromkeys(SLOTS, False)
        for slot in SLOTS:
            terms = _terms_for_slot(record, slot)
            ids = assignment.get(slot, [])
            for term, concept_id in zip(terms, ids, strict=True):
                scan.any_term_total += 1
                scan.text_counts[term.text] += 1
                scan.text_mapped.setdefault(term.text, False)
                if concept_id is None:
                    continue
                concept = vocab.concepts.get(concept_id)
                if concept is None:
                    continue
                scan.text_mapped[term.text] = True
                scan.post_paper_ids[concept_id].add(record.paper_id)
                if not concept.background:
                    scan.any_term_hit += 1
                    record_slot_hit[slot] = True
        if record.ok:
            if record_slot_hit["object"]:
                scan.object_hits += 1
            if record_slot_hit["mechanism"]:
                scan.mechanism_hits += 1
            if record_slot_hit["object"] and record_slot_hit["mechanism"]:
                scan.both_hits += 1
    return scan


def _spearman_pre_post(
    vocab: Vocabulary, post_paper_ids: Mapping[str, set[str]]
) -> float:
    eligible = [c for c in vocab.concepts.values() if not c.background and c.count >= 2]
    if len(eligible) < 5:
        return math.nan
    pre = [c.count for c in eligible]
    post = [len(post_paper_ids.get(c.id, set())) for c in eligible]
    result = spearmanr(pre, post)
    return float(result.statistic)


def _top_post_growth(
    vocab: Vocabulary, post_paper_ids: Mapping[str, set[str]]
) -> list[tuple[str, int, int]]:
    candidates: list[tuple[float, int, str, int]] = []
    for concept in vocab.concepts.values():
        if concept.background or concept.count <= 0:
            continue
        post = len(post_paper_ids.get(concept.id, set()))
        ratio = post / concept.count
        candidates.append((ratio, post, concept.label, concept.count))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [
        (label, pre, post)
        for _ratio, post, label, pre in candidates[:_MAX_GROWTH_CONCEPTS]
    ]


def _unmapped_future_terms(scan: _FutureScan) -> list[tuple[str, int]]:
    unmapped = [text for text, mapped in scan.text_mapped.items() if not mapped]
    ranked = sorted(unmapped, key=lambda t: (-scan.text_counts[t], t))
    return [(t, scan.text_counts[t]) for t in ranked[:_MAX_UNMAPPED_TERMS]]


def run_checks(
    *,
    vocab: Vocabulary,
    train_records: Sequence[ConceptRecord],
    future_records: Sequence[ConceptRecord],
    vectors: Mapping[str, Sequence[float]],
    cfg: VocabConfig,
) -> CheckResult:
    threshold = cfg.checks.assign_threshold
    scan = _scan_future(
        vocab=vocab, future_records=future_records, vectors=vectors, threshold=threshold
    )

    coverage_object = _safe_div(scan.object_hits, scan.n_ok_future)
    coverage_mechanism = _safe_div(scan.mechanism_hits, scan.n_ok_future)
    coverage_both = _safe_div(scan.both_hits, scan.n_ok_future)
    coverage_any_term = _safe_div(scan.any_term_hit, scan.any_term_total)

    future_unique_texts = len(scan.text_counts)
    new_texts = sum(1 for mapped in scan.text_mapped.values() if not mapped)
    future_new_terms_share = _safe_div(new_texts, future_unique_texts)

    background_count = len(vocab.background())
    emerging = vocab.emerging()
    emerging_count = len(emerging)
    emerging_multi_count = sum(1 for c in emerging if c.count >= 2)
    combinable_count = len(vocab.combinable())

    values: dict[str, float] = {
        "coverage_object": coverage_object,
        "coverage_mechanism": coverage_mechanism,
        "coverage_both": coverage_both,
        "coverage_any_term": coverage_any_term,
        "spearman_pre_post": _spearman_pre_post(vocab, scan.post_paper_ids),
        "mid_layer_share": _mid_layer_share(
            vocab, train_records, cfg.checks.mid_layer_min_papers
        ),
        "background_count": float(background_count),
        "emerging_count": float(emerging_count),
        "emerging_multi_count": float(emerging_multi_count),
        "combinable_count": float(combinable_count),
        "n_slot_conflicts": _extras_get(vocab, "n_slot_conflicts"),
        "single_token_share_object": _extras_get(vocab, "single_token_share_object"),
        "single_token_share_mechanism": _extras_get(
            vocab, "single_token_share_mechanism"
        ),
        "single_token_share_problem": _extras_get(vocab, "single_token_share_problem"),
        "future_new_terms_share": future_new_terms_share,
    }
    details: dict[str, object] = {
        "unmapped_future_terms": _unmapped_future_terms(scan),
        "top_post_growth": _top_post_growth(vocab, scan.post_paper_ids),
    }
    return CheckResult(values=values, details=details)


def stability(previous: Vocabulary | None, current: Vocabulary) -> float:
    """Jaccard of non-background concept LABEL sets (slot-agnostic): how
    much of the vocabulary survived a re-build at the next cutoff."""
    if previous is None:
        return math.nan

    def labels(vocab: Vocabulary) -> set[str]:
        return {c.label for c in vocab.concepts.values() if not c.background}

    prev_labels = labels(previous)
    curr_labels = labels(current)
    union = prev_labels | curr_labels
    if not union:
        return 1.0  # both empty: trivially identical
    return len(prev_labels & curr_labels) / len(union)


def _concept(
    id_: str,
    slot: Slot,
    label: str,
    parent: str,
    count: int,
    *,
    variants: tuple[str, ...] = (),
    doc_frac: float = 0.0,
    first_seen: str = "2026-01",
    recent_count: int = 0,
    background: bool = False,
    emerging: bool = False,
) -> Concept:
    return Concept(
        id=id_,
        slot=slot,
        label=label,
        parent=parent,
        variants=variants or (label,),
        paper_ids=tuple(f"p{i}" for i in range(count)),
        count=count,
        doc_frac=doc_frac,
        first_seen=first_seen,
        recent_count=recent_count,
        background=background,
        emerging=emerging,
        slot_share=1.0,
    )


def _fake_vocab() -> tuple[Vocabulary, Vocabulary | None]:
    """A hand-built two-concept vocabulary (one background, one regular)
    whose ``member_of`` covers every future term exactly, so a stub
    ``assign_record`` needs only exact-match lookup."""
    background = _concept(
        "c_bg",
        "object",
        "language model",
        "model",
        5,
        variants=("language model", "lm"),
        doc_frac=1.0,
        background=True,
    )
    regular = _concept(
        "c_reg",
        "mechanism",
        "retrieval augmentation",
        "retrieval",
        2,
        variants=("retrieval augmentation", "retrieval"),
        doc_frac=0.4,
        recent_count=1,
    )
    current = Vocabulary(
        topic_id="t0",
        cutoff_month="2026-03",
        cutoff_date="2026-03-31",
        n_train=5,
        n_with_records=5,
        concepts={"c_bg": background, "c_reg": regular},
        member_of={
            concept_key("object", "language model"): "c_bg",
            concept_key("mechanism", "retrieval augmentation"): "c_reg",
        },
        config_sha="deadbeef0000",
        extras={
            "n_slot_conflicts": 0.0,
            "single_token_share_object": 0.5,
            "single_token_share_mechanism": 0.0,
            "single_token_share_problem": 0.0,
        },
    )
    previous_regular = _concept(
        "c_reg_old",
        "mechanism",
        "retrieval augmentation",
        "retrieval",
        1,
        first_seen="2025-12",
        recent_count=1,
        emerging=True,
    )
    previous_dropped = _concept(
        "c_dropped",
        "problem",
        "catastrophic forgetting",
        "forgetting",
        1,
        first_seen="2025-11",
    )
    previous = Vocabulary(
        topic_id="t0",
        cutoff_month="2025-12",
        cutoff_date="2025-12-31",
        n_train=3,
        n_with_records=3,
        concepts={"c_reg_old": previous_regular, "c_dropped": previous_dropped},
        member_of={
            concept_key("mechanism", "retrieval augmentation"): "c_reg_old",
            concept_key("problem", "catastrophic forgetting"): "c_dropped",
        },
    )
    return current, previous


def _fake_record(pid: str, obj: str, mech: str, prob: str = "") -> ConceptRecord:
    return ConceptRecord(
        paper_id=pid,
        published_date="2026-04-01",
        status=RECORD_OK,
        objects=(Term(obj, ""),),
        mechanisms=(Term(mech, ""),),
        problems=(Term(prob, ""),) if prob else (),
    )


def _fake_future_records() -> tuple[ConceptRecord, ...]:
    return (
        # r1: object hits the BACKGROUND concept -> not satisfied; mechanism
        # hits the regular concept -> satisfied.
        _fake_record("f1", "language model", "retrieval augmentation"),
        # r2: object, mechanism and problem all unmapped (brand-new terms).
        _fake_record(
            "f2", "diffusion planner", "test time search", "diffusion planner"
        ),
        # r3: mechanism maps to the regular concept a second time -> post=2.
        _fake_record("f3", "diffusion planner", "retrieval augmentation"),
    )


def selfcheck() -> None:  # pragma: no cover - exercised manually / in CI
    current, previous = _fake_vocab()
    future_records = _fake_future_records()
    train_records = (_fake_record("p1", "language model", "retrieval augmentation"),)

    def fake_assign_record(
        record: ConceptRecord,
        vocab: Vocabulary,
        vectors: Mapping[str, Sequence[float]],
        threshold: float,
    ) -> dict[Slot, list[str | None]]:
        del vectors, threshold
        out: dict[Slot, list[str | None]] = {}
        for slot in SLOTS:
            terms = _terms_for_slot(record, slot)
            out[slot] = [vocab.member_of.get(concept_key(slot, t.text)) for t in terms]
        return out

    global assign_record
    original = assign_record
    assign_record = fake_assign_record
    try:
        cfg = VocabConfig()
        result = run_checks(
            vocab=current,
            train_records=train_records,
            future_records=future_records,
            vectors={},
            cfg=cfg,
        )
    finally:
        assign_record = original

    # coverage: 3 ok future records. object is never satisfied: r1's object
    # ("language model") maps to the BACKGROUND concept, and r2/r3's object
    # ("diffusion planner") is unmapped -> 0/3. mechanism satisfied by r1 and
    # r3 (both map to the regular concept) -> 2/3. both -> 0/3.
    assert math.isclose(result.values["coverage_object"], 0.0), result.values
    assert math.isclose(result.values["coverage_mechanism"], 2 / 3), result.values
    assert math.isclose(result.values["coverage_both"], 0.0), result.values

    # any_term: 3 object + 3 mechanism + 1 problem = 7 occurrences. Hits
    # (non-background): r1 mechanism, r3 mechanism -> 2/7.
    assert math.isclose(result.values["coverage_any_term"], 2 / 7), result.values

    # mid_layer_share: one train occurrence maps to background (excluded),
    # one maps to the regular concept (count=2 >= default min_papers=3? no
    # -- default mid_layer_min_papers is 3, so it does NOT count) -> 0/1.
    assert math.isclose(result.values["mid_layer_share"], 0.0), result.values

    # future_new_terms_share: unique future texts = {language model,
    # retrieval augmentation, diffusion planner, test time search} = 4;
    # unmapped = {diffusion planner, test time search} = 2 -> 2/4.
    assert math.isclose(result.values["future_new_terms_share"], 0.5), result.values

    # occurrence counts (not unique texts): "diffusion planner" appears in
    # r2's objects, r2's problems, and r3's objects -> 3.
    unmapped_terms = dict(
        cast(list[tuple[str, int]], result.details["unmapped_future_terms"])
    )
    assert unmapped_terms.get("diffusion planner") == 3, unmapped_terms
    assert unmapped_terms.get("test time search") == 1, unmapped_terms

    growth = result.details["top_post_growth"]
    assert growth == [("retrieval augmentation", 2, 2)], growth

    assert result.values["background_count"] == 1.0
    assert result.values["combinable_count"] == 1.0
    assert result.values["n_slot_conflicts"] == 0.0

    # stability: previous has {"retrieval augmentation", "catastrophic
    # forgetting"}, current has {"retrieval augmentation"} (background
    # excluded). Intersection=1, union=2 -> 0.5.
    stab = stability(previous, current)
    assert math.isclose(stab, 0.5), stab
    assert math.isnan(stability(None, current))

    print("checks.selfcheck: OK")


if __name__ == "__main__":  # pragma: no cover
    selfcheck()
