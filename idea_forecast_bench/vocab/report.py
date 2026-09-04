"""Markdown rendering for one vocabulary run: the human-facing report, and
the one-row ledger entry that tracks a (tag, prompt, config) combination
across topics and cutoffs over time."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from idea_forecast_bench.models import PaperRecord
from idea_forecast_bench.vocab.checks import CheckResult
from idea_forecast_bench.vocab.config import VocabConfig
from idea_forecast_bench.vocab.types import (
    Concept,
    ConceptRecord,
    Slot,
    Term,
    Vocabulary,
)

_MAX_EMERGING = 30
_MAX_SLOT_CONFLICTS = 20
_MAX_UNMAPPED_TERMS = 40
_MAX_VARIANTS = 3

_LEDGER_COLUMNS: tuple[str, ...] = (
    "tag",
    "prompt_fp",
    "config_sha",
    "fine_thr",
    "single_thr",
    "parent_thr",
    "bg_df",
    "min_count",
    "topics",
    "cutoffs",
    "cov_obj",
    "cov_mech",
    "cov_both",
    "spearman",
    "stability",
    "mid_layer",
    "single_tok_obj",
    "single_tok_mech",
    "bg_n",
    "emerg_n",
    "emerg_multi",
    "conflicts",
)


@dataclass(frozen=True)
class ProbeRow:
    """One probe paper shown side by side under the old (v1) and new (v2)
    extraction schemas, for a human to eyeball."""

    paper: PaperRecord
    new: ConceptRecord | None
    old: Mapping[str, Sequence[str]] | None
    reason: str


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def _fmt_pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value * 100:.1f}%"


def _render_metrics(checks: CheckResult) -> list[str]:
    lines = ["## Metrics", "", "| metric | value |", "| --- | --- |"]
    for key in sorted(checks.values):
        lines.append(f"| {key} | {_fmt(checks.values[key])} |")
    return lines


def _render_background(vocab: Vocabulary) -> list[str]:
    lines = ["## Background (excluded)", ""]
    background = vocab.background()
    if not background:
        lines.append("(none)")
        return lines
    for concept in background:
        lines.append(
            f"- {concept.label} ({concept.slot}, {concept.count}, "
            f"{_fmt_pct(concept.doc_frac)})"
        )
    return lines


def _variant_preview(concept: Concept) -> str:
    variants = [v for v in concept.variants if v != concept.label][:_MAX_VARIANTS]
    if not variants:
        return ""
    return f" — variants: {', '.join(variants)}"


def _render_child(concept: Concept) -> str:
    line = (
        f"  - {concept.label} ({concept.slot}, {concept.count}, "
        f"{_fmt_pct(concept.doc_frac)}){_variant_preview(concept)}"
    )
    if concept.emerging:
        line += " \U0001f331"
    return line


def _render_concept_tree(
    vocab: Vocabulary, max_parents: int, max_children: int
) -> list[str]:
    lines = ["## Concept tree", ""]
    by_parent: dict[str, list[Concept]] = {}
    for concept in vocab.combinable():
        by_parent.setdefault(concept.parent, []).append(concept)
    if not by_parent:
        lines.append("(none)")
        return lines
    parents = sorted(
        by_parent.items(),
        key=lambda item: (-sum(c.count for c in item[1]), item[0]),
    )
    for parent_label, children in parents[:max_parents]:
        total = sum(c.count for c in children)
        lines.append(f"- {parent_label} (total {total})")
        ranked_children = sorted(children, key=lambda c: (-c.count, c.label))
        for concept in ranked_children[:max_children]:
            lines.append(_render_child(concept))
    return lines


def _render_emerging(vocab: Vocabulary) -> list[str]:
    lines = ["## Emerging (first seen in the last N months)", ""]
    emerging = vocab.emerging()
    if not emerging:
        lines.append("(none)")
        return lines
    for concept in emerging[:_MAX_EMERGING]:
        lines.append(
            f"- {concept.label} ({concept.slot}, {concept.count}, "
            f"{concept.first_seen}, {concept.parent})"
        )
    return lines


def _render_slot_conflicts(vocab: Vocabulary) -> list[str]:
    lines = ["## Slot conflicts", ""]
    if not vocab.slot_conflicts:
        lines.append("(none)")
        return lines
    for conflict in vocab.slot_conflicts[:_MAX_SLOT_CONFLICTS]:
        lines.append(f"- {conflict}")
    return lines


def _unmapped_terms(checks: CheckResult) -> Sequence[tuple[str, int]]:
    raw = checks.details.get("unmapped_future_terms", ())
    return cast(Sequence[tuple[str, int]], raw)


def _render_unmapped_terms(checks: CheckResult) -> list[str]:
    lines = ["## Unmapped future terms (top 40)", ""]
    terms = _unmapped_terms(checks)[:_MAX_UNMAPPED_TERMS]
    if not terms:
        lines.append("(none)")
        return lines
    for text, count in terms:
        lines.append(f"- {text} ({count})")
    return lines


def _join_terms(terms: Sequence[str]) -> str:
    return ", ".join(terms) if terms else "(none)"


def _render_v1(old: Mapping[str, Sequence[str]] | None) -> str:
    if old is None:
        return "- v1: (no record)"
    groups = [
        _join_terms(old.get("themes", ())),
        _join_terms(old.get("domains", ())),
        _join_terms(old.get("methods", ())),
    ]
    return f"- v1: {' | '.join(groups)}"


def _term_group(terms: Sequence[Term]) -> str:
    parts = [
        f"{term.text} ({term.parent})" if term.parent else term.text for term in terms
    ]
    return _join_terms(parts)


def _render_v2(new: ConceptRecord | None) -> str:
    if new is None:
        return "- v2: (no record)"
    groups = [
        _term_group(new.objects),
        _term_group(new.mechanisms),
        _term_group(new.problems),
    ]
    return f"- v2: {' | '.join(groups)}"


def _render_probe(probe: Sequence[ProbeRow]) -> list[str]:
    lines = ["## Probe papers", ""]
    if not probe:
        lines.append("(none)")
        return lines
    for row in probe:
        lines.append(f"### {row.paper.title}")
        lines.append("")
        lines.append(f"reason: {row.reason}")
        lines.append("")
        lines.append(_render_v1(row.old))
        lines.append(_render_v2(row.new))
        lines.append("")
    return lines


def render_vocab_report(
    *,
    vocab: Vocabulary,
    checks: CheckResult,
    probe: Sequence[ProbeRow],
    max_parents: int = 15,
    max_children: int = 8,
) -> str:
    lines: list[str] = [
        f"# {vocab.topic_id} @ {vocab.cutoff_month}",
        "",
        f"n_train: {vocab.n_train} · n_with_records: {vocab.n_with_records} · "
        f"config_sha: {vocab.config_sha}",
        "",
        *_render_metrics(checks),
        "",
        *_render_background(vocab),
        "",
        *_render_concept_tree(vocab, max_parents, max_children),
        "",
        *_render_emerging(vocab),
        "",
        *_render_slot_conflicts(vocab),
        "",
        *_render_unmapped_terms(checks),
        "",
        *_render_probe(probe),
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_ledger_row(
    *,
    tag: str,
    prompt_fingerprint: str,
    cfg: VocabConfig,
    summary: Mapping[str, float],
    n_topics: int,
    n_cutoffs: int,
) -> str:
    def num(value: float) -> str:
        return _fmt(value)

    cells = [
        tag,
        prompt_fingerprint,
        cfg.sha,
        num(cfg.cluster.fine_threshold),
        num(cfg.cluster.single_token_threshold),
        num(cfg.cluster.parent_threshold),
        num(cfg.tag.background_doc_frac),
        num(float(cfg.tag.min_count)),
        num(float(n_topics)),
        num(float(n_cutoffs)),
        num(summary.get("coverage_object", math.nan)),
        num(summary.get("coverage_mechanism", math.nan)),
        num(summary.get("coverage_both", math.nan)),
        num(summary.get("spearman_pre_post", math.nan)),
        num(summary.get("stability", math.nan)),
        num(summary.get("mid_layer_share", math.nan)),
        num(summary.get("single_token_share_object", math.nan)),
        num(summary.get("single_token_share_mechanism", math.nan)),
        num(summary.get("background_count", math.nan)),
        num(summary.get("emerging_count", math.nan)),
        num(summary.get("emerging_multi_count", math.nan)),
        num(summary.get("n_slot_conflicts", math.nan)),
    ]
    return "| " + " | ".join(cells) + " |"


def ledger_header() -> str:
    header = "| " + " | ".join(_LEDGER_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in _LEDGER_COLUMNS) + " |"
    return f"{header}\n{separator}"


def append_ledger(path: Path, row: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8") as handle:
        if needs_header:
            handle.write(ledger_header() + "\n")
        handle.write(row.rstrip("\n") + "\n")


def _probe_concept(
    id_: str,
    slot: Slot,
    label: str,
    parent: str,
    count: int,
    *,
    variants: tuple[str, ...],
    doc_frac: float,
    first_seen: str,
    background: bool = False,
    emerging: bool = False,
) -> Concept:
    return Concept(
        id=id_,
        slot=slot,
        label=label,
        parent=parent,
        variants=variants,
        paper_ids=tuple(f"p{i}" for i in range(count)),
        count=count,
        doc_frac=doc_frac,
        first_seen=first_seen,
        recent_count=1,
        background=background,
        emerging=emerging,
        slot_share=1.0,
    )


def selfcheck() -> None:  # pragma: no cover - exercised manually / in CI
    import tempfile

    from idea_forecast_bench.vocab.checks import CheckResult
    from idea_forecast_bench.vocab.types import RECORD_OK, Term

    background = _probe_concept(
        "c_bg",
        "object",
        "language model",
        "model",
        5,
        variants=("language model", "lm"),
        doc_frac=1.0,
        first_seen="2026-01",
        background=True,
    )
    regular = _probe_concept(
        "c_reg",
        "mechanism",
        "retrieval augmentation",
        "retrieval",
        2,
        variants=("retrieval augmentation", "retrieval aug"),
        doc_frac=0.4,
        first_seen="2026-02",
        emerging=True,
    )
    vocab = Vocabulary(
        topic_id="t0",
        cutoff_month="2026-03",
        cutoff_date="2026-03-31",
        n_train=5,
        n_with_records=5,
        concepts={"c_bg": background, "c_reg": regular},
        member_of={},
        slot_conflicts=("mechanism:ambiguous term",),
        config_sha="deadbeef0000",
        extras={},
    )
    checks = CheckResult(
        values={"coverage_object": 1 / 3, "mid_layer_share": math.nan},
        details={
            "unmapped_future_terms": [("diffusion planner", 2)],
            "top_post_growth": [("retrieval augmentation", 2, 2)],
        },
    )
    paper = PaperRecord(
        paper_id="f1",
        title="A Probe Paper",
        month="2026-04",
        summary="An abstract.",
        keywords=[],
        source_path="",
    )
    new_record = ConceptRecord(
        paper_id="f1",
        published_date="2026-04-01",
        status=RECORD_OK,
        objects=(Term(text="diffusion planner", parent="planner"),),
        mechanisms=(Term(text="retrieval augmentation", parent="retrieval"),),
        problems=(),
    )
    probe = [
        ProbeRow(
            paper=paper,
            new=new_record,
            old={"themes": ("planning",), "domains": ("robotics",), "methods": ()},
            reason="regression check",
        ),
        ProbeRow(
            paper=paper,
            new=None,
            old=None,
            reason="no extraction available",
        ),
    ]

    report = render_vocab_report(vocab=vocab, checks=checks, probe=probe)
    for header in (
        "# t0 @ 2026-03",
        "## Metrics",
        "## Background (excluded)",
        "## Concept tree",
        "## Emerging (first seen in the last N months)",
        "## Slot conflicts",
        "## Unmapped future terms (top 40)",
        "## Probe papers",
    ):
        assert header in report, f"missing section {header!r}"
    assert "n/a" in report  # nan metric rendered as n/a
    assert "diffusion planner (planner)" in report
    assert "- v2: (no record)" in report
    assert "🌱" in report  # emerging child marker

    row1 = render_ledger_row(
        tag="pilot",
        prompt_fingerprint="fp123",
        cfg=VocabConfig(),
        summary={"coverage_object": 0.5, "stability": 0.75},
        n_topics=2,
        n_cutoffs=3,
    )
    row2 = render_ledger_row(
        tag="pilot2",
        prompt_fingerprint="fp456",
        cfg=VocabConfig(),
        summary={"coverage_object": 0.6},
        n_topics=2,
        n_cutoffs=3,
    )
    assert row1.count("|") == len(_LEDGER_COLUMNS) + 1
    assert "0.750" in row1
    assert "n/a" in row2  # stability missing from summary

    with tempfile.TemporaryDirectory() as tmp:
        ledger_path = Path(tmp) / "ledger.md"
        append_ledger(ledger_path, row1)
        append_ledger(ledger_path, row2)
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4, lines  # 2 header lines + 2 rows
        assert lines[0].startswith("| tag |")
        assert lines[2] == row1
        assert lines[3] == row2

    print("report.selfcheck: OK")


if __name__ == "__main__":  # pragma: no cover
    selfcheck()
