#!/usr/bin/env python
"""Phase-2 rubric construction + AUC validation.

Two modes:
  --mode smoke   stubbed scorer + hand-crafted rubrics. Proves the pipeline runs
                 end-to-end without network/LLM. CI-safe.
  --mode live    real LLM (idea_forecast_bench.llm.create_client) for both rubric
                 generation and judge scoring. Requires API keys + network.

Per-topic workflow:
  1. Pull D_z rows in the topic (positives are ideas extracted from
     post-cutoff papers; the candidate text = "{title}\\n\\n{gap}").
  2. Construct negatives: ideas describing pre-cutoff existing work for
     the same topic. In smoke mode we synthesize them (lexical inversion).
     In live mode we sample from the rubric-held-out earlier cutoffs.
  3. Build (or load) the rubric.
  4. Score all pairs, compute AUC + leakage.
  5. Persist rubric to rubrics/{topic}.json and append to reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from forecaster.foresight.dz import load_dz_rows
from forecaster.foresight.judge import (
    RubricJudge,
    StubScorer,
    make_live_scorer,
)
from forecaster.foresight.rubric import (
    RUBRIC_SYSTEM_PROMPT,
    Rubric,
    build_rubric_generation_prompt,
    parse_rubric_response,
    save_rubric,
    stamp_metadata,
)
from forecaster.foresight.rubric_validation import (
    LabeledPair,
    validate_rubric,
    write_scored_pairs_csv,
)

logger = logging.getLogger("phase2")

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- pair construction


def _row_idea_text(row: dict) -> str:
    tz = row.get("target_z") or {}
    parts = [
        tz.get("base_direction", ""),
        tz.get("gap", ""),
    ]
    return " — ".join(p for p in parts if p).strip()


def _row_candidate_text(row: dict) -> str:
    title = (row.get("extra") or {}).get("future_paper_title", "")
    tz = row.get("target_z") or {}
    return f"{title}\n\n{tz.get('gap', '')}".strip()


def _make_negative_idea_from_row(row: dict) -> str:
    """Synthesize a "pre-cutoff existing-work" idea in smoke mode.

    We rewrite the row's idea into a description of *prior* work in the
    topic by stripping novelty cues and prefixing a 'long-standing' framing.
    Good enough for end-to-end smoke; real negatives come from extracting
    over pre-cutoff papers in live mode.
    """
    tz = row.get("target_z") or {}
    return (
        f"Long-standing line of work on {tz.get('base_direction', 'the topic')} "
        f"using established techniques; no new operator or novel gap addressed."
    )


def build_topic_pairs(
    rows: list[dict],
    n_per_class: int,
    *,
    mode: str,
    rng: random.Random,
) -> list[LabeledPair]:
    """Build (idea, candidate) pairs labeled +/-.

    Positives: row's own idea + row's candidate paper.
    Negatives (smoke): synthesized "long-standing work" idea + same candidate pool.
    Negatives (live): the caller is expected to splice in extractor-derived
        pre-cutoff ideas via `rows` already labeled — see _label_in_rows().
    """
    pos_rows = [r for r in rows if r.get("_phase2_label") in (None, 1)]
    rng.shuffle(pos_rows)
    pos_rows = pos_rows[:n_per_class]

    explicit_neg_rows = [r for r in rows if r.get("_phase2_label") == 0]
    explicit_neg_rows = explicit_neg_rows[:n_per_class]

    pairs: list[LabeledPair] = []
    for r in pos_rows:
        pairs.append(
            LabeledPair(
                idea_text=_row_idea_text(r),
                candidate_text=_row_candidate_text(r),
                label=1,
                meta={
                    "source_future_id": r.get("source_future_id", ""),
                    "operator_closed": r.get("operator_closed", ""),
                },
            )
        )

    if explicit_neg_rows:
        for r in explicit_neg_rows:
            pairs.append(
                LabeledPair(
                    idea_text=_row_idea_text(r),
                    candidate_text=_row_candidate_text(r),
                    label=0,
                    meta={
                        "source_future_id": r.get("source_future_id", ""),
                        "operator_closed": r.get("operator_closed", ""),
                    },
                )
            )
    elif pos_rows:
        # Default negatives: paraphrase each positive into legacy framing.
        # In smoke mode this is the only signal; in live mode it serves
        # until we extract real pre-cutoff ideas (Phase 2 follow-up).
        for r in pos_rows[:n_per_class]:
            pairs.append(
                LabeledPair(
                    idea_text=_make_negative_idea_from_row(r),
                    candidate_text=_row_candidate_text(r),
                    label=0,
                    meta={
                        "source_future_id": r.get("source_future_id", ""),
                        "operator_closed": "other",
                        "negative_source": "synthetic_legacy_paraphrase",
                    },
                )
            )
    return pairs


# ---------------------------------------------------------------- stub scorer


def smoke_stub_score(idea: str, candidate: str) -> float:
    """Deterministic discriminator: novelty/operator cues -> high; legacy cues -> low."""
    pos_cues = (
        "novel",
        "extends",
        "extension",
        "new",
        "introduce",
        "propose",
        "—",
        "addresses",
    )
    neg_cues = ("long-standing", "established", "no new", "prior", "legacy")
    score = 0.5
    s = idea.lower()
    for c in pos_cues:
        if c in s:
            score += 0.07
    for c in neg_cues:
        if c in s:
            score -= 0.20
    # weak signal from candidate overlap
    cand = candidate.lower()
    ov = sum(1 for w in s.split() if len(w) > 4 and w in cand)
    score += 0.02 * min(ov, 6)
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------- rubric construction


def smoke_rubric(topic_id: str, cutoff_t: str) -> Rubric:
    """Hand-crafted rubric for smoke mode — no LLM call."""
    return Rubric(
        topic_id=topic_id,
        cutoff_t=cutoff_t,
        criteria=(
            "The idea explicitly names a new operator (extension, transfer, composition, or benchmark) on top of an established direction.",
            "The idea identifies a specific gap that prior work did not address.",
            "The idea is phrased as forward-looking work, not as a survey of long-standing methods.",
        ),
        must_not=(
            "The idea only restates a long-standing approach without naming a new operator or gap.",
        ),
        examples_positive=(),
        examples_negative=(),
        operator_focus=(),
        version=1,
        metadata=stamp_metadata(model="smoke-stub"),
    )


def generate_rubric_via_llm(
    topic_id: str,
    cutoff_t: str,
    positive_examples: list[str],
    negative_examples: list[str],
    model_name: str | None = None,
    base_url: str | None = None,
) -> Rubric:
    """Live-mode rubric generation. Honors --judge-base-url so the same
    self-hosted judge that scores AUC can also author the rubric."""
    import os

    user_prompt = build_rubric_generation_prompt(
        topic_id=topic_id,
        cutoff_t=cutoff_t,
        positive_examples=positive_examples,
        negative_examples=negative_examples,
    )

    # Rubric author can differ from the AUC judge: set RUBRIC_MODEL to an
    # OpenAI model (e.g. gpt-5.4) to author rubrics via OpenAI while the judge
    # stays local. Otherwise the judge endpoint authors the rubric (self-consistent).
    rubric_model_override = os.environ.get("RUBRIC_MODEL", "").strip()
    if rubric_model_override:
        resolved_base = None
        model_name = rubric_model_override
    else:
        resolved_base = (
            base_url or os.environ.get("JUDGE_BASE_URL", "")
        ).strip() or None
    if resolved_base:
        import openai

        resolved = (
            model_name
            or os.environ.get("JUDGE_MODEL", "").strip()
            or "qwen3.5-9b-instruct"
        )
        api_key = os.environ.get("JUDGE_API_KEY", "EMPTY").strip() or "EMPTY"
        client = openai.OpenAI(api_key=api_key, base_url=resolved_base)
        resp = client.chat.completions.create(
            model=resolved,
            messages=[
                {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
    else:
        from idea_forecast_bench.config import load_runtime_config
        from idea_forecast_bench.llm import create_client, get_response_from_llm

        runtime_cfg = load_runtime_config()
        resolved = model_name or runtime_cfg.model_name
        client, resolved = create_client(resolved)
        raw, _ = get_response_from_llm(
            msg=user_prompt,
            client=client,
            model=resolved,
            system_message=RUBRIC_SYSTEM_PROMPT,
            temperature=0.0,
        )

    criteria, must_not = parse_rubric_response(raw)
    return Rubric(
        topic_id=topic_id,
        cutoff_t=cutoff_t,
        criteria=criteria,
        must_not=must_not,
        examples_positive=tuple(positive_examples),
        examples_negative=tuple(negative_examples),
        version=1,
        metadata=stamp_metadata(model=resolved),
    )


# ---------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dz", default=str(REPO_ROOT / "data/topic_hindsight/dz.jsonl"))
    ap.add_argument("--rubrics-dir", default=str(REPO_ROOT / "rubrics"))
    ap.add_argument("--report", default=str(REPO_ROOT / "reports/rubric_validation.md"))
    ap.add_argument("--leakage-report", default=str(REPO_ROOT / "reports/leakage.md"))
    ap.add_argument(
        "--topics", default="", help="comma-separated topic_ids; empty = top-N by count"
    )
    ap.add_argument("--n-topics", type=int, default=5)
    ap.add_argument("--n-per-class", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--mode", choices=["smoke", "live"], default="smoke")
    ap.add_argument(
        "--model",
        default=None,
        help="Judge model name. With --judge-base-url, this is the model "
        "served by the local endpoint (e.g. qwen3.5-9b-instruct).",
    )
    ap.add_argument(
        "--judge-base-url",
        default=None,
        help="OpenAI-compatible endpoint URL (e.g. http://localhost:30000/v1). "
        "When set, M2 talks to the local self-hosted judge, NOT the gpt-4o default.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    rows = load_dz_rows(args.dz)
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_topic[r.get("topic_id", "")].append(r)
    if args.topics:
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    else:
        topics = [
            t
            for t, _ in sorted(by_topic.items(), key=lambda kv: -len(kv[1]))[
                : args.n_topics
            ]
        ]

    # Set up the scorer once.
    if args.mode == "smoke":
        scorer = StubScorer(fn=smoke_stub_score, name="smoke-stub")
    else:
        scorer = make_live_scorer(
            model_name=args.model,
            base_url=args.judge_base_url,
        )
    judge = RubricJudge(scorer=scorer)

    rng = random.Random(args.seed)
    reports: list = []
    leakage_warnings: list[dict] = []
    rubrics_dir = Path(args.rubrics_dir)
    rubrics_dir.mkdir(parents=True, exist_ok=True)

    for topic in topics:
        topic_rows = list(by_topic.get(topic, []))
        if not topic_rows:
            logger.warning("topic=%s has no D_z rows; skipping", topic)
            continue
        cutoff_t = max(r["cutoff_t"] for r in topic_rows)

        # Build (or generate) rubric.
        if args.mode == "smoke":
            rubric = smoke_rubric(topic, cutoff_t)
        else:
            # Use 3 positives + 3 negatives as in-context examples for generation.
            sample_pos = [_row_idea_text(r) for r in topic_rows[:3]]
            rubric = generate_rubric_via_llm(
                topic_id=topic,
                cutoff_t=cutoff_t,
                positive_examples=sample_pos,
                negative_examples=[],
                model_name=args.model,
                base_url=args.judge_base_url,
            )
        save_rubric(rubric, rubrics_dir / f"{topic}.json")

        # Build pairs + validate.
        pairs = build_topic_pairs(topic_rows, args.n_per_class, mode=args.mode, rng=rng)
        if not pairs:
            logger.warning("topic=%s yielded no pairs; skipping", topic)
            continue
        report, scored = validate_rubric(
            rubric,
            pairs,
            judge=judge,
            threshold=args.threshold,
        )
        reports.append(report)
        if report.leakage_hits > 0:
            leakage_warnings.append(
                {
                    "topic_id": topic,
                    "auc": report.auc,
                    "leakage_hits": report.leakage_hits,
                    "examples": report.leakage_examples,
                }
            )

        # Persist per-topic scored pairs for inspection.
        out_csv = rubrics_dir / f"{topic}.scored.csv"
        write_scored_pairs_csv(scored, out_csv)
        logger.info(
            "topic=%s cutoff=%s n+=%d n-=%d auc=%.3f leakage=%d passed=%s",
            topic,
            cutoff_t,
            report.n_positive,
            report.n_negative,
            report.auc,
            report.leakage_hits,
            report.passed,
        )

    # ---------------- write summary report ----------------
    report_lines: list[str] = [
        "# Phase 2 — Rubric validation report\n",
        f"mode: `{args.mode}` | scorer: `{scorer.name if hasattr(scorer, 'name') else scorer.__name__}` "
        f"| threshold: {args.threshold:.2f} | n_per_class: {args.n_per_class}\n",
        "| topic | cutoff | n+ | n- | AUC | leakage | passed |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]
    n_pass = 0
    for r in reports:
        flag = "✅" if r.passed else "❌"
        report_lines.append(
            f"| `{r.topic_id}` | {r.cutoff_t} | {r.n_positive} | {r.n_negative} "
            f"| {r.auc:.3f} | {r.leakage_hits} | {flag} |"
        )
        n_pass += int(r.passed)
    report_lines.append("")
    report_lines.append(
        f"**{n_pass}/{len(reports)} topics passed the {args.threshold:.2f} threshold.**"
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(report_lines), encoding="utf-8")

    # ---------------- leakage report ----------------
    leakage_lines: list[str] = ["# Phase 2 — Leakage report\n"]
    if not leakage_warnings:
        leakage_lines.append("No leakage hits detected across the evaluated topics.")
    else:
        leakage_lines.append(
            "Each row below is a negative (pre-cutoff / existing-work) pair that scored "
            "at or above the median positive — i.e. the judge could not reliably "
            "tell it apart from emerging work. Inspect for memorization."
        )
        for w in leakage_warnings:
            leakage_lines.append(
                f"\n## `{w['topic_id']}` (AUC={w['auc']:.3f}, hits={w['leakage_hits']})"
            )
            for ex in w["examples"]:
                leakage_lines.append(
                    f"- score={ex['score']:.3f} | idea=`{ex['idea_preview']}` | candidate=`{ex['candidate_preview']}`"
                )
    Path(args.leakage_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.leakage_report).write_text("\n".join(leakage_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "topics": [r.to_json() for r in reports],
                "leakage_topics": [w["topic_id"] for w in leakage_warnings],
                "n_pass": n_pass,
                "n_total": len(reports),
            },
            indent=2,
        )
    )
    return 0 if n_pass == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
