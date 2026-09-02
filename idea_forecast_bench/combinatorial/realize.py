from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any

from idea_forecast_bench.combinatorial.config import PromptPair, RealizeConfig
from idea_forecast_bench.combinatorial.extraction import extract_json_object
from idea_forecast_bench.combinatorial.llm_caller import TextCaller
from idea_forecast_bench.combinatorial.types import (
    MOVE_DEFINITIONS,
    Combo,
    CommunityState,
    ParsedIdea,
)
from idea_forecast_bench.models import IdeaPrediction

TEMPLATE_MODEL = "template"


def _heat_rank(state: CommunityState, element_id: str) -> int:
    element = state.elements[element_id]
    same_type = [e for e in state.elements.values() if e.type == element.type]
    same_type.sort(key=lambda e: (-e.heat, e.id))
    return next(i for i, e in enumerate(same_type, start=1) if e.id == element_id)


def _signal_line(combo: Combo) -> str:
    parts: list[str] = []
    rising_max = combo.components.get("rising_max", 0.0)
    unpaired_max = combo.components.get("unpaired_max", 0.0)
    if unpaired_max > 0:
        parts.append("these elements are individually hot but have never co-occurred")
    if rising_max > 0.3:
        parts.append(
            f"their co-occurrence is rising (~{2**rising_max:.1f}x over the last months)"
        )
    if not parts:
        parts.append("co-occurrence is stable")
    return "; ".join(parts)


def format_combos_block(combos: Sequence[Combo], state: CommunityState) -> str:
    lines: list[str] = []
    for index, combo in enumerate(combos, start=1):
        lines.append(f"[{index}]")
        for element in combo.elements:
            lines.append(
                f'  {element.type}: "{element.label}" '
                f"(rank {_heat_rank(state, element.id)} by heat among {element.type}s, "
                f"{element.count} papers, first seen {element.first_seen[:7]})"
            )
        definition = MOVE_DEFINITIONS.get(combo.move, "apply this operation")
        lines.append(f"  move: {combo.move} -- {definition}")
        lines.append(f"  signal: {_signal_line(combo)}")
        if combo.evidence:
            lines.append("  supporting pre-cutoff papers:")
            for ev in combo.evidence:
                lines.append(f"    - [{ev.month}] {ev.title}: {ev.snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _coerce_idea(item: Mapping[str, Any], fallback_index: int) -> ParsedIdea | None:
    title = str(item.get("title") or "").strip()
    if not title:
        return None
    raw_index = item.get("combo_index")
    try:
        combo_index = int(raw_index) if raw_index is not None else fallback_index
    except (TypeError, ValueError):
        combo_index = fallback_index
    try:
        confidence = float(item.get("confidence") or 0.5)
    except (TypeError, ValueError):
        confidence = 0.5
    key_terms_raw = item.get("key_terms") or []
    key_terms = (
        tuple(str(t).strip() for t in key_terms_raw if str(t).strip())
        if isinstance(key_terms_raw, list)
        else ()
    )
    return ParsedIdea(
        combo_index=combo_index,
        title=title,
        rationale=str(item.get("rationale") or "").strip(),
        approach=str(item.get("approach") or "").strip(),
        confidence=max(0.0, min(1.0, confidence)),
        key_terms=key_terms,
    )


def parse_ideas(raw: str, n: int) -> dict[int, ParsedIdea]:
    """Ideas keyed by combo_index (1-based); positional fallback when the
    model dropped the index. Extra or out-of-range indices are ignored."""
    payload = extract_json_object(raw)
    items: list[Any] = []
    if payload is not None:
        ideas = payload.get("ideas") or payload.get("predictions")
        if isinstance(ideas, list):
            items = ideas
    out: dict[int, ParsedIdea] = {}
    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        idea = _coerce_idea(item, position)
        if idea is None or not 1 <= idea.combo_index <= n:
            continue
        out.setdefault(idea.combo_index, idea)
    return out


def template_idea(combo: Combo, index: int) -> ParsedIdea:
    """No-LLM realisation used by dry runs and as the last-resort pad."""
    labels = [e.label for e in combo.elements]
    by_type = {e.type: e.label for e in combo.elements}
    theme = by_type.get("theme", labels[0])
    method = by_type.get("method", labels[-1])
    domain = by_type.get("domain")
    title = f"{combo.move.title()}: {theme} via {method}"
    if domain:
        title += f" for {domain}"
    return ParsedIdea(
        combo_index=index,
        title=title,
        rationale=(
            f"Elements {', '.join(labels)} are active in the pre-cutoff literature; "
            f"the move '{combo.move}' composes them."
        ),
        approach=f"Apply {method} to {theme}" + (f" in {domain}." if domain else "."),
        confidence=0.5,
        key_terms=tuple(labels),
    )


def _metadata(
    combo: Combo,
    *,
    variant: str,
    seed: int,
    state: CommunityState,
    fallback: bool,
) -> dict[str, Any]:
    return {
        "strategy": "combinatorial",
        "variant": variant,
        "seed": seed,
        "move": combo.move,
        "combo_score": round(combo.score, 6),
        "components": {k: round(v, 6) for k, v in combo.components.items()},
        "elements": [
            {
                "id": e.id,
                "type": e.type,
                "label": e.label,
                "count": e.count,
                "heat": round(e.heat, 4),
                "first_seen": e.first_seen,
                "paper_ids": list(e.paper_ids[:10]),
            }
            for e in combo.elements
        ],
        "evidence_paper_ids": [ev.paper_id for ev in combo.evidence],
        "state": {
            "cutoff_date": state.cutoff_date,
            "n_train": state.n_train,
            "n_with_records": state.n_with_records,
            "coverage": round(state.coverage, 4),
            "n_elements": len(state.elements),
        },
        "fallback": fallback,
    }


def to_prediction(
    rank: int,
    idea: ParsedIdea,
    combo: Combo,
    *,
    variant: str,
    seed: int,
    state: CommunityState,
    fallback: bool,
) -> IdeaPrediction:
    key_terms = list(idea.key_terms)
    for element in combo.elements:
        if element.label not in key_terms:
            key_terms.append(element.label)
    return IdeaPrediction(
        rank=rank,
        title=idea.title,
        rationale=idea.rationale,
        approach=idea.approach,
        score=round(combo.score, 6),
        confidence=idea.confidence,
        key_terms=key_terms,
        metadata=_metadata(
            combo, variant=variant, seed=seed, state=state, fallback=fallback
        ),
    )


def realize_combos(
    combos: Sequence[Combo],
    state: CommunityState,
    *,
    caller: TextCaller | None,
    prompt: PromptPair,
    cfg: RealizeConfig,
    cutoff_month: str,
    horizon_months: int,
    variant: str,
    seed: int,
) -> list[IdeaPrediction]:
    """One LLM call for every combo, one retry for the combos it missed, then
    template padding (flagged) if allowed. ``caller=None`` is template-only."""
    n = len(combos)
    if n == 0:
        return []
    ideas: dict[int, ParsedIdea] = {}
    if caller is not None:
        user = prompt.user_template.format(
            cutoff_month=cutoff_month,
            horizon_months=horizon_months,
            n=n,
            combos_block=format_combos_block(combos, state),
        )
        temperatures = (cfg.temperature, cfg.temperature + 0.1)
        for attempt, temperature in enumerate(temperatures):
            if attempt > 0:
                missing = [i for i in range(1, n + 1) if i not in ideas]
                print(
                    f"[combinatorial WARNING] cutoff={cutoff_month}: missing ideas for "
                    f"combos {missing} -- retrying",
                    file=sys.stderr,
                    flush=True,
                )
            raw = caller.complete(
                prompt.system_prompt,
                user,
                temperature=temperature,
                top_p=cfg.top_p,
                seed=seed + attempt,
            )
            for index, parsed in parse_ideas(raw, n).items():
                ideas.setdefault(index, parsed)
            if len(ideas) >= n:
                break

    predictions: list[IdeaPrediction] = []
    for index, combo in enumerate(combos, start=1):
        found = ideas.get(index)
        fallback = found is None
        if found is None:
            if caller is not None and not cfg.fallback_template:
                continue
            found = template_idea(combo, index)
        idea: ParsedIdea = found
        predictions.append(
            to_prediction(
                len(predictions) + 1,
                idea,
                combo,
                variant=variant,
                seed=seed,
                state=state,
                fallback=fallback and caller is not None,
            )
        )
    return predictions
