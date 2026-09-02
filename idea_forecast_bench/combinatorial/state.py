from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from idea_forecast_bench.combinatorial.canonicalize import (
    element_key,
    merge_elements,
    split_key,
)
from idea_forecast_bench.combinatorial.config import (
    CanonicalizeConfig,
    SamplerConfig,
    StateConfig,
)
from idea_forecast_bench.combinatorial.types import (
    MOVES,
    UNKNOWN_MOVE,
    CommunityState,
    Element,
    ExtractionRecord,
    PairKey,
    pair_key,
)
from idea_forecast_bench.models import PaperRecord
from idea_forecast_bench.papers import date_to_ordinal, get_paper_published_date

DAYS_PER_MONTH = 30.44


def months_before(cutoff_ordinal: int, published_date: str) -> float:
    return max(0.0, (cutoff_ordinal - date_to_ordinal(published_date)) / DAYS_PER_MONTH)


def decay_weight(age_months: float, half_life_months: float) -> float:
    return float(2.0 ** (-age_months / half_life_months))


def _uniform_moves() -> dict[str, float]:
    return dict.fromkeys(MOVES, 1.0 / len(MOVES))


def _move_distribution(moves: Sequence[str]) -> dict[str, float]:
    counts = Counter(m for m in moves if m != UNKNOWN_MOVE and m in MOVES)
    total = sum(counts.values())
    if total == 0:
        return _uniform_moves()
    return {m: counts.get(m, 0) / total for m in MOVES}


def build_state(
    train_papers: Sequence[PaperRecord],
    records: Mapping[str, ExtractionRecord],
    cutoff_date: str,
    state_cfg: StateConfig,
    canon_cfg: CanonicalizeConfig,
    vectors: Mapping[str, Sequence[float]],
    *,
    min_count: int | None = None,
) -> CommunityState:
    """Community state at ``cutoff_date`` from the train papers only.

    Every quantity is computed from papers dated on or before the cutoff:
    the function raises if it is handed anything later, so a caller cannot
    leak the future in by accident."""
    cutoff_ord = date_to_ordinal(cutoff_date)
    usable: list[tuple[PaperRecord, ExtractionRecord, float]] = []
    for paper in train_papers:
        published = get_paper_published_date(paper)
        if date_to_ordinal(published) > cutoff_ord:
            raise ValueError(
                f"paper {paper.paper_id} dated {published} is after cutoff {cutoff_date}"
            )
        record = records.get(paper.paper_id)
        if record is None or not record.ok:
            continue
        usable.append((paper, record, months_before(cutoff_ord, published)))

    raw_counts: Counter[str] = Counter()
    per_paper_raw: list[set[str]] = []
    for _paper, record, _age in usable:
        keys = {element_key(t, text) for t, text in record.typed_elements() if text}
        per_paper_raw.append(keys)
        raw_counts.update(keys)

    leader_of = merge_elements(raw_counts, vectors, canon_cfg.merge_threshold)

    members: dict[str, set[str]] = defaultdict(set)
    paper_ids: dict[str, set[str]] = defaultdict(set)
    first_seen: dict[str, str] = {}
    heat: dict[str, float] = defaultdict(float)
    pair_count: dict[PairKey, int] = defaultdict(int)
    pair_heat: dict[PairKey, float] = defaultdict(float)
    pair_recent: dict[PairKey, int] = defaultdict(int)
    pair_older: dict[PairKey, int] = defaultdict(int)
    recent_moves: list[str] = []
    all_moves: list[str] = []
    max_age = 0.0

    for (paper, record, age), raw_keys in zip(usable, per_paper_raw, strict=True):
        canonical = sorted({leader_of[k] for k in raw_keys})
        weight = decay_weight(age, state_cfg.half_life_months)
        published = get_paper_published_date(paper)
        max_age = max(max_age, age)
        is_recent = age <= state_cfg.recent_months
        all_moves.append(record.move)
        if is_recent:
            recent_moves.append(record.move)
        for raw in raw_keys:
            members[leader_of[raw]].add(raw)
        for cid in canonical:
            paper_ids[cid].add(paper.paper_id)
            heat[cid] += weight
            if cid not in first_seen or published < first_seen[cid]:
                first_seen[cid] = published
        for i, a in enumerate(canonical):
            for b in canonical[i + 1 :]:
                key = pair_key(a, b)
                pair_count[key] += 1
                pair_heat[key] += weight
                if is_recent:
                    pair_recent[key] += 1
                else:
                    pair_older[key] += 1

    elements: dict[str, Element] = {}
    for cid, ids in paper_ids.items():
        element_type, label = split_key(cid)
        elements[cid] = Element(
            id=cid,
            type=element_type,
            label=label,
            variants=tuple(sorted(split_key(m)[1] for m in members[cid])),
            paper_ids=tuple(sorted(ids)),
            first_seen=first_seen[cid],
            count=len(ids),
            heat=heat[cid],
        )

    recent_span = max(state_cfg.recent_months, 1e-9)
    older_span = max(1.0, max_age - state_cfg.recent_months)
    pair_recent_rate = {k: v / recent_span for k, v in pair_recent.items()}
    pair_older_rate = {k: v / older_span for k, v in pair_older.items()}

    threshold = min_count if min_count is not None else canon_cfg.min_count
    heats = [e.heat for e in elements.values() if e.count >= threshold]
    max_heat = max(heats) if heats else 0.0
    hot_threshold = (
        float(np.quantile(np.asarray(heats), state_cfg.hot_quantile)) if heats else 0.0
    )

    return CommunityState(
        cutoff_date=cutoff_date,
        n_train=len(train_papers),
        n_with_records=len(usable),
        elements=elements,
        pair_count=dict(pair_count),
        pair_heat=dict(pair_heat),
        pair_recent_rate=pair_recent_rate,
        pair_older_rate=pair_older_rate,
        move_dist=_move_distribution(recent_moves or all_moves),
        max_heat=max_heat,
        hot_threshold=hot_threshold,
    )


# ---- pair signals ----------------------------------------------------------


def rising(
    state: CommunityState, a: str, b: str, state_cfg: StateConfig, clip: float
) -> float:
    """log2 ratio of recent to older co-occurrence rate, smoothed and clipped."""
    key = pair_key(a, b)
    alpha = state_cfg.smoothing_alpha
    recent = state.pair_recent_rate.get(key, 0.0)
    older = state.pair_older_rate.get(key, 0.0)
    value = math.log2((recent + alpha) / (older + alpha))
    return max(-clip, min(clip, value))


def unpaired_bonus(state: CommunityState, a: str, b: str) -> float:
    """Reward for two hot elements that have never co-occurred."""
    if state.pair_count.get(pair_key(a, b), 0) > 0 or state.max_heat <= 0:
        return 0.0
    ha, hb = state.elements[a].heat, state.elements[b].heat
    if ha < state.hot_threshold or hb < state.hot_threshold:
        return 0.0
    return min(ha, hb) / state.max_heat


def lift(state: CommunityState, a: str, b: str) -> float:
    pair = state.pair_count.get(pair_key(a, b), 0)
    if pair == 0 or state.n_with_records == 0:
        return 0.0
    ca, cb = state.elements[a].count, state.elements[b].count
    return (pair * state.n_with_records) / (ca * cb) if ca and cb else 0.0


def freshness(
    state: CommunityState,
    a: str,
    b: str,
    sampler_cfg: SamplerConfig,
    state_cfg: StateConfig,
) -> float:
    r = rising(state, a, b, state_cfg, sampler_cfg.rising_log_clip)
    u = unpaired_bonus(state, a, b)
    return (
        1.0 + sampler_cfg.lambda_rising * max(0.0, r) + sampler_cfg.lambda_unpaired * u
    )
