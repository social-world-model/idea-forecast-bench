from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections.abc import Iterable, Sequence

from idea_forecast_bench.combinatorial.config import SamplerConfig, StateConfig
from idea_forecast_bench.combinatorial.state import (
    freshness,
    lift,
    rising,
    unpaired_bonus,
)
from idea_forecast_bench.combinatorial.types import (
    MOVES,
    Combo,
    CommunityState,
    Element,
    ElementType,
)

VARIANT_FULL = "full"
VARIANT_FREQUENCY = "frequency"
VARIANT_INDEPENDENT = "independent"
VARIANT_RANDOM = "random"
VARIANTS: tuple[str, ...] = (
    VARIANT_FULL,
    VARIANT_FREQUENCY,
    VARIANT_INDEPENDENT,
    VARIANT_RANDOM,
)


def window_seed(cutoff_month: str, train_ids: Iterable[str], variant: str) -> int:
    payload = f"{cutoff_month}|{','.join(sorted(train_ids))}|{variant}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _vocabulary(
    state: CommunityState, min_count: int
) -> dict[ElementType, list[Element]]:
    vocab: dict[ElementType, list[Element]] = {}
    for element in state.elements.values():
        if element.count >= min_count:
            vocab.setdefault(element.type, []).append(element)
    for items in vocab.values():
        items.sort(key=lambda e: (-e.heat, e.id))
    return vocab


def _heat_norm(state: CommunityState, element: Element) -> float:
    return element.heat / state.max_heat if state.max_heat > 0 else 0.0


def _components(
    state: CommunityState,
    elements: Sequence[Element],
    sampler_cfg: SamplerConfig,
    state_cfg: StateConfig,
    *,
    use_freshness: bool,
) -> tuple[float, dict[str, float]]:
    heat_product = 1.0
    for element in elements:
        heat_product *= _heat_norm(state, element)
    pairs = list(itertools.combinations([e.id for e in elements], 2))
    fresh_values = [freshness(state, a, b, sampler_cfg, state_cfg) for a, b in pairs]
    rising_values = [
        rising(state, a, b, state_cfg, sampler_cfg.rising_log_clip) for a, b in pairs
    ]
    unpaired_values = [unpaired_bonus(state, a, b) for a, b in pairs]
    lift_values = [lift(state, a, b) for a, b in pairs]
    fresh = (
        math.exp(sum(math.log(v) for v in fresh_values) / len(fresh_values))
        if fresh_values
        else 1.0
    )
    score = heat_product * (fresh if use_freshness else 1.0)
    components = {
        "heat_product": heat_product,
        "freshness": fresh,
        "rising_max": max(rising_values) if rising_values else 0.0,
        "unpaired_max": max(unpaired_values) if unpaired_values else 0.0,
        "lift_mean": sum(lift_values) / len(lift_values) if lift_values else 0.0,
    }
    return score, components


def _draw_move(state: CommunityState, rng: random.Random, *, uniform: bool) -> str:
    if uniform:
        return rng.choice(MOVES)
    weights = [state.move_dist.get(m, 0.0) for m in MOVES]
    if sum(weights) <= 0:
        return rng.choice(MOVES)
    return rng.choices(MOVES, weights=weights, k=1)[0]


def _usable_patterns(
    vocab: dict[ElementType, list[Element]], cfg: SamplerConfig
) -> list[tuple[ElementType, ...]]:
    return [p for p in cfg.type_patterns if all(vocab.get(t) for t in p)]


def _enumerate_candidates(
    state: CommunityState,
    vocab: dict[ElementType, list[Element]],
    sampler_cfg: SamplerConfig,
    state_cfg: StateConfig,
    *,
    use_freshness: bool,
) -> list[tuple[tuple[Element, ...], float, dict[str, float]]]:
    candidates: list[tuple[tuple[Element, ...], float, dict[str, float]]] = []
    for pattern in _usable_patterns(vocab, sampler_cfg):
        top_m = (
            sampler_cfg.top_m_triple
            if len(pattern) >= 3
            else sampler_cfg.top_m_per_type
        )
        pools = [vocab[t][:top_m] for t in pattern]
        for elements in itertools.product(*pools):
            score, components = _components(
                state, elements, sampler_cfg, state_cfg, use_freshness=use_freshness
            )
            if score > 0:
                candidates.append((tuple(elements), score, components))
    return candidates


def _weighted_without_reuse(
    candidates: list[tuple[tuple[Element, ...], float, dict[str, float]]],
    k: int,
    rng: random.Random,
    gamma: float,
) -> list[tuple[tuple[Element, ...], float, dict[str, float]]]:
    weights = [c[1] ** gamma for c in candidates]
    chosen: list[tuple[tuple[Element, ...], float, dict[str, float]]] = []
    used: set[str] = set()
    while len(chosen) < k and any(w > 0 for w in weights):
        index = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
        elements, score, components = candidates[index]
        chosen.append((elements, score, components))
        used.update(e.id for e in elements)
        weights = [
            0.0 if any(e.id in used for e in c[0]) else w
            for c, w in zip(candidates, weights, strict=True)
        ]
    if len(chosen) < k:
        # The no-reuse rule exhausted the vocabulary (small windows). Top up
        # with distinct combos, allowing elements to repeat, rather than
        # returning a short window.
        taken = {tuple(e.id for e in c[0]) for c in chosen}
        rest = [c for c in candidates if tuple(e.id for e in c[0]) not in taken]
        rest_weights = [c[1] ** gamma for c in rest]
        while len(chosen) < k and rest and any(w > 0 for w in rest_weights):
            index = rng.choices(range(len(rest)), weights=rest_weights, k=1)[0]
            chosen.append(rest.pop(index))
            rest_weights.pop(index)
    return chosen


def _slot_draws(
    state: CommunityState,
    vocab: dict[ElementType, list[Element]],
    patterns: Sequence[tuple[ElementType, ...]],
    k: int,
    rng: random.Random,
    *,
    uniform: bool,
) -> list[tuple[Element, ...]]:
    draws: list[tuple[Element, ...]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(draws) < k and attempts < k * 20:
        attempts += 1
        pattern = rng.choice(list(patterns))
        elements: list[Element] = []
        for element_type in pattern:
            pool = vocab[element_type]
            if uniform:
                elements.append(rng.choice(pool))
            else:
                weights = [_heat_norm(state, e) for e in pool]
                if sum(weights) <= 0:
                    elements.append(rng.choice(pool))
                else:
                    elements.append(rng.choices(pool, weights=weights, k=1)[0])
        ids = tuple(sorted(e.id for e in elements))
        if len(set(ids)) != len(ids) or ids in seen:
            continue
        seen.add(ids)
        draws.append(tuple(elements))
    return draws


def sample_combos(
    state: CommunityState,
    variant: str,
    k: int,
    rng: random.Random,
    sampler_cfg: SamplerConfig,
    state_cfg: StateConfig,
    min_count: int,
) -> list[Combo]:
    """``k`` combinations under one sampling rule.

    full         heat x co-occurrence freshness, no element reused
    frequency    heat only (freshness == 1)
    independent  Llull: each slot drawn by heat independently, pairs ignored
    random       uniform over the sampleable vocabulary and moves
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown sampler variant {variant!r}; choose {VARIANTS}")
    vocab = _vocabulary(state, min_count)
    patterns = _usable_patterns(vocab, sampler_cfg)
    if not patterns and min_count > 1:
        # Small window: the frequent vocabulary does not span any pattern, so
        # fall back to every element seen at least once.
        vocab = _vocabulary(state, 1)
        patterns = _usable_patterns(vocab, sampler_cfg)
    if not patterns:
        return []

    if variant in (VARIANT_FULL, VARIANT_FREQUENCY):
        use_freshness = variant == VARIANT_FULL
        candidates = _enumerate_candidates(
            state, vocab, sampler_cfg, state_cfg, use_freshness=use_freshness
        )
        chosen = _weighted_without_reuse(candidates, k, rng, sampler_cfg.score_gamma)
        return [
            Combo(
                elements=elements,
                move=_draw_move(state, rng, uniform=False),
                sampler=variant,
                score=score,
                components=components,
            )
            for elements, score, components in chosen
        ]

    uniform = variant == VARIANT_RANDOM
    draws = _slot_draws(state, vocab, patterns, k, rng, uniform=uniform)
    combos: list[Combo] = []
    for elements in draws:
        score, components = _components(
            state, elements, sampler_cfg, state_cfg, use_freshness=True
        )
        combos.append(
            Combo(
                elements=elements,
                move=_draw_move(state, rng, uniform=uniform),
                sampler=variant,
                score=score,
                components=components,
            )
        )
    return combos
