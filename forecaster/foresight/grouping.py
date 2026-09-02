from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)


def _coerce_extra(extra: Any) -> dict[str, Any]:
    if isinstance(extra, dict):
        return extra
    if isinstance(extra, str) and extra.strip():
        try:
            # json.loads is typed as returning Any; bind it to a typed local.
            parsed: dict[str, Any] = json.loads(extra)
        except json.JSONDecodeError:
            return {}
        return parsed
    return {}


def _group_key(extra: dict[str, Any]) -> tuple[str, tuple[str, str, str]]:
    """Return (cutoff_date, (b, o, g)) — the locked group-identity tuple."""
    cutoff = str(extra.get("cutoff_date") or "")
    inno = extra.get("innovation") or {}
    z = (
        str(inno.get("base_direction") or ""),
        str(inno.get("operator") or ""),
        str(inno.get("gap") or ""),
    )
    return cutoff, z


class GroupingInvariantError(AssertionError):
    """Raised when the TRL grouping invariant is violated mid-training."""


def assert_group_invariant(
    extra_infos: Sequence[Any],
    *,
    num_generations: int,
    min_group_size: int = 2,
) -> None:
    """Enforce: batch is contiguous groups of `num_generations` rollouts,
    each group shares one (cutoff_t, z), and each group has size >= min_group_size.
    """
    n = len(extra_infos)
    if n == 0:
        raise GroupingInvariantError("empty batch")
    if num_generations < min_group_size:
        raise GroupingInvariantError(
            f"num_generations={num_generations} < min_group_size={min_group_size}; "
            "group-relative advantage normalization is degenerate"
        )
    if n % num_generations != 0:
        raise GroupingInvariantError(
            f"batch size {n} is not a multiple of num_generations={num_generations}"
        )
    for g_start in range(0, n, num_generations):
        group = extra_infos[g_start : g_start + num_generations]
        keys = {_group_key(_coerce_extra(e)) for e in group}
        if len(keys) != 1:
            sample = list(keys)[:3]
            raise GroupingInvariantError(
                f"group at offset {g_start} contains {len(keys)} distinct "
                f"(cutoff_t, z) keys; sample={sample}"
            )
        cutoff, z = keys.pop()
        if not cutoff or not any(z):
            raise GroupingInvariantError(
                f"group at offset {g_start} has an empty (cutoff_t, z) key: "
                f"cutoff={cutoff!r} z={z!r}"
            )


def grouping_report(
    extra_infos: Sequence[Any],
    *,
    num_generations: int,
) -> dict[str, Any]:
    """Lightweight diagnostic — same checks as the assert but returns a dict.

    Useful for first-step logging in the trainer; pairs with the strict
    `assert_group_invariant` which is meant to halt training on drift.
    """
    n = len(extra_infos)
    out: dict[str, Any] = {
        "batch_size": n,
        "num_generations": num_generations,
        "num_groups": 0,
        "violations": [],
        "key_histogram_sample": [],
    }
    if n == 0 or num_generations <= 0 or n % num_generations != 0:
        out["violations"].append(
            f"non-divisible batch: n={n} num_generations={num_generations}"
        )
        return out
    keys: list[tuple[str, tuple[str, str, str]]] = []
    for g_start in range(0, n, num_generations):
        group = extra_infos[g_start : g_start + num_generations]
        group_keys = {_group_key(_coerce_extra(e)) for e in group}
        if len(group_keys) != 1:
            out["violations"].append(
                f"group@{g_start}: distinct_keys={len(group_keys)}"
            )
            continue
        keys.append(group_keys.pop())
    out["num_groups"] = n // num_generations
    counter = Counter(keys)
    out["key_histogram_sample"] = [
        {"cutoff": k[0], "operator": k[1][1], "count": v}
        for k, v in counter.most_common(5)
    ]
    return out


# --------------------------------------------------------------------------- in-group dedup penalty


def _token_set(text: str) -> set[str]:
    return {w for w in (text or "").lower().split() if len(w) > 3}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union


def compute_dedup_penalties(
    completions: Sequence[str],
    *,
    num_generations: int,
    threshold: float = 0.85,
    penalty: float = 0.1,
) -> list[float]:
    """Return per-completion subtractive penalties for near-duplicates within a group.

    A completion gets a penalty equal to `penalty * k` where k is the number
    of other rollouts in its group with Jaccard(token-set) >= threshold.
    """
    out = [0.0] * len(completions)
    if penalty <= 0.0 or num_generations < 2 or threshold <= 0.0:
        return out
    for g_start in range(0, len(completions), num_generations):
        chunk = list(completions[g_start : g_start + num_generations])
        token_sets = [_token_set(c) for c in chunk]
        for i in range(len(chunk)):
            dupes = 0
            for j in range(len(chunk)):
                if i == j:
                    continue
                if jaccard(token_sets[i], token_sets[j]) >= threshold:
                    dupes += 1
            out[g_start + i] = penalty * dupes
    return out


__all__ = [
    "GroupingInvariantError",
    "assert_group_invariant",
    "grouping_report",
    "compute_dedup_penalties",
]
