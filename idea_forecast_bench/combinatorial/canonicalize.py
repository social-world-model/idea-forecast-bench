from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from idea_forecast_bench.combinatorial.types import ELEMENT_TYPES, ElementType

_ARTICLES = ("a ", "an ", "the ")
_KEEP_RE = re.compile(r"[^a-z0-9 \-]+")
_SPACE_RE = re.compile(r"\s+")


def _singularize(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(("ss", "us", "is", "ics")):
        return token
    if token.endswith("s"):
        return token[:-1]
    return token


def normalize_text(text: str, aliases: Mapping[str, str] | None = None) -> str:
    """Canonical surface form: lowercase ASCII, alias-expanded, head noun
    singularised. Pure and deterministic; the merge step sits on top of it."""
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = value.replace("_", " ").replace("/", " ").replace("&", " and ")
    value = _KEEP_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value).strip(" -")
    for article in _ARTICLES:
        if value.startswith(article):
            value = value[len(article) :]
            break
    if not value:
        return ""
    alias_map = aliases or {}
    if value in alias_map:
        value = alias_map[value]
    tokens = [alias_map.get(tok, tok) for tok in value.split(" ") if tok]
    if not tokens:
        return ""
    # Alias expansion may introduce multi-word replacements; re-split.
    tokens = " ".join(tokens).split(" ")
    tokens[-1] = _singularize(tokens[-1])
    return " ".join(tokens)


def element_key(element_type: ElementType, text: str) -> str:
    return f"{element_type}:{text}"


def split_key(key: str) -> tuple[ElementType, str]:
    type_part, _, text = key.partition(":")
    for known in ELEMENT_TYPES:
        if type_part == known:
            return known, text
    raise ValueError(f"malformed element key: {key!r}")


def _unit(vec: Sequence[float]) -> NDArray[np.float64]:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def merge_elements(
    counts: Mapping[str, int],
    vectors: Mapping[str, Sequence[float]],
    threshold: float,
) -> dict[str, str]:
    """Greedy leader clustering within each element type.

    Keys are visited in ``(-count, key)`` order, so the most frequent surface
    form of a cluster becomes its leader and the result is deterministic
    without any RNG. Returns ``member_key -> leader_key`` for every input key
    (leaders map to themselves). Keys without a vector are their own leader."""
    leader_of: dict[str, str] = {}
    by_type: dict[str, list[str]] = {}
    for key in counts:
        by_type.setdefault(key.partition(":")[0], []).append(key)

    for keys in by_type.values():
        ordered = sorted(keys, key=lambda k: (-counts[k], k))
        leaders: list[str] = []
        matrix: NDArray[np.float64] | None = None
        for key in ordered:
            vec = vectors.get(key)
            if vec is None or len(vec) == 0:
                leader_of[key] = key
                continue
            unit = _unit(vec)
            if matrix is not None and matrix.shape[0] > 0:
                sims = matrix @ unit
                best = int(np.argmax(sims))
                if float(sims[best]) >= threshold:
                    leader_of[key] = leaders[best]
                    continue
            leader_of[key] = key
            leaders.append(key)
            matrix = (
                unit[np.newaxis, :]
                if matrix is None
                else np.vstack([matrix, unit[np.newaxis, :]])
            )
    return leader_of
