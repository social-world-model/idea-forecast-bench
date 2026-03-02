"""Output validation and normalization layer for generated research ideas.

This module provides a stable contract between raw LLM output (or any upstream
source) and the downstream consumers:

- ``_transform_ideas_for_dashboard`` in ``backend/app.py`` – reads PascalCase
  keys and coerces numeric fields via ``float(idea.get("Score", 0))``.
- ``RunService._build_report`` in ``backend/services/run_service.py`` – reads
  ``Score``, ``Novelty``, ``Feasibility`` directly via ``_safe_float``.

Legacy key contract (PascalCase, as expected downstream):
    Title           str   – defaults to "Untitled Idea"
    Problem         str   – defaults to ""
    Approach        str   – defaults to ""
    Score           float – defaults to 0.0
    Novelty         float – defaults to 0.0
    Feasibility     float – defaults to 0.0
    Interestingness float – defaults to 0.0
    source_url      Any   – defaults to None

Numeric coercion rules:
    - int / float / str(numeric)  → float  (accepted)
    - None / missing              → default value 0.0
    - str(non-numeric)            → raises ValueError
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NUMERIC_FIELDS: tuple[str, ...] = (
    "Score",
    "Novelty",
    "Feasibility",
    "Interestingness",
)

_STRING_FIELDS: tuple[str, ...] = ("Title", "Problem", "Approach")

_STRING_DEFAULTS: dict[str, str] = {
    "Title": "Untitled Idea",
    "Problem": "",
    "Approach": "",
}


def _coerce_numeric(field: str, value: Any) -> float:
    """Coerce *value* to float for the named numeric *field*.

    Rules
    -----
    - ``None`` or key absent (represented by sentinel ``_MISSING``) → ``0.0``
    - ``int`` / ``float``                                           → ``float``
    - ``str``                                                       →
        - parseable as float → ``float``
        - not parseable      → ``ValueError``

    Any other type that cannot be converted via ``float()`` raises
    ``ValueError`` as well.
    """
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Field '{field}' has a non-coercible value: {value!r}. "
            "Expected a numeric value or a string representation of a number."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_idea(raw: dict) -> dict:
    """Return a *new* dict with all legacy keys normalised.

    Parameters
    ----------
    raw:
        Raw dict as produced by the idea generator or loaded from JSON.

    Returns
    -------
    dict
        A new dict containing at least the following keys with the correct
        types:

        =================  ======  ============================
        Key                Type    Fallback
        =================  ======  ============================
        Title              str     "Untitled Idea"
        Problem            str     ""
        Approach           str     ""
        Score              float   0.0
        Novelty            float   0.0
        Feasibility        float   0.0
        Interestingness    float   0.0
        source_url         Any     None
        =================  ======  ============================

        All other keys from *raw* are passed through unchanged.

    Raises
    ------
    ValueError
        If a numeric field is present in *raw* but cannot be coerced to
        ``float`` (e.g. ``Score="abc"``).
    """
    # Start with a shallow copy so the caller's dict is not mutated.
    result: dict = dict(raw)

    # Normalize string fields
    for field in _STRING_FIELDS:
        if field not in result or result[field] is None:
            result[field] = _STRING_DEFAULTS[field]
        else:
            result[field] = str(result[field])

    # Normalize numeric fields
    for field in _NUMERIC_FIELDS:
        raw_value = result.get(field)
        result[field] = _coerce_numeric(field, raw_value)

    # Normalize source_url: keep as-is but ensure key is present
    if "source_url" not in result:
        result["source_url"] = None

    return result


def normalize_ideas(raw_list: list[dict]) -> list[dict]:
    """Apply :func:`normalize_idea` to every element of *raw_list*.

    Parameters
    ----------
    raw_list:
        A list of raw idea dicts.

    Returns
    -------
    list[dict]
        A new list of normalised idea dicts.  The original list and its
        elements are never mutated.

    Raises
    ------
    ValueError
        If any element contains a non-coercible numeric field.
    """
    return [normalize_idea(idea) for idea in raw_list]
