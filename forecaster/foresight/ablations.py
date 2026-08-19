"""Phase-8 ablation switches.

Each ablation flips a single switch on the foresight reward + context.
A run is described by an `AblationConfig` row; the runner produces an
identically-shaped row of metrics. This module owns *only* the toggles
and the metric-record shape; the actual eval harness lives in
scripts/phase8_ablations.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The single switches mandated by the plan.
REWARD_VARIANTS: tuple[str, ...] = ("foresight", "embedding_threshold", "raw_judge")
DECOMPOSITION_VARIANTS: tuple[str, ...] = ("per_z", "single_shot_k")
RUBRIC_VARIANTS: tuple[str, ...] = ("static", "co_evolve")
GATE_VARIANTS: tuple[str, ...] = ("both", "no_grounding", "no_operator", "neither")


@dataclass(frozen=True)
class AblationConfig:
    """One ablation cell. Set exactly one field away from the baseline."""

    name: str
    reward_variant: str = "foresight"           # ours
    decomposition_variant: str = "per_z"        # ours
    rubric_variant: str = "static"              # ours (Phase 6 off)
    gate_variant: str = "both"                  # ours

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reward_variant": self.reward_variant,
            "decomposition_variant": self.decomposition_variant,
            "rubric_variant": self.rubric_variant,
            "gate_variant": self.gate_variant,
        }


def baseline_set() -> list[AblationConfig]:
    """Return the 8-row ablation grid called out in the plan (Phase 8)."""
    out: list[AblationConfig] = [AblationConfig(name="ours")]
    for v in REWARD_VARIANTS:
        if v != "foresight":
            out.append(AblationConfig(name=f"reward={v}", reward_variant=v))
    for v in DECOMPOSITION_VARIANTS:
        if v != "per_z":
            out.append(AblationConfig(name=f"decomp={v}", decomposition_variant=v))
    for v in RUBRIC_VARIANTS:
        if v != "static":
            out.append(AblationConfig(name=f"rubric={v}", rubric_variant=v))
    for v in GATE_VARIANTS:
        if v != "both":
            out.append(AblationConfig(name=f"gates={v}", gate_variant=v))
    return out


@dataclass
class AblationResult:
    """One row of the results table."""

    config: AblationConfig
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "config": self.config.to_json(),
            "metrics": {k: round(float(v), 4) for k, v in self.metrics.items()},
            "notes": self.notes,
        }


__all__ = [
    "AblationConfig",
    "AblationResult",
    "baseline_set",
    "REWARD_VARIANTS",
    "DECOMPOSITION_VARIANTS",
    "RUBRIC_VARIANTS",
    "GATE_VARIANTS",
]
