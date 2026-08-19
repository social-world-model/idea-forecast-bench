"""Tests for Phase-8 non-optimized metrics + ablation registry."""
from __future__ import annotations

import numpy as np
import pytest

from forecaster.foresight.ablations import (
    AblationConfig,
    AblationResult,
    baseline_set,
)
from forecaster.foresight.metrics import (
    impact_stratified_breakdown,
    mmd_rbf,
    wasserstein_1d,
)

# --------------------------------------------------------------------------- MMD


def test_mmd_zero_when_distributions_identical():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(50, 8)).astype(np.float32)
    val = mmd_rbf(P, P.copy())
    assert val < 1e-3


def test_mmd_positive_when_distributions_differ():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(50, 8)).astype(np.float32)
    Q = rng.normal(size=(50, 8)).astype(np.float32) + 3.0
    assert mmd_rbf(P, Q) > 0.05


def test_mmd_empty_inputs_return_zero():
    assert mmd_rbf(np.zeros((0, 4)), np.zeros((5, 4))) == 0.0


# --------------------------------------------------------------------------- Wasserstein-1


def test_wasserstein_zero_for_identical_samples():
    assert wasserstein_1d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_wasserstein_translation_invariant_shift_grows():
    base = [1.0, 2.0, 3.0]
    near = wasserstein_1d(base, [1.1, 2.1, 3.1])
    far = wasserstein_1d(base, [11.0, 12.0, 13.0])
    assert far > near


def test_wasserstein_empty_returns_zero():
    assert wasserstein_1d([], [1.0]) == 0.0


# --------------------------------------------------------------------------- impact stratification


def test_impact_bucket_default_split():
    rows = [
        {"citation_count": 0, "hit_at_k": 0.4},
        {"citation_count": 1, "hit_at_k": 0.3},
        {"citation_count": 10, "hit_at_k": 0.5},
        {"citation_count": 80, "hit_at_k": 0.7},
    ]
    out = impact_stratified_breakdown(rows, metric_keys=("hit_at_k",))
    assert "low" in out and "mid" in out and "high" in out
    assert out["low"]["count"] == 2
    assert out["mid"]["count"] == 1
    assert out["high"]["count"] == 1
    assert out["high"]["hit_at_k"] == pytest.approx(0.7)


def test_impact_bucket_custom_bucket_fn():
    rows = [{"x": 1}, {"x": 2}, {"x": 3}]
    out = impact_stratified_breakdown(
        rows, bucket_fn=lambda r: "even" if r["x"] % 2 == 0 else "odd",
        metric_keys=("x",),
    )
    assert out["odd"]["count"] == 2
    assert out["even"]["count"] == 1


# --------------------------------------------------------------------------- ablations registry


def test_baseline_set_includes_ours_plus_one_per_switch():
    grid = baseline_set()
    names = [c.name for c in grid]
    assert "ours" in names
    # 2 reward, 1 decomposition, 1 rubric, 3 gate variants → 7 alternatives.
    assert len(grid) == 1 + (len(["embedding_threshold", "raw_judge"])
                             + 1 + 1 + 3)
    assert any("reward=" in n for n in names)
    assert any("decomp=" in n for n in names)
    assert any("rubric=" in n for n in names)
    assert any("gates=" in n for n in names)


def test_ablation_result_round_trip():
    cfg = AblationConfig(name="ours")
    res = AblationResult(config=cfg, metrics={"hit_at_k": 0.314159})
    j = res.to_json()
    assert j["config"]["name"] == "ours"
    assert j["metrics"]["hit_at_k"] == pytest.approx(0.3142, abs=1e-4)
