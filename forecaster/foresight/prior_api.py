"""sample_z(memory_text, n, temperature) — diverse z sampling for inference.

A thin wrapper over the existing forecaster.prior.sampler that takes a
precomputed memory string (e.g., from build_memory()) and returns a list
of Innovation samples. Use this as the prior interface from Phase 7's
forecast() composition.

This module deliberately does no model loading itself; it composes with
whatever loader the existing sampler exposes.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from forecaster.config import InferenceConfig
from forecaster.foresight.prior_io import RawMemoryStore
from forecaster.models import Innovation

logger = logging.getLogger(__name__)


@dataclass
class SampleZRequest:
    memory_text: str
    n: int
    temperature: float


SamplerFn = Callable[[RawMemoryStore, int, float], list[Innovation]]


def sample_z(
    memory_text: str,
    n: int,
    temperature: float = 0.9,
    *,
    sampler: SamplerFn | None = None,
    inference_config: InferenceConfig | None = None,
    prior_model_path: str | None = None,
) -> list[Innovation]:
    """Sample n innovations conditioned on a memory string.

    Args:
        memory_text: A precomputed memory snapshot (e.g. `build_memory()` output).
        n: Number of samples to draw.
        temperature: Decoding temperature (>0).
        sampler: Optional injected sampler (used in tests). When None,
            falls back to forecaster.prior.sampler.sample_innovations.
        inference_config: Optional InferenceConfig override (forwarded to
            the underlying sampler when used in fallback mode).
        prior_model_path: Path or model alias for the SFT-trained prior.

    Returns:
        List of Innovation objects, possibly with duplicates. Callers
        should dedupe + truncate to the desired top-K after scoring.
    """
    if n <= 0:
        return []
    store = RawMemoryStore(memory_text=memory_text)

    if sampler is not None:
        return sampler(store, n, temperature)

    # Fallback: use the live sampler. Imported lazily to keep test imports cheap.
    try:
        from forecaster.prior.sampler import sample_innovations
    except ImportError as exc:  # pragma: no cover - only when prior deps missing
        raise RuntimeError(
            "sample_z fallback requires forecaster.prior.sampler.sample_innovations"
        ) from exc

    if inference_config is None:
        # Minimal default that keeps the existing sampler happy.
        from forecaster.config import InferenceConfig as _IC

        inference_config = _IC()
    return sample_innovations(
        memory_store=store,
        n=n,
        temperature=temperature,
        config=inference_config,
        prior_model_path=prior_model_path,
    )


def operator_distribution(innovations: list[Innovation]) -> dict[str, int]:
    """Helper for the acceptance check: count operators in a sampled set."""
    out: dict[str, int] = {}
    for z in innovations:
        out[z.operator] = out.get(z.operator, 0) + 1
    return out


__all__ = ["sample_z", "operator_distribution", "SamplerFn", "SampleZRequest"]
