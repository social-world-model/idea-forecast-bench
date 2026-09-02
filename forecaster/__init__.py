"""Forecaster package: implements the factorized latent variable forecasting method.

Main entry points:
- ForecasterPipeline: full 4-phase pipeline
- run_joint_inference: joint inference (Algorithm 1)
- MemoryStore: memory inventory for the innovation prior
"""
# Dependency direction: forecaster -> idea_forecast_bench, one way only.
# idea_forecast_bench never imports forecaster at module scope; the MDF strategy
# adapters under idea_forecast_bench/strategy/ import it inside functions.
#
# The symbols below that pull in idea_forecast_bench.backtest (hindsight,
# orchestrator, inference) stay behind __getattr__ so that `import forecaster`
# does not drag in the whole benchmark runner. This used to be load-bearing --
# it worked around a real cycle through idea_forecast_bench.strategy.policy_rl --
# but that cycle is gone; the laziness is now just about import cost.

from forecaster.config import (
    HindsightConfig,
    InferenceConfig,
    PriorConfig,
    RealizationConfig,
    SFTTrainConfig,
    load_hindsight_config,
    load_inference_config,
    load_prior_config,
    load_realization_config,
    load_sft_train_config,
)
from forecaster.models import (
    HindsightSample,
    Innovation,
    JointCandidate,
    MemoryEntry,
    MemoryInventory,
    ScoredProposal,
    innovation_from_dict,
    innovation_to_dict,
    memory_inventory_from_dict,
    memory_inventory_to_dict,
)
from forecaster.prior.memory import MemoryStore

__all__ = [
    "Innovation",
    "MemoryEntry",
    "MemoryInventory",
    "HindsightSample",
    "JointCandidate",
    "ScoredProposal",
    "innovation_to_dict",
    "innovation_from_dict",
    "memory_inventory_to_dict",
    "memory_inventory_from_dict",
    "HindsightConfig",
    "PriorConfig",
    "SFTTrainConfig",
    "RealizationConfig",
    "InferenceConfig",
    "load_hindsight_config",
    "load_prior_config",
    "load_sft_train_config",
    "load_realization_config",
    "load_inference_config",
    "MemoryStore",
    # Lazily-loaded (idea_forecast_bench.backtest dependency):
    "extract_innovation",
    "build_hindsight_dataset",
    "run_joint_inference",
    "ForecasterPipeline",
]

# ---------------------------------------------------------------------------
# Lazy loaders for symbols that depend on idea_forecast_bench.backtest
# ---------------------------------------------------------------------------

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "extract_innovation": ("forecaster.hindsight", "extract_innovation"),
    "build_hindsight_dataset": ("forecaster.hindsight", "build_hindsight_dataset"),
    "run_joint_inference": ("forecaster.inference", "run_joint_inference"),
    "ForecasterPipeline": ("forecaster.orchestrator", "ForecasterPipeline"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_MAP:
        module_path, attr = _LAZY_MAP[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'forecaster' has no attribute {name!r}")
