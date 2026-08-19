"""Tests for forecaster package public API (Phase 6)."""

from __future__ import annotations


def test_forecaster_imports() -> None:
    """All main symbols must be importable from forecaster."""
    from forecaster import (  # noqa: F401
        ForecasterPipeline,
        HindsightConfig,
        HindsightSample,
        InferenceConfig,
        Innovation,
        JointCandidate,
        MemoryEntry,
        MemoryInventory,
        MemoryStore,
        PriorConfig,
        RealizationConfig,
        ScoredProposal,
        SFTTrainConfig,
        build_hindsight_dataset,
        extract_innovation,
        innovation_from_dict,
        innovation_to_dict,
        load_hindsight_config,
        load_inference_config,
        load_prior_config,
        load_realization_config,
        load_sft_train_config,
        memory_inventory_from_dict,
        memory_inventory_to_dict,
        run_joint_inference,
    )


def test_forecaster_pipeline_importable() -> None:
    """ForecasterPipeline must be importable from forecaster."""
    from forecaster import ForecasterPipeline

    assert ForecasterPipeline is not None


def test_forecaster_strategy_registry() -> None:
    """create_strategy('forecaster') must return a ForecasterStrategy instance."""
    from live_idea_bench.strategy.forecaster import ForecasterStrategy
    from live_idea_bench.strategy.registry import create_strategy

    strategy = create_strategy("forecaster")
    assert isinstance(strategy, ForecasterStrategy)
    assert strategy.name == "forecaster"


def test_forecaster_all_exports() -> None:
    """All symbols in __all__ must actually be present in the module."""
    import forecaster

    for name in forecaster.__all__:
        assert hasattr(forecaster, name), (
            f"Symbol {name!r} missing from forecaster module"
        )


def test_forecaster_strategy_in_strategy_init() -> None:
    """ForecasterStrategy should be importable from live_idea_bench.strategy."""
    from live_idea_bench.strategy import ForecasterStrategy  # noqa: F401

    assert ForecasterStrategy is not None
