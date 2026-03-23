"""Tests for forecaster config loaders."""
from __future__ import annotations

import pytest

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


# ---------------------------------------------------------------------------
# Default value tests
# ---------------------------------------------------------------------------


class TestHindsightConfigDefaults:
    def test_defaults(self) -> None:
        cfg = HindsightConfig()
        assert cfg.llm_model == "gpt-4o"
        assert cfg.temperature == 0.2
        assert cfg.max_context_papers == 15
        assert cfg.max_retries == 2

    def test_mutable(self) -> None:
        cfg = HindsightConfig()
        cfg.temperature = 0.5
        assert cfg.temperature == 0.5


class TestPriorConfigDefaults:
    def test_defaults(self) -> None:
        cfg = PriorConfig()
        assert cfg.memory_path == "data/memory_inventory.json"
        assert cfg.recency_decay == 0.9
        assert cfg.frequency_cap == 10
        assert cfg.utility_ema_alpha == 0.3

    def test_mutable(self) -> None:
        cfg = PriorConfig()
        cfg.recency_decay = 0.8
        assert cfg.recency_decay == 0.8


class TestSFTTrainConfigDefaults:
    def test_defaults(self) -> None:
        cfg = SFTTrainConfig()
        assert cfg.model_alias == "qwen2.5-3b-instruct"
        assert cfg.num_epochs == 3
        assert cfg.learning_rate == pytest.approx(2e-5)
        assert cfg.per_device_batch_size == 4
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == pytest.approx(0.05)
        assert cfg.max_seq_length == 2048
        assert cfg.output_dir == "output/prior_sft"


class TestRealizationConfigDefaults:
    def test_defaults(self) -> None:
        cfg = RealizationConfig()
        assert cfg.evidence_top_k == 5
        assert cfg.evidence_similarity_threshold == pytest.approx(0.3)
        assert cfg.proposal_max_tokens == 1024
        assert cfg.evidence_accuracy_weight == pytest.approx(0.2)
        assert cfg.operator_adherence_weight == pytest.approx(0.3)
        assert cfg.coherence_weight == pytest.approx(0.5)

    def test_weights_sum_to_one(self) -> None:
        cfg = RealizationConfig()
        total = cfg.evidence_accuracy_weight + cfg.operator_adherence_weight + cfg.coherence_weight
        assert total == pytest.approx(1.0)


class TestInferenceConfigDefaults:
    def test_defaults(self) -> None:
        cfg = InferenceConfig()
        assert cfg.num_candidates == 16
        assert cfg.prior_weight == pytest.approx(0.4)
        assert cfg.realization_weight == pytest.approx(0.6)
        assert cfg.top_k == 5
        assert cfg.dedup_threshold == pytest.approx(0.8)
        assert cfg.prior_temperature == pytest.approx(0.8)

    def test_weights_sum_to_one(self) -> None:
        cfg = InferenceConfig()
        total = cfg.prior_weight + cfg.realization_weight
        assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# YAML loader tests
# ---------------------------------------------------------------------------


class TestLoadHindsightConfig:
    def test_load_from_yaml(self) -> None:
        cfg = load_hindsight_config("hindsight.yaml")
        assert isinstance(cfg, HindsightConfig)
        assert cfg.llm_model == "gpt-4o"
        assert cfg.temperature == pytest.approx(0.2)
        assert cfg.max_context_papers == 15
        assert cfg.max_retries == 2

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_hindsight_config("nonexistent_file_xyz.yaml")


class TestLoadPriorConfig:
    def test_load_from_yaml(self) -> None:
        cfg = load_prior_config("prior.yaml")
        assert isinstance(cfg, PriorConfig)
        assert cfg.recency_decay == pytest.approx(0.9)
        assert cfg.frequency_cap == 10


class TestLoadSFTTrainConfig:
    def test_load_from_yaml(self) -> None:
        cfg = load_sft_train_config("prior.yaml")
        assert isinstance(cfg, SFTTrainConfig)
        assert cfg.model_alias == "qwen2.5-3b-instruct"
        assert cfg.num_epochs == 3
        assert cfg.lora_r == 16


class TestLoadRealizationConfig:
    def test_load_from_yaml(self) -> None:
        cfg = load_realization_config("realization.yaml")
        assert isinstance(cfg, RealizationConfig)
        assert cfg.evidence_top_k == 5
        assert cfg.proposal_max_tokens == 1024

    def test_weights_sum_to_one_from_yaml(self) -> None:
        cfg = load_realization_config("realization.yaml")
        total = cfg.evidence_accuracy_weight + cfg.operator_adherence_weight + cfg.coherence_weight
        assert total == pytest.approx(1.0)


class TestLoadInferenceConfig:
    def test_load_from_yaml(self) -> None:
        cfg = load_inference_config("inference.yaml")
        assert isinstance(cfg, InferenceConfig)
        assert cfg.num_candidates == 16
        assert cfg.top_k == 5

    def test_weights_sum_to_one_from_yaml(self) -> None:
        cfg = load_inference_config("inference.yaml")
        total = cfg.prior_weight + cfg.realization_weight
        assert total == pytest.approx(1.0)
