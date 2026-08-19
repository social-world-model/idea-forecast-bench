"""Tests for forecaster/prior/sampler.py (Gap 2 – LoRA loading fix)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from forecaster.config import InferenceConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory_store() -> MagicMock:
    store = MagicMock()
    store.format_for_prompt.return_value = "(no entries in memory)"
    return store


def _make_inference_config(num_candidates: int = 2) -> InferenceConfig:
    return InferenceConfig(num_candidates=num_candidates, prior_temperature=1.0)


def _write_adapter_config(directory: Path, base_model_id: str) -> None:
    """Write a minimal adapter_config.json to simulate PEFT's save_pretrained output."""
    config = {"base_model_name_or_path": base_model_id, "peft_type": "LORA"}
    (directory / "adapter_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )


def _valid_innovation_json(direction: str = "test direction") -> str:
    return json.dumps(
        {"base_direction": direction, "operator": "extend", "gap": "close the gap"}
    )


# ---------------------------------------------------------------------------
# Tests for _detect_base_model
# ---------------------------------------------------------------------------

class TestDetectBaseModel:
    def test_returns_base_model_id_when_adapter_config_present(self, tmp_path: Path) -> None:
        """_detect_base_model should return the base_model_name_or_path field."""
        from forecaster.prior.sampler import _detect_base_model

        _write_adapter_config(tmp_path, "Qwen/Qwen2.5-3B-Instruct")
        result = _detect_base_model(str(tmp_path))
        assert result == "Qwen/Qwen2.5-3B-Instruct"

    def test_returns_none_when_adapter_config_missing(self, tmp_path: Path) -> None:
        """_detect_base_model should return None if no adapter_config.json."""
        from forecaster.prior.sampler import _detect_base_model

        result = _detect_base_model(str(tmp_path))
        assert result is None

    def test_returns_none_when_adapter_config_malformed(self, tmp_path: Path) -> None:
        """_detect_base_model should return None on JSON parse error."""
        from forecaster.prior.sampler import _detect_base_model

        (tmp_path / "adapter_config.json").write_text("not-json", encoding="utf-8")
        result = _detect_base_model(str(tmp_path))
        assert result is None

    def test_returns_none_when_field_missing(self, tmp_path: Path) -> None:
        """_detect_base_model should return None if field absent in config."""
        from forecaster.prior.sampler import _detect_base_model

        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA"}), encoding="utf-8"
        )
        result = _detect_base_model(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# Tests for sample_innovations with LoRA adapter checkpoint
# ---------------------------------------------------------------------------

class TestSampleInnovationsLoRAPath:
    """sample_innovations should load via PeftModel when adapter_config.json is present.

    Patches sys.modules for torch/transformers/peft (none are installed in test env)
    and uses correctly-configured MagicMocks for the tokenizer and model objects.
    """

    def _make_fake_seq(self) -> MagicMock:
        """A mock tensor sequence supporting seq[n:].tolist()."""
        fake_seq = MagicMock()
        fake_seq.__getitem__ = lambda self, sl: MagicMock(
            **{"tolist.return_value": [1, 2, 3]}
        )
        return fake_seq

    def _make_fake_tokenizer(self, innovation_json: str) -> MagicMock:
        """Tokenizer mock that returns a proper encoded dict."""
        fake_input_ids = MagicMock()
        fake_input_ids.shape = (1, 5)

        fake_tokenizer = MagicMock()
        fake_tokenizer.pad_token = "eos"
        fake_tokenizer.pad_token_id = 0
        fake_tokenizer.return_value = {"input_ids": fake_input_ids}
        fake_tokenizer.decode.return_value = innovation_json
        return fake_tokenizer

    def _make_fake_model(self, n_seqs: int = 1) -> MagicMock:
        """Model mock that generates n_seqs fake token sequences."""
        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_model.generate.return_value = [self._make_fake_seq() for _ in range(n_seqs)]
        return fake_model

    def _make_ml_mocks(
        self, fake_tokenizer: MagicMock, fake_base_model: MagicMock, fake_peft_model: MagicMock
    ) -> dict:
        """Build sys.modules-ready mocks for torch, transformers, peft."""
        fake_peft_cls = MagicMock()
        fake_peft_cls.from_pretrained.return_value = fake_peft_model
        fake_peft_module = MagicMock()
        fake_peft_module.PeftModel = fake_peft_cls

        fake_auto_tokenizer_cls = MagicMock()
        fake_auto_tokenizer_cls.from_pretrained.return_value = fake_tokenizer
        fake_auto_model_cls = MagicMock()
        fake_auto_model_cls.from_pretrained.return_value = fake_base_model

        fake_transformers = MagicMock()
        fake_transformers.AutoTokenizer = fake_auto_tokenizer_cls
        fake_transformers.AutoModelForCausalLM = fake_auto_model_cls

        fake_torch = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = lambda s: s
        ctx.__exit__ = MagicMock(return_value=False)
        fake_torch.no_grad.return_value = ctx

        return {
            "peft_cls": fake_peft_cls,
            "peft_module": fake_peft_module,
            "auto_tokenizer_cls": fake_auto_tokenizer_cls,
            "auto_model_cls": fake_auto_model_cls,
            "transformers": fake_transformers,
            "torch": fake_torch,
        }

    def test_peft_model_loaded_when_adapter_config_present(self, tmp_path: Path) -> None:
        """PeftModel.from_pretrained should be called with base model and adapter path."""
        _write_adapter_config(tmp_path, "Qwen/Qwen2.5-3B-Instruct")

        innovation_json = _valid_innovation_json("peft direction")
        fake_base_model = self._make_fake_model(n_seqs=0)  # not directly used for generation
        fake_peft_model = self._make_fake_model(n_seqs=2)
        fake_tokenizer = self._make_fake_tokenizer(innovation_json)
        ml = self._make_ml_mocks(fake_tokenizer, fake_base_model, fake_peft_model)
        # The peft model is returned by PeftModel.from_pretrained — give it the generate output
        # (already set above via _make_fake_model)

        from forecaster.prior import sampler as sampler_mod

        with patch.dict(
            sys.modules,
            {"torch": ml["torch"], "transformers": ml["transformers"], "peft": ml["peft_module"]},
        ), patch.object(
            sampler_mod,
            "_load_prompt_config",
            return_value={"system_prompt": "sys", "input_template": "tmpl {memory_summary}"},
        ):
            sampler_mod.sample_innovations(
                str(tmp_path), _make_memory_store(), _make_inference_config(num_candidates=2)
            )

        ml["peft_cls"].from_pretrained.assert_called_once_with(
            ml["auto_model_cls"].from_pretrained.return_value,
            str(tmp_path),
            torch_dtype=ml["auto_model_cls"].from_pretrained.return_value.dtype,
        )

    def test_auto_model_loaded_directly_when_no_adapter_config(self, tmp_path: Path) -> None:
        """AutoModelForCausalLM.from_pretrained should be called directly when no adapter."""
        # No adapter_config.json — direct load path

        innovation_json = _valid_innovation_json("direct direction")
        fake_base_model = self._make_fake_model(n_seqs=1)
        fake_peft_model = self._make_fake_model(n_seqs=0)
        fake_tokenizer = self._make_fake_tokenizer(innovation_json)
        ml = self._make_ml_mocks(fake_tokenizer, fake_base_model, fake_peft_model)

        from forecaster.prior import sampler as sampler_mod

        with patch.dict(
            sys.modules,
            {"torch": ml["torch"], "transformers": ml["transformers"], "peft": ml["peft_module"]},
        ), patch.object(
            sampler_mod,
            "_load_prompt_config",
            return_value={"system_prompt": "sys", "input_template": "tmpl {memory_summary}"},
        ):
            sampler_mod.sample_innovations(
                str(tmp_path), _make_memory_store(), _make_inference_config(num_candidates=1)
            )

        # PeftModel.from_pretrained must NOT have been called
        ml["peft_cls"].from_pretrained.assert_not_called()
        # AutoModelForCausalLM.from_pretrained SHOULD have been called directly
        ml["auto_model_cls"].from_pretrained.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for trainer.py – prior_metadata.json is written
# ---------------------------------------------------------------------------

class TestTrainPriorMetadata:
    def test_prior_metadata_written_after_save(self, tmp_path: Path) -> None:
        """train_prior should write prior_metadata.json into the checkpoint directory."""
        from forecaster.config import SFTTrainConfig

        sft_config = SFTTrainConfig(model_alias="qwen2.5-3b-instruct")

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False

        fake_tokenizer = MagicMock()
        fake_tokenizer.return_value = {
            "input_ids": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
        }
        fake_model = MagicMock()
        fake_peft_model = MagicMock()
        fake_lora_config_cls = MagicMock()
        fake_task_type = MagicMock()
        fake_task_type.CAUSAL_LM = "CAUSAL_LM"
        fake_trainer = MagicMock()
        MagicMock()

        fake_peft_module = MagicMock()
        fake_peft_module.get_peft_model.return_value = fake_peft_model
        fake_peft_module.LoraConfig = fake_lora_config_cls
        fake_peft_module.TaskType = fake_task_type

        fake_transformers = MagicMock()
        fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tokenizer
        fake_transformers.AutoModelForCausalLM.from_pretrained.return_value = fake_model
        fake_transformers.TrainingArguments = MagicMock()
        fake_transformers.Trainer.return_value = fake_trainer

        fake_datasets_module = MagicMock()
        fake_ds = MagicMock()
        fake_ds.map.return_value = fake_ds
        fake_datasets_module.Dataset.from_dict.return_value = fake_ds

        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "transformers": fake_transformers,
                "peft": fake_peft_module,
                "datasets": fake_datasets_module,
            },
        ), patch(
            "forecaster.prior.trainer._load_system_prompt",
            return_value="system prompt",
        ), patch(
            "forecaster.prior.trainer._build_hf_dataset",
            return_value=fake_ds,
        ):
            import importlib

            from forecaster.prior import trainer as trainer_mod
            importlib.reload(trainer_mod)

            result = trainer_mod.train_prior(
                sft_samples=[{"input": "x", "target": "y"}],
                config=sft_config,
                output_dir=tmp_path,
            )

        checkpoint_path = Path(result)
        metadata_path = checkpoint_path / "prior_metadata.json"
        assert metadata_path.exists(), "prior_metadata.json was not written"

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["checkpoint_type"] == "lora_adapter"
        assert "base_model_id" in metadata
        assert "model_alias" in metadata
        assert metadata["model_alias"] == "qwen2.5-3b-instruct"
