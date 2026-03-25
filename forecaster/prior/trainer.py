"""SFT trainer for the innovation prior using HuggingFace Trainer + LoRA."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from forecaster.config import SFTTrainConfig
from forecaster.prior.sft_dataset import build_sft_samples, load_sft_dataset

logger = logging.getLogger(__name__)

_PROMPT_YAML_PATH = Path(__file__).resolve().parents[2] / "forecaster" / "prompt" / "prior_sft.yaml"


def _load_system_prompt() -> str:
    raw = yaml.safe_load(_PROMPT_YAML_PATH.read_text(encoding="utf-8"))
    return str(raw["system_prompt"]).strip()


def _format_sample(system_prompt: str, sample: dict[str, str]) -> str:
    return (
        f"<system>\n{system_prompt}\n</system>\n"
        f"<user>\n{sample['input']}\n</user>\n"
        f"<assistant>\n{sample['target']}\n</assistant>"
    )


def _build_hf_dataset(sft_samples: list[dict[str, str]], system_prompt: str, tokenizer: object, max_seq_length: int) -> object:
    import datasets as ds
    texts = [_format_sample(system_prompt, s) for s in sft_samples]
    raw_ds = ds.Dataset.from_dict({"text": texts})

    def tokenize(batch: dict) -> dict:
        return tokenizer(  # type: ignore[operator]
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    tokenized = raw_ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.map(lambda b: {"labels": b["input_ids"]}, batched=True)
    return tokenized


def train_prior(
    sft_samples: list[dict[str, str]],
    config: SFTTrainConfig,
    *,
    output_dir: str | Path | None = None,
) -> str:
    """Train the innovation prior model via SFT with LoRA.

    Raises ImportError with clear message if heavy ML deps are missing.

    Args:
        sft_samples: List of {"input": str, "target": str} samples.
        config: SFTTrainConfig with model alias, LoRA params, etc.
        output_dir: Override output directory (defaults to config.output_dir).

    Returns:
        Path to the saved checkpoint directory.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        from peft import get_peft_model, LoraConfig, TaskType
        import datasets
    except ImportError as exc:
        raise ImportError(
            "SFT training requires: torch, transformers, peft, datasets. "
            "Install with: pip install torch transformers peft datasets accelerate"
        ) from exc

    from forecaster.realization.model_zoo import resolve_small_model
    model_spec = resolve_small_model(config.model_alias)

    save_dir = str(output_dir) if output_dir is not None else config.output_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer from %s", model_spec.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_spec.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model from %s", model_spec.model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_spec.model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    system_prompt = _load_system_prompt()
    dataset = _build_hf_dataset(sft_samples, system_prompt, tokenizer, config.max_seq_length)

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    checkpoint_path = Path(save_dir) / "final_checkpoint"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_path))
    tokenizer.save_pretrained(str(checkpoint_path))

    # Write metadata so sampler.py can identify the base model without
    # relying solely on adapter_config.json (which is written by PEFT).
    prior_metadata = {
        "base_model_id": model_spec.model_id,
        "model_alias": config.model_alias,
        "checkpoint_type": "lora_adapter",
    }
    metadata_path = checkpoint_path / "prior_metadata.json"
    metadata_path.write_text(
        json.dumps(prior_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved checkpoint to %s", checkpoint_path)
    return str(checkpoint_path)
