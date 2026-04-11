"""SFT trainer for the innovation prior using HuggingFace Trainer + LoRA."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from forecaster.config import SFTTrainConfig
from forecaster.prior.prompting import (
    load_prior_prompt_config,
    render_prior_chat_transcript,
)

logger = logging.getLogger(__name__)


def _load_system_prompt() -> str:
    return load_prior_prompt_config()["system_prompt"]


def _normalize_tokenizer_output(value: object, key: str) -> list[int]:
    from collections.abc import Mapping

    if not isinstance(value, Mapping):
        raise TypeError("Tokenizer output must be a mapping.")
    payload = value.get(key)
    if payload is None:
        if key == "attention_mask":
            input_ids = _normalize_tokenizer_output(value, "input_ids")
            return [1] * len(input_ids)
        raise KeyError(f"Tokenizer output missing required key: {key}")
    if isinstance(payload, list):
        if payload and isinstance(payload[0], list):
            return [int(token) for token in payload[0]]
        return [int(token) for token in payload]
    raise TypeError(f"Tokenizer output[{key!r}] must be a list or list[list].")


def _build_target_only_labels(
    input_ids: list[int], prompt_token_count: int
) -> list[int]:
    masked_prefix = min(len(input_ids), max(0, prompt_token_count))
    return ([-100] * masked_prefix) + input_ids[masked_prefix:]


def _tokenize_training_sample(
    *,
    system_prompt: str,
    sample: dict[str, str],
    tokenizer: object,
    max_seq_length: int,
) -> dict[str, list[int]]:
    prompt_text = render_prior_chat_transcript(
        system_prompt=system_prompt,
        user_prompt=sample["input"],
        assistant_text=None,
    )
    full_text = render_prior_chat_transcript(
        system_prompt=system_prompt,
        user_prompt=sample["input"],
        assistant_text=sample["target"],
    )
    prompt_encoded = tokenizer(  # type: ignore[operator]
        prompt_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        add_special_tokens=False,
    )
    full_encoded = tokenizer(  # type: ignore[operator]
        full_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        add_special_tokens=False,
    )
    input_ids = _normalize_tokenizer_output(full_encoded, "input_ids")
    attention_mask = _normalize_tokenizer_output(full_encoded, "attention_mask")
    prompt_ids = _normalize_tokenizer_output(prompt_encoded, "input_ids")
    labels = _build_target_only_labels(input_ids, len(prompt_ids))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _build_hf_dataset(
    sft_samples: list[dict[str, str]],
    system_prompt: str,
    tokenizer: object,
    max_seq_length: int,
) -> object:
    import datasets as ds

    rows = [
        _tokenize_training_sample(
            system_prompt=system_prompt,
            sample=sample,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        for sample in sft_samples
    ]
    return ds.Dataset.from_dict(
        {
            "input_ids": [row["input_ids"] for row in rows],
            "attention_mask": [row["attention_mask"] for row in rows],
            "labels": [row["labels"] for row in rows],
        }
    )


def train_prior(
    sft_samples: list[dict[str, str]],
    config: SFTTrainConfig,
    *,
    output_dir: str | Path | None = None,
) -> str:
    """Train the innovation prior via SFT with Unsloth + TRL (METHOD §3.2).

    The loss is the standard target-only NLL
    ``-log p_θ(̃z_{t+1} | M_t)`` — labels are ``-100`` over the memory-prompt
    prefix and the real token ids over the innovation-target suffix
    (``_build_target_only_labels``). Unsloth provides the optimized
    ``FastLanguageModel`` loader + LoRA attachment, ``trl.SFTTrainer`` runs
    the training loop on the pre-tokenized rows.

    Args:
        sft_samples: List of {"input": str, "target": str} samples produced by
            ``forecaster.prior.sft_dataset.build_sft_samples`` (memory-augmented
            per METHOD §3.2).
        config: SFTTrainConfig with model alias, LoRA params, etc.
        output_dir: Override output directory (defaults to config.output_dir).

    Returns:
        Path to the saved checkpoint directory.
    """
    try:
        # Unsloth must be imported BEFORE transformers/trl so its patches apply.
        from unsloth import FastLanguageModel
        import torch
        from trl import SFTTrainer, SFTConfig
        from transformers import AutoTokenizer, DataCollatorForSeq2Seq
    except ImportError as exc:
        raise ImportError(
            "Prior SFT training requires: unsloth, torch, trl, transformers, datasets. "
            "Install with scripts/forecaster/setup_env.sh"
        ) from exc

    from forecaster.realization.model_zoo import resolve_small_model

    model_spec = resolve_small_model(config.model_alias)

    save_dir = str(output_dir) if output_dir is not None else config.output_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s with Unsloth (max_seq_length=%d)", model_spec.model_id, config.max_seq_length)
    model, _unsloth_processor = FastLanguageModel.from_pretrained(
        model_name=model_spec.model_id,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    # Qwen3.5 is registered as a VLM (Qwen2VLForConditionalGeneration). Unsloth
    # returns the multi-modal processor whose image_processor would route plain
    # text prompts through Qwen2VL's image_processing path and crash. Force a
    # text-only AutoTokenizer here so SFTTrainer never touches the image side.
    tokenizer = AutoTokenizer.from_pretrained(model_spec.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    system_prompt = _load_system_prompt()
    # _build_hf_dataset produces input_ids/attention_mask/labels with the
    # prompt-only mask (-100 over memory prompt, real ids over innovation
    # target). This IS the METHOD §3.2 NLL term — do not modify.
    dataset = _build_hf_dataset(
        sft_samples, system_prompt, tokenizer, config.max_seq_length
    )

    training_args = SFTConfig(
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
        bf16=torch.cuda.is_available(),
        max_seq_length=config.max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        # 8-bit AdamW: ~50% optimizer-state memory savings vs fp32 state.
        optim="adamw_8bit",
        # Auto-retry with smaller batch on OOM (catches PyTorch OOMs from
        # accelerate's find_executable_batch_size wrapper).
        auto_find_batch_size=True,
        # group_by_length is intentionally OFF — Unsloth's patched SFTConfig
        # rejects it (TypeError) because it conflicts with their internal
        # sample-packing path.
        # Prefetch next batch on CPU workers while GPU trains.
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
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
