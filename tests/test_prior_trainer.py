"""Tests for prior target-only training labels."""
from __future__ import annotations

from forecaster.prior.prompting import render_prior_chat_transcript
from forecaster.prior.trainer import _build_target_only_labels, _tokenize_training_sample


class _CharTokenizer:
    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        padding: bool,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        del truncation, padding, add_special_tokens
        input_ids = [ord(char) for char in text[:max_length]]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }


def test_build_target_only_labels_masks_prompt_prefix() -> None:
    labels = _build_target_only_labels([11, 12, 13, 14], 2)
    assert labels == [-100, -100, 13, 14]


def test_tokenize_training_sample_masks_all_prompt_tokens() -> None:
    tokenizer = _CharTokenizer()
    sample = {
        "input": "Current research memory:\n1. {...}",
        "target": '{"base_direction":"test","operator":"extend","gap":"gap"}',
    }
    row = _tokenize_training_sample(
        system_prompt="system prompt",
        sample=sample,
        tokenizer=tokenizer,
        max_seq_length=2048,
    )

    prompt_text = render_prior_chat_transcript(
        system_prompt="system prompt",
        user_prompt=sample["input"],
        assistant_text=None,
    )
    prompt_len = len(prompt_text)

    assert all(label == -100 for label in row["labels"][:prompt_len])
    assert row["labels"][prompt_len:] == row["input_ids"][prompt_len:]
