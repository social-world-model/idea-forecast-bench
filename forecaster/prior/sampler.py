"""Sample innovation candidates from the trained prior model."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from forecaster.models import Innovation
from forecaster.config import InferenceConfig

logger = logging.getLogger(__name__)

_PROMPT_YAML_PATH = Path(__file__).resolve().parents[2] / "forecaster" / "prompt" / "prior_sft.yaml"


def _load_prompt_config() -> dict[str, str]:
    raw = yaml.safe_load(_PROMPT_YAML_PATH.read_text(encoding="utf-8"))
    return {
        "system_prompt": str(raw["system_prompt"]).strip(),
        "input_template": str(raw["input_template"]).strip(),
    }


def _parse_innovation(text: str) -> Innovation | None:
    """Parse a JSON string into an Innovation. Returns None on failure."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        obj = json.loads(text[start:end])
        return Innovation(
            base_direction=str(obj["base_direction"]),
            operator=str(obj["operator"]),
            gap=str(obj["gap"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to parse innovation from text: %s — %s", text[:80], exc)
        return None


def _build_prompt(system_prompt: str, input_template: str, memory_store: Any) -> str:
    memory_summary = memory_store.format_for_prompt()
    user_content = input_template.format(memory_summary=memory_summary)
    return f"<system>\n{system_prompt}\n</system>\n<user>\n{user_content}\n</user>\n<assistant>"


def sample_innovations(
    model_path: str,
    memory_store: Any,
    config: InferenceConfig,
) -> list[Innovation]:
    """Sample C candidate innovations from the trained prior model.

    Requires torch, transformers, peft. Raises ImportError if not available.
    Generates config.num_candidates innovations at config.prior_temperature.
    Returns list of Innovation objects (failed parses are skipped with warning).
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "Sampling from the prior requires: torch, transformers. "
            "Install with: pip install torch transformers peft"
        ) from exc

    prompt_cfg = _load_prompt_config()
    prompt = _build_prompt(
        prompt_cfg["system_prompt"],
        prompt_cfg["input_template"],
        memory_store,
    )

    model_path_str = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path_str)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path_str,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    encoded = tokenizer([prompt], return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": 256,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": True,
        "temperature": config.prior_temperature,
        "top_p": 0.9,
        "num_return_sequences": config.num_candidates,
    }

    with torch.no_grad():
        generated = model.generate(**encoded, **generation_kwargs)

    innovations: list[Innovation] = []
    input_len = encoded["input_ids"].shape[1]
    for seq in generated:
        output_ids = seq[input_len:].tolist()
        raw_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        inn = _parse_innovation(raw_text)
        if inn is not None:
            innovations.append(inn)
        else:
            logger.warning("Skipping unparseable sample: %s", raw_text[:120])
    return innovations
