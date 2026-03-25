"""Sample innovation candidates from the trained prior model."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import yaml

from forecaster.models import Innovation, innovation_to_dict
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


def _detect_base_model(model_path_str: str) -> str | None:
    """Read adapter_config.json written by PEFT's save_pretrained and extract base model ID."""
    adapter_config_path = Path(model_path_str) / "adapter_config.json"
    if not adapter_config_path.exists():
        return None
    try:
        adapter_cfg = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model_id = adapter_cfg.get("base_model_name_or_path")
        if base_model_id:
            return str(base_model_id)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read adapter_config.json: %s", exc)
    return None


def _load_prior_model_and_tokenizer(model_path_str: str) -> tuple[Any, Any]:
    """Load the prior checkpoint and tokenizer, including LoRA adapter checkpoints."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import peft
    except ImportError as exc:
        raise ImportError(
            "Sampling/scoring from the prior requires: torch, transformers, peft. "
            "Install with: pip install torch transformers peft"
        ) from exc

    base_model_id = _detect_base_model(model_path_str)

    if base_model_id:
        logger.info(
            "Detected LoRA adapter checkpoint. Loading base model %r then adapter from %r.",
            base_model_id,
            model_path_str,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        model = peft.PeftModel.from_pretrained(base_model, model_path_str)
    else:
        logger.info("No adapter_config.json found; loading model directly from %r.", model_path_str)
        tokenizer = AutoTokenizer.from_pretrained(model_path_str)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path_str,
            torch_dtype="auto",
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


def build_prior_scorer(
    model_path: str,
    memory_store: Any,
    config: InferenceConfig,
) -> Callable[[Innovation], float]:
    """Build a scorer for log p(z | M_t) under the trained prior.

    Returns a callable that emits a normalized conditional log-probability for
    one Innovation candidate. The default contract uses per-token normalization.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "Prior scoring requires torch. Install with: pip install torch"
        ) from exc

    prompt_cfg = _load_prompt_config()
    prompt = _build_prompt(
        prompt_cfg["system_prompt"],
        prompt_cfg["input_template"],
        memory_store,
    )

    model, tokenizer = _load_prior_model_and_tokenizer(str(model_path))
    prompt_encoded = tokenizer([prompt], return_tensors="pt")
    prompt_ids = prompt_encoded["input_ids"]
    prompt_len = prompt_ids.shape[1]
    normalization = str(getattr(config, "score_normalization", "per_token")).strip().lower()
    temperature = float(getattr(config, "score_temperature", 1.0) or 1.0)
    if temperature <= 0:
        temperature = 1.0

    def score(innovation: Innovation) -> float:
        target = json.dumps(innovation_to_dict(innovation), ensure_ascii=False)
        full_text = f"{prompt}{target}"
        encoded = tokenizer([full_text], return_tensors="pt")
        encoded = {name: value.to(model.device) for name, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        logits = outputs.logits[:, :-1, :]
        if temperature != 1.0:
            logits = logits / temperature
        target_ids = encoded["input_ids"][:, 1:]
        target_log_probs = F.log_softmax(logits, dim=-1).gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        target_start = max(0, prompt_len - 1)
        conditioned_log_probs = target_log_probs[:, target_start:]
        if conditioned_log_probs.numel() == 0:
            return float("-inf")
        if normalization == "sum":
            return float(conditioned_log_probs.sum().item())
        return float(conditioned_log_probs.mean().item())

    return score


def sample_innovations(
    model_path: str,
    memory_store: Any,
    config: InferenceConfig,
) -> list[Innovation]:
    """Sample C candidate innovations from the trained prior model.

    Requires torch, transformers, peft. Raises ImportError if not available.
    Generates config.num_candidates innovations at config.prior_temperature.
    Returns list of Innovation objects (failed parses are skipped with warning).

    If adapter_config.json is present in model_path, loads via PeftModel.from_pretrained
    to handle LoRA adapter-only checkpoints saved by train_prior().
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Sampling from the prior requires torch. Install with: pip install torch"
        ) from exc

    prompt_cfg = _load_prompt_config()
    prompt = _build_prompt(
        prompt_cfg["system_prompt"],
        prompt_cfg["input_template"],
        memory_store,
    )

    model, tokenizer = _load_prior_model_and_tokenizer(str(model_path))

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
