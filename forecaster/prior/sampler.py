"""Sample innovation candidates from the trained prior model."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forecaster.config import InferenceConfig
from forecaster.models import Innovation, innovation_from_json, innovation_to_json
from forecaster.prior.prompting import (
    load_prior_prompt_config,
    render_prior_chat_transcript,
    render_prior_user_prompt,
)

logger = logging.getLogger(__name__)


def _load_prompt_config() -> dict[str, str]:
    return load_prior_prompt_config()


def _parse_innovation(text: str) -> Innovation | None:
    """Parse a JSON string into an Innovation. Returns None on failure.

    Handles models that output extra text after the JSON object by trying
    progressively shorter substrings ending at each '}'.
    """
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    # Try each closing brace from left to right (first complete object)
    search_from = start
    while True:
        end = text.find("}", search_from)
        if end == -1:
            break
        candidate = text[start:end + 1]
        try:
            return innovation_from_json(candidate)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            search_from = end + 1
            continue
    logger.warning("Failed to parse innovation from text: %s", text[:80])
    return None


def _build_prompt(system_prompt: str, input_template: str, memory_store: Any) -> str:
    memory_summary = memory_store.format_for_prompt()
    user_content = render_prior_user_prompt(memory_summary, input_template=input_template)
    return render_prior_chat_transcript(
        system_prompt=system_prompt,
        user_prompt=user_content,
        assistant_text=None,
    )


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


_PRIOR_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _load_prior_model_and_tokenizer(model_path_str: str) -> tuple[Any, Any]:
    """Load the prior checkpoint and tokenizer, including LoRA adapter checkpoints.

    Caches loaded models to avoid reloading per sliding window.
    """
    if model_path_str in _PRIOR_MODEL_CACHE:
        return _PRIOR_MODEL_CACHE[model_path_str]

    try:
        import peft
        import torch  # noqa: F401  -- availability probe only
        from transformers import AutoModelForCausalLM, AutoTokenizer
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

        import torch as _torch
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=_torch.bfloat16 if _torch.cuda.is_available() else _torch.float32,
            device_map="auto",
        )
        model = peft.PeftModel.from_pretrained(
            base_model, model_path_str,
            torch_dtype=base_model.dtype,
        )
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
    _PRIOR_MODEL_CACHE[model_path_str] = (model, tokenizer)
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
    prompt_encoded = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
    prompt_ids = prompt_encoded["input_ids"]
    prompt_len = prompt_ids.shape[1]
    normalization = str(getattr(config, "score_normalization", "per_token")).strip().lower()
    temperature = float(getattr(config, "score_temperature", 1.0) or 1.0)
    if temperature <= 0:
        temperature = 1.0

    def score(innovation: Innovation) -> float:
        target = innovation_to_json(innovation)
        full_text = f"{prompt}{target}"
        encoded = tokenizer([full_text], return_tensors="pt", add_special_tokens=False)
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


def _sglang_prior_url() -> str | None:
    """Return the SGLang server URL for prior model if set."""
    import os
    url = os.environ.get("SGLANG_PRIOR_URL", "").strip()
    return url if url else None


_SGLANG_MODEL_ID_CACHE: dict[str, str] = {}


def _sample_via_sglang(
    prompt: str,
    config: InferenceConfig,
    sglang_url: str,
) -> list[Innovation]:
    """Sample innovations via SGLang/vLLM OpenAI-compatible API."""
    import openai

    client = openai.OpenAI(base_url=f"{sglang_url}/v1", api_key="none")

    if sglang_url not in _SGLANG_MODEL_ID_CACHE:
        _SGLANG_MODEL_ID_CACHE[sglang_url] = client.models.list().data[0].id
    model_id = _SGLANG_MODEL_ID_CACHE[sglang_url]

    innovations: list[Innovation] = []
    for _ in range(config.num_candidates):
        try:
            resp = client.completions.create(
                model=model_id,
                prompt=prompt,
                max_tokens=512,
                temperature=config.prior_temperature,
                top_p=0.9,
            )
            raw_text = resp.choices[0].text
            inn = _parse_innovation(raw_text)
            if inn is not None:
                innovations.append(inn)
            else:
                logger.warning("SGLang: unparseable sample: %s", raw_text[:120])
        except Exception as exc:
            logger.warning("SGLang prior sampling failed: %s", exc)

    return innovations


def sample_innovations(
    model_path: str,
    memory_store: Any,
    config: InferenceConfig,
) -> list[Innovation]:
    """Sample C candidate innovations from the trained prior model.

    Requires torch, transformers, peft. Raises ImportError if not available.
    Generates config.num_candidates innovations at config.prior_temperature.
    Returns list of Innovation objects (failed parses are skipped with warning).

    If SGLANG_PRIOR_URL is set, uses the SGLang API for fast inference.
    Otherwise loads the model locally via HF generate.
    """
    prompt_cfg = _load_prompt_config()
    prompt = _build_prompt(
        prompt_cfg["system_prompt"],
        prompt_cfg["input_template"],
        memory_store,
    )

    # Fast path: SGLang API
    sglang_url = _sglang_prior_url()
    if sglang_url:
        logger.info("Using SGLang API at %s for prior sampling", sglang_url)
        return _sample_via_sglang(prompt, config, sglang_url)

    # Slow path: local HF model
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Sampling from the prior requires torch. Install with: pip install torch"
        ) from exc

    model, tokenizer = _load_prior_model_and_tokenizer(str(model_path))

    encoded = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": 512,
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
