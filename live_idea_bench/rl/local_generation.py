from __future__ import annotations

import dataclasses
import importlib
import logging
import threading
from typing import Any

from live_idea_bench.config import load_predictor_config
from live_idea_bench.daily import coerce_prediction
from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.predictor import (
    _build_abstract_block,
    _heuristic_predictions,
    _infer_domain,
    _extract_json_payload,
    _parse_prediction_items,
)

_LOCAL_MODEL_CACHE: dict[tuple[str, str | None], tuple[Any, Any]] = {}
_LOCAL_MODEL_CACHE_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def build_prediction_prompt(
    train_papers: list[PaperRecord],
    cutoff_month: str,
    n_ideas: int,
    *,
    predictor_config_path: str = "predictor.yaml",
) -> str:
    predictor_config = load_predictor_config(predictor_config_path)
    user_prompt = predictor_config.user_template.format(
        domain=_infer_domain(train_papers),
        horizon=f"the months after {cutoff_month}",
        n_ideas=n_ideas,
        abstracts=_build_abstract_block(train_papers, predictor_config.max_context_papers),
        cutoff_month=cutoff_month,
    )
    return f"{predictor_config.system_prompt}\n\n{user_prompt}".strip()


def _completion_to_text(raw_completion: Any) -> str:
    if isinstance(raw_completion, str):
        return raw_completion
    if isinstance(raw_completion, dict):
        content = raw_completion.get("content")
        return _completion_to_text(content)
    if isinstance(raw_completion, list):
        chunks: list[str] = []
        for item in raw_completion:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    chunks.append(_completion_to_text(content))
                else:
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunk for chunk in chunks if chunk.strip())
    return str(raw_completion or "")


def parse_completion_predictions(raw_completion: Any, *, limit: int) -> list[IdeaPrediction]:
    raw_text = _completion_to_text(raw_completion).strip()
    if not raw_text:
        return []

    try:
        payload = _extract_json_payload(raw_text)
        items = _parse_prediction_items(payload)
    except Exception as exc:
        logger.warning("Failed to parse completion predictions: %s", exc)
        return []

    predictions: list[IdeaPrediction] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        prediction = coerce_prediction(item, idx)
        if not prediction.title.strip():
            continue
        predictions.append(dataclasses.replace(prediction, rank=len(predictions) + 1))
        if len(predictions) >= limit:
            break
    return predictions


def _require_local_generation_stack() -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Local HuggingFace generation requires torch and transformers to be installed."
        ) from exc

    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _load_local_model(model_name_or_path: str, *, base_model_name: str | None = None) -> tuple[Any, Any]:
    cache_key = (model_name_or_path, base_model_name)
    if cache_key in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[cache_key]

    with _LOCAL_MODEL_CACHE_LOCK:
        if cache_key in _LOCAL_MODEL_CACHE:
            return _LOCAL_MODEL_CACHE[cache_key]

        deps = _require_local_generation_stack()
        tokenizer_source = base_model_name or model_name_or_path
        tokenizer = deps["AutoTokenizer"].from_pretrained(tokenizer_source)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if base_model_name:
            try:
                peft = importlib.import_module("peft")
            except ImportError as exc:
                raise RuntimeError(
                    "Loading LoRA adapters for local RL inference requires peft to be installed."
                ) from exc
            peft_model = getattr(peft, "PeftModel")
            model = deps["AutoModelForCausalLM"].from_pretrained(
                base_model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            model = peft_model.from_pretrained(model, model_name_or_path)
        else:
            model = deps["AutoModelForCausalLM"].from_pretrained(
                model_name_or_path,
                torch_dtype="auto",
                device_map="auto",
            )

        _LOCAL_MODEL_CACHE[cache_key] = (model, tokenizer)
        return model, tokenizer


def _apply_chat_template(tokenizer: Any, full_prompt: str, enable_thinking: bool | None) -> str:
    messages = [{"role": "user", "content": full_prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return str(tokenizer.apply_chat_template(messages, **kwargs))
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return str(tokenizer.apply_chat_template(messages, **kwargs))
    return full_prompt


def generate_local_predictions(
    train_papers: list[PaperRecord],
    cutoff_month: str,
    top_k: int,
    *,
    model_name_or_path: str,
    predictor_config_path: str = "predictor.yaml",
    temperature: float | None = None,
    top_p: float = 0.9,
    sampling_top_k: int = 40,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.05,
    enable_thinking: bool | None = None,
    seed: int | None = None,
    base_model_name: str | None = None,
) -> list[IdeaPrediction]:
    prompt = build_prediction_prompt(
        train_papers,
        cutoff_month,
        top_k,
        predictor_config_path=predictor_config_path,
    )
    model, tokenizer = _load_local_model(model_name_or_path, base_model_name=base_model_name)
    deps = _require_local_generation_stack()
    torch = deps["torch"]

    if seed is not None:
        torch.manual_seed(seed)

    chat_prompt = _apply_chat_template(tokenizer, prompt, enable_thinking)
    encoded = tokenizer([chat_prompt], return_tensors="pt")
    encoded = {name: value.to(model.device) for name, value in encoded.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": repetition_penalty,
    }
    if temperature is not None and temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": sampling_top_k,
            }
        )

    generated = model.generate(**encoded, **generation_kwargs)
    output_ids = generated[0][len(encoded["input_ids"][0]) :].tolist()
    raw_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    predictions = parse_completion_predictions(raw_text, limit=top_k)
    if not predictions:
        predictions = _heuristic_predictions(train_papers, cutoff_month, top_k)

    result: list[IdeaPrediction] = []
    for idx, prediction in enumerate(predictions[:top_k], start=1):
        new_metadata = {**prediction.metadata, "local_model_name": model_name_or_path}
        if base_model_name:
            new_metadata.setdefault("base_model_name", base_model_name)
        result.append(dataclasses.replace(prediction, rank=idx, metadata=new_metadata))
    return result
