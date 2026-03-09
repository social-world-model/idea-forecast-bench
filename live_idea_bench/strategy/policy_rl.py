from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from live_idea_bench.config import load_predictor_config
from live_idea_bench.daily import coerce_prediction
from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.predictor import (
    _build_abstract_block,
    _coerce_key_terms,
    _coerce_score,
    _extract_json_payload,
    _infer_domain,
    _parse_prediction_items,
    generate_predictions,
)
from live_idea_bench.strategy.base import IdeaStrategy

_LOCAL_POLICY_CACHE: dict[str, tuple[Any, Any]] = {}


class PolicyRLStrategy(IdeaStrategy):
    name = "policy_rl"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        predictor_config: str = "predictor.yaml",
        similarity_config: str = "similarity.yaml",
        temperature: float | None = None,
        policy_manifest_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.predictor_config = predictor_config
        self.similarity_config = similarity_config
        self.temperature = temperature
        self.policy_manifest_path = policy_manifest_path

    def _load_manifest(self) -> dict[str, Any]:
        if not self.policy_manifest_path:
            return {}
        path = Path(self.policy_manifest_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"RL policy manifest does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"RL policy manifest at {path} must decode to a mapping")
        return payload

    @staticmethod
    def _from_static_predictions(
        raw_predictions: Any,
        *,
        top_k: int,
    ) -> list[IdeaPrediction]:
        if not isinstance(raw_predictions, list):
            return []
        predictions: list[IdeaPrediction] = []
        for idx, item in enumerate(raw_predictions, start=1):
            if not isinstance(item, dict):
                continue
            predictions.append(coerce_prediction(item, idx))
            if len(predictions) >= top_k:
                break
        for idx, prediction in enumerate(predictions, start=1):
            prediction.rank = idx
        return predictions

    @staticmethod
    def _load_local_policy(checkpoint_path: str) -> tuple[Any, Any]:
        if checkpoint_path in _LOCAL_POLICY_CACHE:
            return _LOCAL_POLICY_CACHE[checkpoint_path]
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "transformers is required to load a local RL checkpoint for policy_rl."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        _LOCAL_POLICY_CACHE[checkpoint_path] = (model, tokenizer)
        return model, tokenizer

    def _generate_from_local_checkpoint(
        self,
        train_papers: List[PaperRecord],
        cutoff_month: str,
        top_k: int,
        *,
        checkpoint_path: str,
        predictor_config_path: str,
        temperature: float | None,
    ) -> list[IdeaPrediction]:
        model, tokenizer = self._load_local_policy(checkpoint_path)
        predictor_config = load_predictor_config(predictor_config_path)
        prompt = predictor_config.user_template.format(
            domain=_infer_domain(train_papers),
            horizon=f"the months after {cutoff_month}",
            n_ideas=top_k,
            abstracts=_build_abstract_block(train_papers, predictor_config.max_context_papers),
            cutoff_month=cutoff_month,
        )
        full_prompt = f"{predictor_config.system_prompt}\n\n{prompt}".strip()
        encoded = tokenizer(full_prompt, return_tensors="pt")
        generation_kwargs = {
            "max_new_tokens": 768,
            "do_sample": temperature is not None and temperature > 0,
            "temperature": temperature if temperature is not None else 1.0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        generated = model.generate(**encoded, **generation_kwargs)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        raw_text = decoded[len(full_prompt) :].strip() if decoded.startswith(full_prompt) else decoded.strip()
        payload = _extract_json_payload(raw_text)
        items = _parse_prediction_items(payload)

        predictions: list[IdeaPrediction] = []
        for item in items:
            title = str(item.get("title") or item.get("Title") or "").strip()
            if not title:
                continue
            predictions.append(
                IdeaPrediction(
                    rank=len(predictions) + 1,
                    title=title,
                    rationale=str(item.get("rationale") or item.get("Rationale") or "").strip(),
                    approach=str(item.get("approach") or item.get("Approach") or "").strip(),
                    score=_coerce_score(item.get("score", item.get("Score", 0.5))),
                    confidence=_coerce_score(
                        item.get("confidence", item.get("Confidence", item.get("score", item.get("Score", 0.5))))
                    ),
                    key_terms=_coerce_key_terms(item.get("key_terms") or item.get("keywords")),
                    metadata={"checkpoint_path": checkpoint_path},
                )
            )
            if len(predictions) >= top_k:
                break
        return predictions

    def generate(
        self,
        train_papers: List[PaperRecord],
        cutoff_month: str,
        top_k: int,
    ) -> List[IdeaPrediction]:
        manifest = self._load_manifest() if self.policy_manifest_path else {}
        static_predictions = manifest.get("static_predictions") or {}
        if isinstance(static_predictions, dict):
            from_manifest = self._from_static_predictions(static_predictions.get(cutoff_month), top_k=top_k)
            if from_manifest:
                return from_manifest

        if isinstance(manifest.get("predictions"), list):
            from_manifest = self._from_static_predictions(manifest.get("predictions"), top_k=top_k)
            if from_manifest:
                return from_manifest

        resolved_predictor_config = (
            str(manifest.get("predictor_config") or "").strip()
            or self.predictor_config
        )
        resolved_temperature = (
            self.temperature
            if self.temperature is not None
            else (
                float(manifest["temperature"])
                if manifest.get("temperature") is not None
                else None
            )
        )
        checkpoint_path = str(manifest.get("checkpoint_path") or "").strip()
        if checkpoint_path and Path(checkpoint_path).expanduser().exists():
            predictions = self._generate_from_local_checkpoint(
                train_papers=train_papers,
                cutoff_month=cutoff_month,
                top_k=top_k,
                checkpoint_path=str(Path(checkpoint_path).expanduser()),
                predictor_config_path=resolved_predictor_config,
                temperature=resolved_temperature,
            )
            if predictions:
                return predictions[:top_k]

        resolved_model = (
            self.model_name
            or str(manifest.get("inference_model_name") or "").strip()
            or None
        )
        predictions = generate_predictions(
            train_papers=train_papers,
            cutoff_month=cutoff_month,
            top_k=top_k,
            model_name=resolved_model,
            predictor_config_path=resolved_predictor_config,
            temperature=resolved_temperature,
        )
        for idx, prediction in enumerate(predictions, start=1):
            prediction.rank = idx
            prediction.metadata.setdefault("policy_stage", str(manifest.get("stage") or "policy_rl"))
            if self.policy_manifest_path:
                prediction.metadata.setdefault("policy_manifest_path", self.policy_manifest_path)
        return predictions[:top_k]
