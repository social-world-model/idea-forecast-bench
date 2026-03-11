from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import dataclasses

from live_idea_bench.daily import coerce_prediction
from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.predictor import generate_predictions
from live_idea_bench.rl.config import SelectionConfig, load_selection_config
from live_idea_bench.rl.selection import select_top_k_predictions
from live_idea_bench.strategy.base import IdeaStrategy


class PolicyRLStrategy(IdeaStrategy):
    name = "policy_rl"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        predictor_config: str = "predictor.yaml",
        selection_config: str = "selection.yaml",
        similarity_config: str = "similarity.yaml",
        temperature: float | None = None,
        policy_manifest_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.predictor_config = predictor_config
        self.selection_config = selection_config
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
        return [dataclasses.replace(p, rank=idx) for idx, p in enumerate(predictions, start=1)]

    def _resolve_selection_config(self, manifest: dict[str, Any]) -> SelectionConfig:
        path = str(manifest.get("selection_config_path") or "").strip() or self.selection_config
        return load_selection_config(path)

    def _generate_from_local_checkpoint(
        self,
        train_papers: List[PaperRecord],
        cutoff_month: str,
        candidate_pool_size: int,
        *,
        checkpoint_path: str,
        predictor_config_path: str,
        temperature: float | None,
        base_model_name: str | None,
    ) -> list[IdeaPrediction]:
        from live_idea_bench.rl.local_generation import generate_local_predictions

        candidates: list[IdeaPrediction] = []
        for idx in range(candidate_pool_size):
            predictions = generate_local_predictions(
                train_papers=train_papers,
                cutoff_month=cutoff_month,
                top_k=1,
                model_name_or_path=checkpoint_path,
                predictor_config_path=predictor_config_path,
                temperature=temperature,
                max_new_tokens=768,
                top_p=0.9,
                sampling_top_k=40,
                repetition_penalty=1.05,
                seed=idx,
                base_model_name=base_model_name,
            )
            for prediction in predictions[:1]:
                candidates.append(dataclasses.replace(prediction, metadata={
                    "checkpoint_path": checkpoint_path,
                    "candidate_sample_index": idx,
                    **prediction.metadata,
                }))
        return candidates

    def _generate_candidate_pool_from_model(
        self,
        train_papers: List[PaperRecord],
        cutoff_month: str,
        *,
        model_name: str | None,
        predictor_config_path: str,
        temperature: float | None,
        candidate_pool_size: int,
    ) -> list[IdeaPrediction]:
        candidates: list[IdeaPrediction] = []
        for idx in range(candidate_pool_size):
            predictions = generate_predictions(
                train_papers=train_papers,
                cutoff_month=cutoff_month,
                top_k=1,
                model_name=model_name,
                predictor_config_path=predictor_config_path,
                temperature=temperature,
            )
            for prediction in predictions[:1]:
                candidates.append(dataclasses.replace(prediction, metadata={
                    "candidate_sample_index": idx,
                    **prediction.metadata,
                }))
        return candidates

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
        resolved_selection_config = self._resolve_selection_config(manifest)
        resolved_temperature = (
            self.temperature
            if self.temperature is not None
            else (
                float(manifest.get("temperature"))
                if manifest.get("temperature") is not None
                else 0.8
            )
        )
        checkpoint_path = str(manifest.get("checkpoint_path") or "").strip()
        if checkpoint_path and Path(checkpoint_path).expanduser().exists():
            candidates = self._generate_from_local_checkpoint(
                train_papers=train_papers,
                cutoff_month=cutoff_month,
                candidate_pool_size=resolved_selection_config.candidate_pool_size,
                checkpoint_path=str(Path(checkpoint_path).expanduser()),
                predictor_config_path=resolved_predictor_config,
                temperature=resolved_temperature,
                base_model_name=(
                    str(manifest.get("inference_model_name") or "").strip() or None
                ),
            )
            if candidates:
                selected = select_top_k_predictions(
                    candidates,
                    train_papers,
                    resolved_selection_config,
                    top_k=top_k,
                )
                if selected:
                    return selected

        resolved_model = (
            self.model_name
            or str(manifest.get("inference_model_name") or "").strip()
            or None
        )
        candidates = self._generate_candidate_pool_from_model(
            train_papers=train_papers,
            cutoff_month=cutoff_month,
            model_name=resolved_model,
            predictor_config_path=resolved_predictor_config,
            temperature=resolved_temperature,
            candidate_pool_size=resolved_selection_config.candidate_pool_size,
        )
        predictions = select_top_k_predictions(
            candidates,
            train_papers,
            resolved_selection_config,
            top_k=top_k,
        )
        policy_stage = str(manifest.get("trainer") or manifest.get("stage") or "policy_rl")
        return [
            dataclasses.replace(p, rank=idx, metadata={
                "policy_stage": policy_stage,
                **({"policy_manifest_path": self.policy_manifest_path} if self.policy_manifest_path else {}),
                **p.metadata,
            })
            for idx, p in enumerate(predictions[:top_k], start=1)
        ]
