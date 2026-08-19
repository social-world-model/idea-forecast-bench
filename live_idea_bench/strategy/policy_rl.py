from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from typing import Any

from forecaster.realization.config import SelectionConfig, load_selection_config
from forecaster.realization.selection import select_top_k_predictions
from live_idea_bench.daily import coerce_prediction
from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.predictor import generate_predictions
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

    @staticmethod
    def _sampling_plan(selection_config: SelectionConfig, default_temperature: float | None) -> list[dict[str, Any]]:
        temperatures = list(selection_config.temperature_schedule) or [default_temperature or 0.8]
        top_ps = list(selection_config.top_p_schedule) or [0.9]
        combinations = [(temperature, top_p) for temperature in temperatures for top_p in top_ps]
        if not combinations:
            combinations = [(default_temperature or 0.8, 0.9)]

        plan: list[dict[str, Any]] = []
        for idx in range(selection_config.candidate_pool_size):
            temperature, top_p = combinations[idx % len(combinations)]
            plan.append(
                {
                    "candidate_sample_index": idx,
                    "sampling_temperature": float(temperature),
                    "sampling_top_p": float(top_p),
                    "context_shuffle_seed": idx if selection_config.enable_context_shuffle else None,
                }
            )
        return plan

    @staticmethod
    def _shuffle_train_papers(
        train_papers: list[PaperRecord],
        context_shuffle_seed: int | None,
    ) -> list[PaperRecord]:
        ordered = list(train_papers)
        if context_shuffle_seed is None:
            return ordered
        rng = random.Random(context_shuffle_seed)
        rng.shuffle(ordered)
        return ordered

    def _generate_from_local_checkpoint(
        self,
        train_papers: list[PaperRecord],
        cutoff_month: str,
        selection_config: SelectionConfig,
        *,
        checkpoint_path: str,
        predictor_config_path: str,
        temperature: float | None,
        base_model_name: str | None,
    ) -> list[IdeaPrediction]:
        from forecaster.realization.local_generation import generate_local_predictions

        candidates: list[IdeaPrediction] = []
        for sample in self._sampling_plan(selection_config, temperature):
            predictions = generate_local_predictions(
                train_papers=self._shuffle_train_papers(train_papers, sample["context_shuffle_seed"]),
                cutoff_month=cutoff_month,
                top_k=1,
                model_name_or_path=checkpoint_path,
                predictor_config_path=predictor_config_path,
                temperature=sample["sampling_temperature"],
                max_new_tokens=768,
                top_p=sample["sampling_top_p"],
                sampling_top_k=40,
                repetition_penalty=1.05,
                seed=sample["candidate_sample_index"],
                base_model_name=base_model_name,
                fallback_to_heuristic=False,
            )
            for prediction in predictions[:1]:
                candidates.append(dataclasses.replace(prediction, metadata={
                    "checkpoint_path": checkpoint_path,
                    **sample,
                    **prediction.metadata,
                }))
        return candidates

    def _generate_candidate_pool_from_model(
        self,
        train_papers: list[PaperRecord],
        cutoff_month: str,
        *,
        model_name: str | None,
        predictor_config_path: str,
        temperature: float | None,
        selection_config: SelectionConfig,
    ) -> list[IdeaPrediction]:
        candidates: list[IdeaPrediction] = []
        for sample in self._sampling_plan(selection_config, temperature):
            predictions = generate_predictions(
                train_papers=self._shuffle_train_papers(train_papers, sample["context_shuffle_seed"]),
                cutoff_month=cutoff_month,
                top_k=1,
                model_name=model_name,
                predictor_config_path=predictor_config_path,
                temperature=sample["sampling_temperature"],
                top_p=sample["sampling_top_p"],
                seed=sample["candidate_sample_index"],
                fallback_to_heuristic=False,
            )
            for prediction in predictions[:1]:
                candidates.append(dataclasses.replace(prediction, metadata={
                    **sample,
                    **prediction.metadata,
                }))
        return candidates

    def generate(
        self,
        train_papers: list[PaperRecord],
        cutoff_month: str,
        top_k: int,
    ) -> list[IdeaPrediction]:
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
        import os as _os
        checkpoint_path = str(manifest.get("checkpoint_path") or "").strip()
        # When LIBENCH_POLICY_RL_REMOTE=1 bypass the local-checkpoint path so
        # generation goes through the OpenAI-compatible client (routed to a
        # vLLM server serving the LoRA-adapted model via OPENAI_BASE_URL).
        if _os.environ.get("LIBENCH_POLICY_RL_REMOTE", "") == "1":
            checkpoint_path = ""
        if checkpoint_path and Path(checkpoint_path).expanduser().exists():
            candidates = self._generate_from_local_checkpoint(
                train_papers=train_papers,
                cutoff_month=cutoff_month,
                selection_config=resolved_selection_config,
                checkpoint_path=str(Path(checkpoint_path).expanduser()),
                predictor_config_path=resolved_predictor_config,
                temperature=resolved_temperature,
                base_model_name=(
                    str(manifest.get("base_model_name") or "").strip()
                    or str(manifest.get("inference_model_name") or "").strip()
                    or None
                ),
            )
            selected = select_top_k_predictions(
                candidates,
                train_papers,
                resolved_selection_config,
                top_k=top_k,
            )
            predictions = selected
        else:
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
                selection_config=resolved_selection_config,
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
