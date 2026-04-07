"""ForecasterStrategy: wraps run_joint_inference as an IdeaStrategy."""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, List, Optional

from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.model_refs import resolve_model_reference
from live_idea_bench.strategy.base import IdeaStrategy

logger = logging.getLogger(__name__)


class ForecasterStrategy(IdeaStrategy):
    """Benchmark-facing wrapper around the forecaster runtime.

    This strategy intentionally keeps heuristic fallbacks for benchmark
    integration. The paper-faithful strict-eval entrypoint remains
    ``forecaster.orchestrator.ForecasterPipeline``.
    """

    name = "forecaster"
    runtime_surface = "benchmark_wrapper"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        memory_path: str | None = None,
        prior_checkpoint: str | None = None,
        realization_checkpoint: str | None = None,
        inference_config_path: str = "inference.yaml",
        realization_config_path: str = "realization.yaml",
    ) -> None:
        self.model_name = model_name
        self.memory_path = memory_path or ""
        self.prior_checkpoint = prior_checkpoint or None
        self.realization_checkpoint = realization_checkpoint or None
        self.inference_config_path = inference_config_path
        self.realization_config_path = realization_config_path

    def _load_inference_config(self) -> Any:
        from forecaster.config import load_inference_config, InferenceConfig

        try:
            return load_inference_config(self.inference_config_path)
        except (FileNotFoundError, ValueError):
            return InferenceConfig()

    def _load_realization_config(self) -> Any:
        from forecaster.config import load_realization_config, RealizationConfig

        try:
            return load_realization_config(self.realization_config_path)
        except (FileNotFoundError, ValueError):
            return RealizationConfig()

    def _artifact_exists(self, raw_path: str | None) -> bool:
        return bool(raw_path and Path(str(raw_path)).exists())

    def _resolve_model_reference(self, raw_path: str | None) -> str | None:
        return resolve_model_reference(raw_path)

    def _resolve_runtime_mode(self, inference_config: Any) -> dict[str, Any]:
        requested_mode = str(getattr(inference_config, "runtime_mode", "demo") or "demo").strip().lower()
        requested_mode = requested_mode or "demo"
        missing_artifacts: list[str] = []
        if not self._artifact_exists(self.memory_path):
            missing_artifacts.append("memory_snapshot")
        if not self._artifact_exists(self.prior_checkpoint):
            missing_artifacts.append("prior_checkpoint")
        if not self._artifact_exists(self.realization_checkpoint):
            missing_artifacts.append("realization_checkpoint")

        strict_ready = not missing_artifacts
        effective_mode = "strict_eval" if requested_mode == "strict_eval" and strict_ready else "demo"
        fallback_events: list[dict[str, Any]] = []
        if requested_mode == "strict_eval" and not strict_ready:
            fallback_events.append(
                {
                    "phase": "runtime_boundary",
                    "fallback": "demo_wrapper",
                    "reason": "missing_strict_artifacts",
                    "missing_artifacts": list(missing_artifacts),
                }
            )
        return {
            "requested_mode": requested_mode,
            "effective_mode": effective_mode,
            "missing_artifacts": missing_artifacts,
            "fallback_events": fallback_events,
        }

    def _load_memory_store(
        self,
        train_papers: Optional[List] = None,
        cutoff_month: Optional[str] = None,
        *,
        strict_mode: bool = False,
    ) -> Any:
        from forecaster.prior.memory import MemoryStore

        if self.memory_path:
            try:
                store = MemoryStore.load(self.memory_path)
                if cutoff_month and store.inventory.last_updated_month > cutoff_month:
                    logger.warning(
                        "Memory last_updated_month=%s is newer than cutoff_month=%s; "
                        "this may introduce temporal leakage.",
                        store.inventory.last_updated_month,
                        cutoff_month,
                    )
                return store
            except Exception as exc:
                if strict_mode:
                    raise RuntimeError(
                        f"Strict mode could not load memory store from {self.memory_path!r}: {exc}"
                    ) from exc
                logger.warning("Could not load memory store from %r: %s", self.memory_path, exc)

        if strict_mode:
            raise FileNotFoundError(
                "Strict forecaster serving requires a memory snapshot artifact."
            )

        # No explicit memory path: build a minimal memory from training papers
        # so p(z|M_t) has meaningful conditioning rather than returning -2.0 for all innovations.
        if train_papers:
            return self._build_memory_from_papers(train_papers, cutoff_month or "1970-01")
        return MemoryStore.empty("1970-01")

    def _build_memory_from_papers(self, train_papers: List, current_month: str) -> Any:
        """Build a MemoryStore from training papers as a heuristic M_t.

        Creates Innovation entries from paper metadata chronologically,
        mirroring how the orchestrator populates memory from hindsight samples.
        """
        from forecaster.prior.memory import MemoryStore
        from forecaster.models import Innovation

        store = MemoryStore.empty(current_month)
        for paper in train_papers:
            keywords = paper.keywords or []
            base_direction = (
                " ".join(keywords[:3]) if keywords else " ".join(paper.title.split()[:5])
            )
            gap = paper.summary[:100] if paper.summary else paper.title
            innovation = Innovation(
                base_direction=base_direction,
                operator="extend",
                gap=gap,
            )
            store = store.append(innovation, paper.paper_id, paper.month)
        return store

    def _build_heuristic_innovations(
        self,
        train_papers: List[PaperRecord],
        top_k: int,
    ) -> list:
        """Create Innovation objects from paper keywords as a heuristic fallback."""
        from forecaster.models import Innovation

        max_innovations = top_k * 3
        innovations: list[Innovation] = []

        for paper in train_papers:
            keywords = paper.keywords or []
            base_direction = (
                " ".join(keywords[:3])
                if keywords
                else " ".join(paper.title.split()[:5])
            )
            gap = paper.summary[:100] if paper.summary else paper.title

            innovations.append(
                Innovation(
                    base_direction=base_direction,
                    operator="extend",
                    gap=gap,
                )
            )
            if len(innovations) >= max_innovations:
                break

        return innovations

    def generate(
        self,
        train_papers: List[PaperRecord],
        cutoff_month: str,
        top_k: int,
    ) -> List[IdeaPrediction]:
        """Generate forecasts using joint inference.

        If prior_checkpoint is provided, samples innovations from the trained model.
        Otherwise, uses a heuristic fallback: creates innovations from paper keywords.

        Converts ScoredProposal → IdeaPrediction for benchmark compatibility.
        """
        from forecaster.inference.algorithm import run_joint_inference
        from forecaster.realization.proposal_generator import proposal_to_idea_prediction

        if not train_papers:
            return []

        from forecaster.inference.algorithm import run_joint_inference
        from forecaster.prior.sampler import sample_innovations

        inference_config = self._load_inference_config()
        realization_config = self._load_realization_config()

        prior_model_path = self._resolve_model_reference(self.prior_checkpoint)
        realization_model_path = self._resolve_model_reference(self.realization_checkpoint)

        # Force flexible runtime when using local models — strict_eval requires
        # memory snapshots we don't have; demo mode falls back to LLM API.
        # Flexible: use trained models when available, heuristic scoring otherwise.
        # Force flexible runtime (not strict_eval) for local model inference.
        # When SGLang handles proposal generation, skip expensive conditional_logprob
        # realization scorer to enable the batched fast path.
        import os
        replace_kwargs: dict[str, Any] = {"runtime_mode": "flexible"}
        if os.environ.get("SGLANG_URL") or os.environ.get("SGLANG_PRIOR_URL"):
            replace_kwargs["realization_score_method"] = "heuristic"
            replace_kwargs["prior_score_method"] = "heuristic"
        inference_config = dataclasses.replace(inference_config, **replace_kwargs)

        fallback_events: list[dict[str, Any]] = []
        memory_store = self._load_memory_store(
            train_papers=train_papers,
            cutoff_month=cutoff_month,
            strict_mode=False,
        )

        # Sample innovations from trained prior, or fall back to heuristic
        innovations: list
        if prior_model_path:
            try:
                sampled = sample_innovations(
                    prior_model_path, memory_store, inference_config
                )
                innovations = sampled if sampled else []
                if innovations:
                    logger.info("Sampled %d innovations from trained prior.", len(innovations))
                else:
                    logger.warning("Prior sampling returned empty; using heuristic.")
            except Exception as exc:
                logger.warning("Prior sampling failed (%s); using heuristic.", exc)
                innovations = []

            if not innovations:
                innovations = self._build_heuristic_innovations(train_papers, top_k=top_k)
                fallback_events.append({"phase": "prior", "fallback": "heuristic_innovations"})
        else:
            innovations = self._build_heuristic_innovations(train_papers, top_k=top_k)
            fallback_events.append({"phase": "prior", "fallback": "heuristic_innovations", "reason": "no_checkpoint"})

        if not innovations:
            logger.warning("No innovations generated for cutoff=%s", cutoff_month)
            return []

        # LLM client only needed when no local realization model
        llm_client, model = None, ""
        if not realization_model_path:
            try:
                from live_idea_bench.llm import create_client
                llm_client, model = create_client(self.model_name or "gpt-4o")
            except Exception as exc:
                logger.warning("No LLM client and no realization model: %s", exc)
                return []

        proposals = run_joint_inference(
            innovations=innovations,
            papers=train_papers,
            memory_store=memory_store,
            llm_client=llm_client,
            model=model,
            inference_config=inference_config,
            realization_config=realization_config,
            prior_model_path=prior_model_path,
            realization_model_path=realization_model_path,
        )

        # Convert ScoredProposal → IdeaPrediction with 1-indexed ranks
        predictions: list[IdeaPrediction] = []
        for idx, proposal in enumerate(proposals[:top_k], start=1):
            prediction = proposal_to_idea_prediction(
                proposal_text=proposal.proposal_text,
                innovation=proposal.innovation,
                rank=idx,
            )
            prediction = dataclasses.replace(
                prediction,
                rank=idx,
                score=float(proposal.joint_score),
                metadata={
                    "prior_score": proposal.prior_score,
                    "realization_score": proposal.realization_score,
                    "joint_score": proposal.joint_score,
                    "strategy": self.name,
                    "runtime_surface": self.runtime_surface,
                    "effective_runtime_mode": "flexible",
                    "fallback_events": list(fallback_events),
                },
            )
            predictions.append(prediction)

        return predictions
