"""ForecasterStrategy: wraps run_joint_inference as an IdeaStrategy."""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, List

from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.strategy.base import IdeaStrategy

logger = logging.getLogger(__name__)


class ForecasterStrategy(IdeaStrategy):
    """Strategy using the full forecaster pipeline (prior + realization + inference).

    At inference time, uses:
    1. A MemoryStore built from the training papers
    2. Pre-sampled innovations (or heuristic fallback)
    3. run_joint_inference to produce proposals
    4. Converts ScoredProposals to IdeaPredictions
    """

    name = "forecaster"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        memory_path: str | None = None,
        prior_checkpoint: str | None = None,
        inference_config_path: str = "inference.yaml",
        realization_config_path: str = "realization.yaml",
    ) -> None:
        self.model_name = model_name
        self.memory_path = memory_path or ""
        self.prior_checkpoint = prior_checkpoint or None
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

    def _load_memory_store(self) -> Any:
        from forecaster.prior.memory import MemoryStore

        if self.memory_path:
            try:
                return MemoryStore.load(self.memory_path)
            except Exception as exc:
                logger.warning("Could not load memory store from %r: %s", self.memory_path, exc)
        return MemoryStore.empty("1970-01")

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

        inference_config = self._load_inference_config()
        realization_config = self._load_realization_config()
        memory_store = self._load_memory_store()

        # Build innovations: use trained prior if checkpoint is available, else heuristic fallback
        innovations: list
        if self.prior_checkpoint and Path(self.prior_checkpoint).exists():
            try:
                from forecaster.prior.sampler import sample_innovations

                sampled = sample_innovations(
                    self.prior_checkpoint, memory_store, inference_config
                )
                if sampled:
                    innovations = sampled
                    logger.info(
                        "Using %d innovations from trained prior.", len(innovations)
                    )
                else:
                    logger.warning(
                        "Prior sampling empty; falling back to heuristic."
                    )
                    innovations = self._build_heuristic_innovations(
                        train_papers, top_k=top_k
                    )
            except Exception as exc:
                logger.warning(
                    "Prior sampling failed (%s); falling back to heuristic.", exc
                )
                innovations = self._build_heuristic_innovations(
                    train_papers, top_k=top_k
                )
        else:
            innovations = self._build_heuristic_innovations(train_papers, top_k=top_k)

        if not innovations:
            return []

        # Resolve LLM client
        resolved_model = self.model_name or "gpt-4o"
        try:
            from live_idea_bench.llm import create_client

            llm_client, model = create_client(resolved_model)
        except Exception as exc:
            logger.warning(
                "Could not create LLM client for model %r: %s. Returning empty list.",
                resolved_model,
                exc,
            )
            return []

        proposals = run_joint_inference(
            innovations=innovations,
            papers=train_papers,
            memory_store=memory_store,
            llm_client=llm_client,
            model=model,
            inference_config=inference_config,
            realization_config=realization_config,
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
                },
            )
            predictions.append(prediction)

        return predictions
