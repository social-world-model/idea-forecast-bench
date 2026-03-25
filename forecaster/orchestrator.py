"""ForecasterPipeline: orchestrates all 4 phases of the forecasting method."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from live_idea_bench.models import PaperRecord
from live_idea_bench.llm import create_client

from forecaster.models import HindsightSample, Innovation, ScoredProposal
from forecaster.prior.sampler import sample_innovations
from forecaster.config import (
    HindsightConfig,
    PriorConfig,
    SFTTrainConfig,
    RealizationConfig,
    InferenceConfig,
    load_hindsight_config,
    load_prior_config,
    load_sft_train_config,
    load_realization_config,
    load_inference_config,
)
from forecaster.prior.memory import MemoryStore
from forecaster.hindsight.dataset_builder import build_hindsight_dataset
from forecaster.prior.sft_dataset import build_sft_samples
from forecaster.prior.trainer import train_prior
from forecaster.inference.algorithm import run_joint_inference as run_joint_inference_fn


logger = logging.getLogger(__name__)


class ForecasterPipeline:
    """Orchestrates all 4 phases of the factorized forecasting method.

    Phases:
    1. Hindsight extraction: build D_z from historical papers
    2. Prior SFT: train p_θ(z|M_t) on D_z
    3. Realization GRPO: train p_ψ(y|z,X) [delegates to existing RL pipeline]
    4. Joint inference: Algorithm 1 to produce ranked forecast

    Usage:
        pipeline = ForecasterPipeline(papers=papers, output_dir="output/forecaster")
        proposals = pipeline.run_full_pipeline(cutoff_months=["2024-06"], horizon_months=6)
    """

    def __init__(
        self,
        papers: list[PaperRecord],
        output_dir: str | Path = "output/forecaster",
        *,
        hindsight_config: HindsightConfig | None = None,
        prior_config: PriorConfig | None = None,
        sft_config: SFTTrainConfig | None = None,
        realization_config: RealizationConfig | None = None,
        inference_config: InferenceConfig | None = None,
        llm_model: str = "gpt-4o",
    ) -> None:
        self.papers = papers
        self.output_dir = Path(output_dir)
        self.hindsight_config = hindsight_config or load_hindsight_config()
        self.prior_config = prior_config or load_prior_config()
        self.sft_config = sft_config or load_sft_train_config()
        self.realization_config = realization_config or load_realization_config()
        self.inference_config = inference_config or load_inference_config()
        self.llm_model = llm_model
        self._memory_store: MemoryStore = MemoryStore.empty("1970-01")

    def run_hindsight_extraction(
        self,
        cutoff_months: list[str],
        horizon_months: int = 6,
    ) -> list[HindsightSample]:
        """Phase 1: extract hindsight innovations from historical papers."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        llm_client, model = create_client(self.llm_model)

        logger.info(
            "Starting hindsight extraction for %d cutoff months.", len(cutoff_months)
        )
        samples = build_hindsight_dataset(
            papers=self.papers,
            cutoff_months=cutoff_months,
            horizon_months=horizon_months,
            config=self.hindsight_config,
            llm_client=llm_client,
            model=model,
        )
        logger.info("Hindsight extraction complete: %d samples.", len(samples))

        # Persist samples to disk for reproducibility
        hindsight_path = self.output_dir / "hindsight_samples.json"
        hindsight_path.write_text(
            json.dumps(
                [
                    {
                        "context_paper_ids": list(s.context_paper_ids),
                        "cutoff_month": s.cutoff_month,
                        "future_paper_id": s.future_paper_id,
                        "innovation": {
                            "base_direction": s.innovation.base_direction,
                            "operator": s.innovation.operator,
                            "gap": s.innovation.gap,
                        },
                    }
                    for s in samples
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return samples

    def run_prior_training(
        self,
        hindsight_samples: list[HindsightSample],
    ) -> str:
        """Phase 2: train the innovation prior via SFT.

        Returns path to checkpoint.
        """
        logger.info(
            "Building SFT dataset from %d hindsight samples.", len(hindsight_samples)
        )
        sft_samples = build_sft_samples(hindsight_samples)

        prior_output_dir = self.output_dir / "prior_sft"
        logger.info("Training prior SFT model; output dir: %s", prior_output_dir)

        checkpoint_path = train_prior(
            sft_samples=sft_samples,
            config=self.sft_config,
            output_dir=prior_output_dir,
        )
        logger.info("Prior SFT training complete. Checkpoint: %s", checkpoint_path)
        return checkpoint_path

    def run_realization_training(
        self,
        cutoff_months: list[str],
        horizon_months: int = 6,
        *,
        model_name: str | None = None,
        dry_run: bool = False,
    ) -> str | None:
        """Phase 3: train the realization module via GRPO.

        Delegates to the existing RL pipeline. Returns manifest path or None.
        """
        from forecaster.realization.pipeline import run_policy_rl_pipeline
        from forecaster.realization.config import (
            load_episode_build_config,
            load_candidate_generation_config,
            load_reward_config,
            load_selection_config,
        )
        from forecaster.realization.trainers import create_trainer_runner

        realization_output_dir = str(self.output_dir / "realization_grpo")
        resolved_model = model_name or "gpt-4o-mini"

        logger.info(
            "Starting realization GRPO training (dry_run=%s, model=%s).",
            dry_run,
            resolved_model,
        )

        try:
            episode_config = load_episode_build_config()
            candidate_config = load_candidate_generation_config()
            reward_config = load_reward_config()
            selection_config = load_selection_config()

            manifest = run_policy_rl_pipeline(
                self.papers,
                trainer="grpo",
                model_name=resolved_model,
                output_dir=realization_output_dir,
                episode_config=episode_config,
                candidate_config=candidate_config,
                reward_config=reward_config,
                selection_config=selection_config,
                trainer_config=None,
                trainer_config_path="grpo.yaml",
                selection_config_path="selection.yaml",
                prepare_only=dry_run,
                skip_alignment_check=True,
            )
            manifest_path = str(
                Path(realization_output_dir) / "pipeline_manifest.json"
            )
            logger.info("Realization training complete. Manifest: %s", manifest_path)
            return manifest_path
        except Exception as exc:
            logger.warning("Realization training failed: %s", exc)
            return None

    def run_joint_inference(
        self,
        cutoff_month: str,
        innovations: list[Innovation],
        *,
        realization_model_path: Optional[str] = None,
    ) -> list[ScoredProposal]:
        """Phase 4: run Algorithm 1 for joint inference.

        Args:
            cutoff_month: The forecasting cutoff month.
            innovations: Pre-sampled candidate innovations from the prior.
            realization_model_path: Optional path to the GRPO-trained realization
                checkpoint (from Phase 3). When provided, proposal generation uses
                p_ψ(y|z,X) trained model instead of the generic LLM client.

        Returns:
            Top-K ScoredProposal objects.
        """
        llm_client, model = create_client(self.llm_model)

        # Filter papers to training window (up to and including cutoff_month)
        training_papers = [
            p for p in self.papers if p.month <= cutoff_month
        ]

        logger.info(
            "Running joint inference: %d innovations, %d training papers, "
            "realization_model_path=%s.",
            len(innovations),
            len(training_papers),
            realization_model_path or "none (LLM fallback)",
        )

        proposals = run_joint_inference_fn(
            innovations=innovations,
            papers=training_papers,
            memory_store=self._memory_store,
            llm_client=llm_client,
            model=model,
            inference_config=self.inference_config,
            realization_config=self.realization_config,
            realization_model_path=realization_model_path,
        )

        logger.info("Joint inference complete: %d proposals.", len(proposals))
        return proposals

    def run_full_pipeline(
        self,
        cutoff_months: list[str],
        horizon_months: int = 6,
        *,
        skip_training: bool = False,
    ) -> dict[str, Any]:
        """Run all 4 phases sequentially.

        Returns dict with keys:
        - "hindsight_samples": list[HindsightSample]
        - "prior_checkpoint": str (path)
        - "proposals": list[ScoredProposal] (from last cutoff month)
        - "output_dir": str
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Hindsight extraction
        logger.info("Phase 1: Hindsight extraction.")
        hindsight_samples = self.run_hindsight_extraction(
            cutoff_months=cutoff_months,
            horizon_months=horizon_months,
        )

        # Populate memory store chronologically from hindsight samples (Gap 3 fix)
        sorted_samples = sorted(hindsight_samples, key=lambda s: s.cutoff_month)
        for sample in sorted_samples:
            self._memory_store = self._memory_store.append(
                sample.innovation,
                sample.future_paper_id,
                sample.cutoff_month,
            )

        # Phase 2: Prior SFT
        prior_checkpoint: str = ""
        if not skip_training:
            logger.info("Phase 2: Prior SFT training.")
            prior_checkpoint = self.run_prior_training(hindsight_samples)
        else:
            logger.info("Phase 2: Skipping prior SFT training (skip_training=True).")

        # Phase 3: Realization GRPO (skip if skip_training)
        realization_model_path: Optional[str] = None
        if not skip_training:
            logger.info("Phase 3: Realization GRPO training.")
            manifest_path = self.run_realization_training(
                cutoff_months=cutoff_months,
                horizon_months=horizon_months,
            )
            realization_model_path = _extract_realization_model_path(manifest_path)
            if realization_model_path:
                logger.info("Phase 3 artifact: realization model at %s", realization_model_path)
            else:
                logger.info("Phase 3: No realization model checkpoint found; Phase 4 will use LLM fallback.")
        else:
            logger.info("Phase 3: Skipping realization training (skip_training=True).")

        # Phase 4: Joint inference on the last cutoff month
        last_cutoff = sorted(cutoff_months)[-1] if cutoff_months else ""
        proposals: list[ScoredProposal] = []
        if last_cutoff:
            # Apply recency decay and persist memory before inference
            self._memory_store = self._memory_store.decay_recency(last_cutoff)
            self._memory_store.persist(self.output_dir / "memory_inventory.json")

            logger.info("Phase 4: Joint inference at cutoff %s.", last_cutoff)
            training_papers = [p for p in self.papers if p.month <= last_cutoff]

            # Use trained prior if checkpoint available; otherwise fall back to heuristic (Gap 1 fix)
            if prior_checkpoint and Path(prior_checkpoint).exists():
                try:
                    innovations = sample_innovations(
                        prior_checkpoint, self._memory_store, self.inference_config
                    )
                    if not innovations:
                        logger.warning(
                            "Prior sampling returned no innovations; falling back to heuristic."
                        )
                        innovations = _heuristic_innovations(
                            training_papers, n=self.inference_config.num_candidates
                        )
                except Exception as exc:
                    logger.warning(
                        "Prior sampling failed (%s); falling back to heuristic.", exc
                    )
                    innovations = _heuristic_innovations(
                        training_papers, n=self.inference_config.num_candidates
                    )
            else:
                logger.info("No prior checkpoint available; using heuristic innovations.")
                innovations = _heuristic_innovations(
                    training_papers, n=self.inference_config.num_candidates
                )

            proposals = self.run_joint_inference(
                cutoff_month=last_cutoff,
                innovations=innovations,
                realization_model_path=realization_model_path,
            )

            # Delayed utility update: close the self-evolving memory loop.
            # Use the next cutoff month's papers (if any) as the "future" for
            # evaluating how useful each proposal's source innovation was.
            # When running a single cutoff, future papers come from hindsight samples.
            future_paper_ids = {s.future_paper_id for s in sorted_samples if s.cutoff_month == last_cutoff}
            if future_paper_ids:
                self._memory_store = _apply_delayed_utility_update(
                    self._memory_store, proposals, future_paper_ids
                )
                self._memory_store.persist(self.output_dir / "memory_inventory.json")
                logger.info("Delayed utility update applied; memory persisted.")

        return {
            "hindsight_samples": hindsight_samples,
            "prior_checkpoint": prior_checkpoint,
            "realization_model_path": realization_model_path or "",
            "proposals": proposals,
            "output_dir": str(self.output_dir),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_realization_model_path(manifest_path: Optional[str]) -> Optional[str]:
    """Extract the trained realization model path from the pipeline manifest.

    The manifest's trainer_output_dir is where the GRPO-trained realization
    checkpoint lands. Returns None if the manifest is absent or incomplete.
    """
    if not manifest_path:
        return None
    path = Path(manifest_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        trainer_output_dir = payload.get("trainer_output_dir", "")
        if trainer_output_dir and Path(trainer_output_dir).exists():
            return trainer_output_dir
    except Exception:
        pass
    return None


def _apply_delayed_utility_update(
    memory_store: MemoryStore,
    proposals: list[ScoredProposal],
    future_paper_ids: set[str],
) -> MemoryStore:
    """Apply delayed utility updates to memory based on proposal evaluation.

    For each proposal, determines whether the innovation's source paper appeared
    in the future paper set (a proxy for the innovation being correct/useful).
    A match yields a positive utility delta; no match yields a small negative delta.

    This closes the self-evolving loop: M_t^+ = U(M_t, Δ_t) as described
    in the paper's abstract.

    Args:
        memory_store: Current memory store.
        proposals: Ranked proposals from joint inference.
        future_paper_ids: Set of future paper IDs used as ground truth.

    Returns:
        Updated MemoryStore with utility scores adjusted via EMA.
    """
    updated = memory_store
    for proposal in proposals:
        # A proposal is considered "useful" if any evidence paper it relied on
        # appears in the future set, or if the proposal's innovation is semantically
        # represented in the future papers (approximated by evidence overlap).
        evidence_ids = set(proposal.evidence_paper_ids)
        matched = bool(evidence_ids & future_paper_ids)
        utility_delta = 1.0 if matched else -0.1
        # Update memory entry for the source paper that generated this innovation
        # (stored in source_paper_id when memory was populated from hindsight)
        for entry in updated.inventory.entries:
            inn = entry.innovation
            if (
                inn.base_direction == proposal.innovation.base_direction
                and inn.operator == proposal.innovation.operator
                and inn.gap == proposal.innovation.gap
            ):
                updated = updated.update_utility(entry.source_paper_id, utility_delta)
                break
    return updated


def _heuristic_innovations(
    papers: list[PaperRecord],
    n: int = 16,
) -> list[Innovation]:
    """Build Innovation objects heuristically from paper metadata.

    Used as a fallback when no trained prior model is available.
    """
    innovations: list[Innovation] = []
    for paper in papers:
        keywords = paper.keywords or []
        base_direction = " ".join(keywords[:3]) if keywords else " ".join(paper.title.split()[:5])
        gap = paper.summary[:100] if paper.summary else paper.title
        innovation = Innovation(
            base_direction=base_direction,
            operator="extend",
            gap=gap,
        )
        innovations.append(innovation)
        if len(innovations) >= n:
            break
    return innovations
