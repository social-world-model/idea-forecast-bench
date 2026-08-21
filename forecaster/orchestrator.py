"""ForecasterPipeline: orchestrates all 4 phases of the forecasting method."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from forecaster.config import (
    HindsightConfig,
    InferenceConfig,
    PriorConfig,
    RealizationConfig,
    SFTTrainConfig,
    load_hindsight_config,
    load_inference_config,
    load_prior_config,
    load_realization_config,
    load_sft_train_config,
    strict_inference_score_contract,
)
from forecaster.hindsight.dataset_builder import build_hindsight_dataset
from forecaster.inference.algorithm import run_joint_inference as run_joint_inference_fn
from forecaster.models import (
    HindsightSample,
    Innovation,
    ScoredProposal,
)
from forecaster.orchestration_helpers import (
    _apply_delayed_utility_update,
    _extract_realization_model_path,
    _filter_training_hindsight_samples,
    _heuristic_innovations,
    _persist_cutoff_memory_snapshots,
    _persist_runtime_contract,
    _resolve_pipeline_cutoffs,
    _score_proposals_for_delayed_feedback,
)
from forecaster.prior.memory import (
    MemoryStore,
    build_memory_store_from_hindsight_samples,
)
from forecaster.prior.sampler import sample_innovations
from forecaster.prior.sft_dataset import build_sft_samples, save_sft_dataset
from forecaster.prior.trainer import train_prior
from live_idea_bench.llm import create_client
from live_idea_bench.models import PaperRecord

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
        proposals = pipeline.run_full_pipeline(cutoff_months=["2024-06"], horizon_months=3)
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
        horizon_months: int = 3,
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

        hindsight_path = self.output_dir / "hindsight_samples.json"
        hindsight_path.write_text(
            json.dumps(
                [
                    {
                        "context_paper_ids": list(s.context_paper_ids),
                        "cutoff_month": s.cutoff_month,
                        "future_paper_id": s.future_paper_id,
                        "future_paper_published_date": s.future_paper_published_date,
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
        *,
        output_subdir: str = "prior_sft",
        memory_snapshots_by_cutoff: dict[str, MemoryStore] | None = None,
        config_override: SFTTrainConfig | None = None,
    ) -> str:
        """Phase 2: train the innovation prior via SFT.

        Returns path to checkpoint.
        """
        resolved_config = config_override or self.sft_config
        max_mem = resolved_config.max_memory_entries
        logger.info(
            "Building SFT dataset from %d hindsight samples (max_memory_entries=%d).",
            len(hindsight_samples),
            max_mem,
        )
        if memory_snapshots_by_cutoff is None:
            sft_samples = build_sft_samples(
                hindsight_samples,
                max_memory_entries=max_mem,
            )
        else:
            sft_samples = build_sft_samples(
                hindsight_samples,
                max_memory_entries=max_mem,
                # `dict` is invariant, so `dict[str, MemoryStore]` is not a
                # `dict[str, object]`; the callee only reads the mapping.
                memory_snapshots_by_cutoff=cast(
                    "dict[str, object]", memory_snapshots_by_cutoff
                ),
            )

        prior_output_dir = self.output_dir / output_subdir
        prior_output_dir.mkdir(parents=True, exist_ok=True)
        save_sft_dataset(sft_samples, prior_output_dir / "dataset.jsonl")
        logger.info("Training prior SFT model; output dir: %s", prior_output_dir)

        checkpoint_path = train_prior(
            sft_samples=sft_samples,
            config=resolved_config,
            output_dir=prior_output_dir,
        )
        logger.info("Prior SFT training complete. Checkpoint: %s", checkpoint_path)
        return checkpoint_path

    def run_prior_refresh(
        self,
        *,
        train_cutoffs: list[str],
        horizon_months: int,
        hindsight_samples: list[HindsightSample],
        bootstrap_prior_checkpoint: str,
        realization_model_path: str,
    ) -> tuple[str, dict[str, MemoryStore]]:
        """Replay train cutoffs, update utility, and train a short refresh prior."""
        if not train_cutoffs:
            return "", {}

        llm_client, model = create_client(self.llm_model)
        refresh_dir = self.output_dir / "prior_refresh"
        refresh_dir.mkdir(parents=True, exist_ok=True)
        memory_snapshot_dir = refresh_dir / "memory_snapshots"
        memory_snapshot_dir.mkdir(parents=True, exist_ok=True)

        utility_overrides: dict[str, tuple[float, dict[str, Any] | None]] = {}
        refreshed_snapshots: dict[str, MemoryStore] = {}
        replay_events: list[dict[str, Any]] = []

        for cutoff in sorted(train_cutoffs):
            replay_memory = build_memory_store_from_hindsight_samples(
                hindsight_samples,
                cutoff,
            ).decay_recency(cutoff)
            replay_memory = replay_memory.apply_utility_overrides(utility_overrides)
            replay_memory.persist(memory_snapshot_dir / f"{cutoff}_pre_refresh.json")
            refreshed_snapshots[cutoff] = replay_memory

            training_papers = [paper for paper in self.papers if paper.month <= cutoff]
            innovations = sample_innovations(
                bootstrap_prior_checkpoint,
                replay_memory,
                self.inference_config,
            )
            proposals = run_joint_inference_fn(
                innovations=innovations,
                papers=training_papers,
                memory_store=replay_memory,
                llm_client=llm_client,
                model=model,
                inference_config=self.inference_config,
                realization_config=self.realization_config,
                prior_model_path=bootstrap_prior_checkpoint,
                realization_model_path=realization_model_path,
            )
            delayed_matches = _score_proposals_for_delayed_feedback(
                papers=self.papers,
                proposals=proposals,
                cutoff_month=cutoff,
                horizon_months=horizon_months,
            )
            updated_memory = _apply_delayed_utility_update(
                replay_memory,
                proposals,
                delayed_matches,
                cutoff_month=cutoff,
            )
            updated_memory.persist(memory_snapshot_dir / f"{cutoff}_post_refresh.json")
            refreshed_snapshots[cutoff] = updated_memory
            utility_overrides = {
                entry.source_paper_id: (
                    float(entry.utility_score),
                    dict(entry.metadata),
                )
                for entry in updated_memory.inventory.entries
            }
            replay_events.append(
                {
                    "cutoff_month": cutoff,
                    "innovation_count": len(innovations),
                    "proposal_count": len(proposals),
                    "future_match_count": sum(
                        1
                        for event in delayed_matches
                        if bool(event.get("future_support_confirmed"))
                    ),
                }
            )

        refresh_config = replace(
            self.sft_config,
            num_epochs=1,
            output_dir=str((self.output_dir / "prior_refresh").resolve()),
        )
        refresh_checkpoint = self.run_prior_training(
            hindsight_samples,
            output_subdir="prior_refresh",
            memory_snapshots_by_cutoff=refreshed_snapshots,
            config_override=refresh_config,
        )
        (refresh_dir / "refresh_manifest.json").write_text(
            json.dumps(
                {
                    "bootstrap_prior_checkpoint": bootstrap_prior_checkpoint,
                    "refresh_prior_checkpoint": refresh_checkpoint,
                    "realization_model_path": realization_model_path,
                    "train_cutoffs": sorted(train_cutoffs),
                    "memory_snapshot_dir": str(memory_snapshot_dir.resolve()),
                    "replay_events": replay_events,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return refresh_checkpoint, refreshed_snapshots

    def run_realization_training(
        self,
        cutoff_months: list[str],
        horizon_months: int = 3,
        *,
        hindsight_samples: list[HindsightSample] | None = None,
        model_name: str | None = None,
        dry_run: bool = False,
        strict_mode: bool | None = None,
    ) -> str | None:
        """Phase 3: train the realization module via GRPO.

        Delegates to the existing RL pipeline. Returns manifest path or None.
        """
        from forecaster.realization.config import (
            load_candidate_generation_config,
            load_episode_build_config,
            load_reward_config,
            load_selection_config,
        )
        from forecaster.realization.pipeline import run_policy_rl_pipeline

        realization_output_dir = str(self.output_dir / "realization_grpo")
        resolved_model = model_name or "gpt-4o-mini"
        use_strict_mode = (
            self.inference_config.runtime_mode == "strict_eval"
            if strict_mode is None
            else strict_mode
        )

        logger.info(
            "Starting realization GRPO training (dry_run=%s, strict_mode=%s, model=%s).",
            dry_run,
            use_strict_mode,
            resolved_model,
        )

        try:
            episode_config = load_episode_build_config()
            episode_config.horizon_months = horizon_months
            candidate_config = load_candidate_generation_config()
            reward_config = load_reward_config()
            selection_config = load_selection_config()

            run_policy_rl_pipeline(
                self.papers,
                trainer="grpo",
                model_name=resolved_model,
                output_dir=realization_output_dir,
                episode_config=episode_config,
                candidate_config=candidate_config,
                realization_config=self.realization_config,
                reward_config=reward_config,
                selection_config=selection_config,
                trainer_config=None,
                trainer_config_path="grpo.yaml",
                selection_config_path="selection.yaml",
                strict_mode=use_strict_mode,
                prepare_only=dry_run,
                skip_alignment_check=(dry_run or not use_strict_mode),
                hindsight_samples=hindsight_samples,
            )
            manifest_path = str(Path(realization_output_dir) / "pipeline_manifest.json")
            logger.info("Realization training complete. Manifest: %s", manifest_path)
            return manifest_path
        except Exception as exc:
            logger.warning("Realization training failed: %s", exc)
            if use_strict_mode and not dry_run:
                raise
            return None

    def run_joint_inference(
        self,
        cutoff_month: str,
        innovations: list[Innovation],
        *,
        prior_model_path: str | None = None,
        realization_model_path: str | None = None,
    ) -> list[ScoredProposal]:
        """Phase 4: run Algorithm 1 for joint inference.

        Args:
            cutoff_month: The forecasting cutoff month.
            innovations: Pre-sampled candidate innovations from the prior.
            prior_model_path: Optional path to the trained prior checkpoint. When
                provided, inference scores candidates with conditional log-probability
                under p_theta(z | M_t); otherwise it falls back to memory heuristics.
            realization_model_path: Optional path to the GRPO-trained realization
                checkpoint (from Phase 3). When provided, proposal generation uses
                p_ψ(y|z,X) trained model instead of the generic LLM client.

        Returns:
            Top-K ScoredProposal objects.
        """
        llm_client, model = create_client(self.llm_model)

        training_papers = [p for p in self.papers if p.month <= cutoff_month]

        logger.info(
            "Running joint inference: %d innovations, %d training papers, "
            "realization_model_path=%s.",
            len(innovations),
            len(training_papers),
            realization_model_path or "none",
        )

        proposals = run_joint_inference_fn(
            innovations=innovations,
            papers=training_papers,
            memory_store=self._memory_store,
            llm_client=llm_client,
            model=model,
            inference_config=self.inference_config,
            realization_config=self.realization_config,
            prior_model_path=prior_model_path,
            realization_model_path=realization_model_path,
        )

        logger.info("Joint inference complete: %d proposals.", len(proposals))
        return proposals

    def run_full_pipeline(
        self,
        cutoff_months: list[str],
        horizon_months: int = 3,
        *,
        skip_training: bool = False,
        strict_eval: bool | None = None,
        eval_cutoff_month: str | None = None,
    ) -> dict[str, Any]:
        """Run all 4 phases sequentially.

        Returns dict with keys:
        - "hindsight_samples": list[HindsightSample]
        - "prior_checkpoint": str (path)
        - "proposals": list[ScoredProposal] (from last cutoff month)
        - "output_dir": str
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sorted_cutoffs = sorted(set(cutoff_months))
        if not sorted_cutoffs:
            return {
                "hindsight_samples": [],
                "training_hindsight_samples": [],
                "prior_checkpoint": "",
                "realization_model_path": "",
                "proposals": [],
                "output_dir": str(self.output_dir),
            }

        use_strict_eval = (
            self.inference_config.runtime_mode == "strict_eval"
            if strict_eval is None
            else strict_eval
        )
        train_cutoffs, eval_cutoff = _resolve_pipeline_cutoffs(
            sorted_cutoffs,
            strict_eval=use_strict_eval,
            eval_cutoff_month=eval_cutoff_month,
        )
        fallback_events: list[dict[str, Any]] = []

        logger.info("Phase 1: Hindsight extraction.")
        hindsight_samples = self.run_hindsight_extraction(
            cutoff_months=sorted_cutoffs,
            horizon_months=horizon_months,
        )
        training_hindsight_samples = _filter_training_hindsight_samples(
            hindsight_samples,
            eval_cutoff,
            strict_eval=use_strict_eval,
        )
        snapshot_dir = self.output_dir / "memory_snapshots"
        _persist_cutoff_memory_snapshots(
            hindsight_samples if not use_strict_eval else training_hindsight_samples,
            sorted_cutoffs,
            snapshot_dir,
        )

        prior_checkpoint: str = ""
        bootstrap_prior_checkpoint: str = ""
        refresh_prior_checkpoint: str = ""
        refresh_memory_snapshots: dict[str, MemoryStore] = {}
        if not skip_training and training_hindsight_samples:
            logger.info("Phase 2: Prior SFT training.")
            prior_checkpoint = self.run_prior_training(training_hindsight_samples)
            bootstrap_prior_checkpoint = prior_checkpoint
        elif not skip_training:
            logger.info(
                "Phase 2: No legal training hindsight samples; skipping prior SFT."
            )
        else:
            logger.info("Phase 2: Skipping prior SFT training (skip_training=True).")

        realization_model_path: str | None = None
        if not skip_training and train_cutoffs:
            logger.info("Phase 3: Realization GRPO training.")
            manifest_path = self.run_realization_training(
                cutoff_months=train_cutoffs,
                horizon_months=horizon_months,
                hindsight_samples=training_hindsight_samples,
                strict_mode=use_strict_eval,
            )
            realization_model_path = _extract_realization_model_path(manifest_path)
            if realization_model_path:
                logger.info(
                    "Phase 3 artifact: realization model at %s", realization_model_path
                )
            else:
                if use_strict_eval:
                    raise RuntimeError(
                        "Strict mode requires a realization artifact for joint inference."
                    )
                logger.info(
                    "Phase 3: No realization model checkpoint found; Phase 4 will use demo LLM fallback."
                )
                fallback_events.append(
                    {
                        "phase": "realization",
                        "fallback": "llm_generation",
                        "reason": "artifact_missing_after_training",
                    }
                )
        elif not skip_training:
            logger.info(
                "Phase 3: No train cutoffs available; skipping realization training."
            )
        else:
            logger.info("Phase 3: Skipping realization training (skip_training=True).")
            if not realization_model_path and not use_strict_eval:
                fallback_events.append(
                    {
                        "phase": "realization",
                        "fallback": "llm_generation",
                        "reason": "skip_training",
                    }
                )

        if (
            use_strict_eval
            and not skip_training
            and training_hindsight_samples
            and train_cutoffs
            and prior_checkpoint
            and realization_model_path
        ):
            logger.info("Phase 3.5: Prior refresh via offline replay.")
            refresh_prior_checkpoint, refresh_memory_snapshots = self.run_prior_refresh(
                train_cutoffs=train_cutoffs,
                horizon_months=horizon_months,
                hindsight_samples=training_hindsight_samples,
                bootstrap_prior_checkpoint=prior_checkpoint,
                realization_model_path=realization_model_path,
            )
            if refresh_prior_checkpoint:
                prior_checkpoint = refresh_prior_checkpoint

        last_cutoff = eval_cutoff
        proposals: list[ScoredProposal] = []
        if last_cutoff:
            base_memory_samples = (
                training_hindsight_samples if use_strict_eval else hindsight_samples
            )
            self._memory_store = build_memory_store_from_hindsight_samples(
                base_memory_samples,
                last_cutoff,
            ).decay_recency(last_cutoff)
            if use_strict_eval and refresh_memory_snapshots:
                latest_refresh_cutoff = sorted(refresh_memory_snapshots)[-1]
                latest_refresh_memory = refresh_memory_snapshots[latest_refresh_cutoff]
                utility_overrides: dict[str, tuple[float, dict[str, Any] | None]] = {
                    entry.source_paper_id: (
                        float(entry.utility_score),
                        dict(entry.metadata),
                    )
                    for entry in latest_refresh_memory.inventory.entries
                }
                self._memory_store = self._memory_store.apply_utility_overrides(
                    utility_overrides
                )
            pre_inference_memory_path = (
                snapshot_dir / f"{last_cutoff}_pre_inference.json"
            )
            self._memory_store.persist(pre_inference_memory_path)
            self._memory_store.persist(self.output_dir / "memory_inventory.json")

            logger.info("Phase 4: Joint inference at cutoff %s.", last_cutoff)
            training_papers = [p for p in self.papers if p.month <= last_cutoff]

            if use_strict_eval and (
                not prior_checkpoint or not Path(prior_checkpoint).exists()
            ):
                raise RuntimeError(
                    "Strict mode requires a legal prior checkpoint for innovation sampling."
                )
            if use_strict_eval and not realization_model_path:
                raise RuntimeError(
                    "Strict mode requires a realization artifact for proposal generation and scoring."
                )

            if prior_checkpoint and Path(prior_checkpoint).exists():
                try:
                    innovations = sample_innovations(
                        prior_checkpoint, self._memory_store, self.inference_config
                    )
                    if not innovations:
                        if use_strict_eval:
                            raise RuntimeError(
                                "Strict mode prior sampling returned no innovations."
                            )
                        logger.warning(
                            "Prior sampling returned no innovations; falling back to heuristic demo path."
                        )
                        innovations = _heuristic_innovations(
                            training_papers, n=self.inference_config.num_candidates
                        )
                        fallback_events.append(
                            {
                                "phase": "prior",
                                "fallback": "heuristic_innovations",
                                "reason": "empty_prior_samples",
                            }
                        )
                except Exception as exc:
                    if use_strict_eval:
                        raise RuntimeError(
                            f"Strict mode prior sampling failed: {exc}"
                        ) from exc
                    logger.warning(
                        "Prior sampling failed (%s); falling back to heuristic demo path.",
                        exc,
                    )
                    innovations = _heuristic_innovations(
                        training_papers, n=self.inference_config.num_candidates
                    )
                    fallback_events.append(
                        {
                            "phase": "prior",
                            "fallback": "heuristic_innovations",
                            "reason": "prior_sampling_error",
                            "detail": str(exc),
                        }
                    )
            else:
                logger.info(
                    "No prior checkpoint available; using heuristic innovations."
                )
                innovations = _heuristic_innovations(
                    training_papers, n=self.inference_config.num_candidates
                )
                fallback_events.append(
                    {
                        "phase": "prior",
                        "fallback": "heuristic_innovations",
                        "reason": "missing_prior_checkpoint",
                    }
                )

            proposals = self.run_joint_inference(
                cutoff_month=last_cutoff,
                innovations=innovations,
                prior_model_path=prior_checkpoint or None,
                realization_model_path=realization_model_path,
            )

            delayed_matches = _score_proposals_for_delayed_feedback(
                papers=self.papers,
                proposals=proposals,
                cutoff_month=last_cutoff,
                horizon_months=horizon_months,
            )
            if delayed_matches:
                self._memory_store = _apply_delayed_utility_update(
                    self._memory_store,
                    proposals,
                    delayed_matches,
                    cutoff_month=last_cutoff,
                )
                post_update_memory_path = (
                    snapshot_dir / f"{last_cutoff}_post_update.json"
                )
                self._memory_store.persist(post_update_memory_path)
                self._memory_store.persist(self.output_dir / "memory_inventory.json")
                logger.info("Delayed utility update applied; memory persisted.")

        score_contract = (
            strict_inference_score_contract(
                score_normalization=self.inference_config.score_normalization,
                score_temperature=self.inference_config.score_temperature,
            )
            if use_strict_eval
            else {
                "prior_score_method": self.inference_config.prior_score_method,
                "realization_score_method": self.inference_config.realization_score_method,
                "score_normalization": self.inference_config.score_normalization,
                "score_temperature": self.inference_config.score_temperature,
                "joint_score_mode": self.inference_config.joint_score_mode,
                "popularity_weight": self.inference_config.popularity_weight,
            }
        )

        _persist_runtime_contract(
            output_dir=self.output_dir,
            strict_eval=use_strict_eval,
            train_cutoffs=train_cutoffs,
            eval_cutoff=eval_cutoff,
            bootstrap_prior_checkpoint=bootstrap_prior_checkpoint,
            refresh_prior_checkpoint=refresh_prior_checkpoint,
            prior_checkpoint=prior_checkpoint,
            realization_model_path=realization_model_path,
            score_contract=score_contract,
            fallback_events=fallback_events,
        )

        return {
            "hindsight_samples": hindsight_samples,
            "training_hindsight_samples": training_hindsight_samples,
            "prior_checkpoint": prior_checkpoint,
            "bootstrap_prior_checkpoint": bootstrap_prior_checkpoint,
            "refresh_prior_checkpoint": refresh_prior_checkpoint,
            "realization_model_path": realization_model_path or "",
            "proposals": proposals,
            "output_dir": str(self.output_dir),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
