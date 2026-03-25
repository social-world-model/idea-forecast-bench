"""ForecasterPipeline: orchestrates all 4 phases of the forecasting method."""
from __future__ import annotations

from dataclasses import replace
import json
import logging
from pathlib import Path
from typing import Any, Optional

from live_idea_bench.backtest import split_train_future_by_cutoff
from live_idea_bench.models import PaperRecord
from live_idea_bench.llm import create_client
from live_idea_bench.papers import month_start_date
from live_idea_bench.similarity import score_prediction_list

from forecaster.models import (
    HindsightSample,
    Innovation,
    ScoredProposal,
    innovation_schema_contract,
)
from forecaster.prior.sampler import sample_innovations
from forecaster.realization.proposal_generator import proposal_to_idea_prediction
from forecaster.config import (
    HindsightConfig,
    PriorConfig,
    SFTTrainConfig,
    RealizationConfig,
    InferenceConfig,
    strict_inference_score_contract,
    load_hindsight_config,
    load_prior_config,
    load_sft_train_config,
    load_realization_config,
    load_inference_config,
)
from forecaster.prior.memory import (
    MemoryStore,
    build_memory_store_from_hindsight_samples,
    hindsight_sample_available_by_cutoff,
)
from forecaster.hindsight.dataset_builder import build_hindsight_dataset
from forecaster.prior.sft_dataset import build_sft_samples, save_sft_dataset
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

        # Persist samples to disk for reproducibility
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
        logger.info(
            "Building SFT dataset from %d hindsight samples.", len(hindsight_samples)
        )
        if memory_snapshots_by_cutoff is None:
            sft_samples = build_sft_samples(hindsight_samples)
        else:
            sft_samples = build_sft_samples(
                hindsight_samples,
                memory_snapshots_by_cutoff=memory_snapshots_by_cutoff,
            )

        prior_output_dir = self.output_dir / output_subdir
        prior_output_dir.mkdir(parents=True, exist_ok=True)
        save_sft_dataset(sft_samples, prior_output_dir / "dataset.jsonl")
        resolved_config = config_override or self.sft_config
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
                        1 for event in delayed_matches if bool(event.get("future_support_confirmed"))
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
                    "train_cutoffs": list(sorted(train_cutoffs)),
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

            manifest = run_policy_rl_pipeline(
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
            manifest_path = str(
                Path(realization_output_dir) / "pipeline_manifest.json"
            )
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
        prior_model_path: Optional[str] = None,
        realization_model_path: Optional[str] = None,
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

        # Filter papers to training window (up to and including cutoff_month)
        training_papers = [
            p for p in self.papers if p.month <= cutoff_month
        ]

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

        # Phase 1: Hindsight extraction
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

        # Phase 2: Prior SFT
        prior_checkpoint: str = ""
        bootstrap_prior_checkpoint: str = ""
        refresh_prior_checkpoint: str = ""
        refresh_memory_snapshots: dict[str, MemoryStore] = {}
        if not skip_training and training_hindsight_samples:
            logger.info("Phase 2: Prior SFT training.")
            prior_checkpoint = self.run_prior_training(training_hindsight_samples)
            bootstrap_prior_checkpoint = prior_checkpoint
        elif not skip_training:
            logger.info("Phase 2: No legal training hindsight samples; skipping prior SFT.")
        else:
            logger.info("Phase 2: Skipping prior SFT training (skip_training=True).")

        # Phase 3: Realization GRPO (skip if skip_training)
        realization_model_path: Optional[str] = None
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
                logger.info("Phase 3 artifact: realization model at %s", realization_model_path)
            else:
                if use_strict_eval:
                    raise RuntimeError(
                        "Strict mode requires a realization artifact for joint inference."
                    )
                logger.info("Phase 3: No realization model checkpoint found; Phase 4 will use demo LLM fallback.")
                fallback_events.append(
                    {
                        "phase": "realization",
                        "fallback": "llm_generation",
                        "reason": "artifact_missing_after_training",
                    }
                )
        elif not skip_training:
            logger.info("Phase 3: No train cutoffs available; skipping realization training.")
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

        # Phase 4: Joint inference on the eval cutoff month
        last_cutoff = eval_cutoff
        proposals: list[ScoredProposal] = []
        if last_cutoff:
            base_memory_samples = (
                training_hindsight_samples
                if use_strict_eval
                else hindsight_samples
            )
            self._memory_store = build_memory_store_from_hindsight_samples(
                base_memory_samples,
                last_cutoff,
            ).decay_recency(last_cutoff)
            if use_strict_eval and refresh_memory_snapshots:
                latest_refresh_cutoff = sorted(refresh_memory_snapshots)[-1]
                latest_refresh_memory = refresh_memory_snapshots[latest_refresh_cutoff]
                utility_overrides = {
                    entry.source_paper_id: (float(entry.utility_score), dict(entry.metadata))
                    for entry in latest_refresh_memory.inventory.entries
                }
                self._memory_store = self._memory_store.apply_utility_overrides(utility_overrides)
            pre_inference_memory_path = snapshot_dir / f"{last_cutoff}_pre_inference.json"
            self._memory_store.persist(pre_inference_memory_path)
            self._memory_store.persist(self.output_dir / "memory_inventory.json")

            logger.info("Phase 4: Joint inference at cutoff %s.", last_cutoff)
            training_papers = [p for p in self.papers if p.month <= last_cutoff]

            if use_strict_eval and (not prior_checkpoint or not Path(prior_checkpoint).exists()):
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
                        "Prior sampling failed (%s); falling back to heuristic demo path.", exc
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
                logger.info("No prior checkpoint available; using heuristic innovations.")
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
                post_update_memory_path = snapshot_dir / f"{last_cutoff}_post_update.json"
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


def _resolve_pipeline_cutoffs(
    cutoff_months: list[str],
    *,
    strict_eval: bool,
    eval_cutoff_month: str | None = None,
) -> tuple[list[str], str]:
    """Split train/eval cutoffs under the frozen runtime contract."""
    if not cutoff_months:
        return [], ""

    if eval_cutoff_month:
        if eval_cutoff_month not in cutoff_months:
            raise ValueError(
                f"eval_cutoff_month {eval_cutoff_month!r} is not present in cutoff_months"
            )
        eval_cutoff = eval_cutoff_month
    else:
        eval_cutoff = cutoff_months[-1]

    if not strict_eval:
        return list(cutoff_months), eval_cutoff
    return [cutoff for cutoff in cutoff_months if cutoff < eval_cutoff], eval_cutoff


def _filter_training_hindsight_samples(
    hindsight_samples: list[HindsightSample],
    eval_cutoff: str,
    *,
    strict_eval: bool,
) -> list[HindsightSample]:
    """Keep only hindsight labels that are legal for training before eval."""
    if not strict_eval:
        return sorted(
            hindsight_samples,
            key=lambda sample: (
                sample.cutoff_month,
                sample.future_paper_published_date,
                sample.future_paper_id,
            ),
        )
    return sorted(
        (
            sample
            for sample in hindsight_samples
            if hindsight_sample_available_by_cutoff(sample, eval_cutoff)
        ),
        key=lambda sample: (
            sample.cutoff_month,
            sample.future_paper_published_date,
            sample.future_paper_id,
        ),
    )


def _persist_cutoff_memory_snapshots(
    hindsight_samples: list[HindsightSample],
    cutoff_months: list[str],
    snapshot_dir: Path,
) -> None:
    """Persist one legal memory snapshot per cutoff."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for cutoff in cutoff_months:
        memory = build_memory_store_from_hindsight_samples(hindsight_samples, cutoff)
        memory.persist(snapshot_dir / f"{cutoff}.json")


def _persist_runtime_contract(
    *,
    output_dir: Path,
    strict_eval: bool,
    train_cutoffs: list[str],
    eval_cutoff: str,
    bootstrap_prior_checkpoint: str,
    refresh_prior_checkpoint: str,
    prior_checkpoint: str,
    realization_model_path: str | None,
    score_contract: dict[str, Any],
    fallback_events: list[dict[str, Any]] | None = None,
) -> None:
    """Persist the paper-faithful runtime contract for reproducibility."""
    payload = {
        "implementation_source_of_truth": "paper/main.tex method section and Algorithm 1",
        "runtime_mode": "strict_eval" if strict_eval else "demo",
        "train_cutoffs": train_cutoffs,
        "eval_cutoff": eval_cutoff,
        "innovation_contract": innovation_schema_contract(),
        "score_contract": score_contract,
        "artifacts": {
            "bootstrap_prior_checkpoint": bootstrap_prior_checkpoint,
            "refresh_prior_checkpoint": refresh_prior_checkpoint,
            "final_prior_checkpoint": prior_checkpoint,
            "prior_checkpoint": prior_checkpoint,
            "realization_model_path": realization_model_path or "",
            "memory_snapshot_dir": str(output_dir / "memory_snapshots"),
            "memory_inventory": str(output_dir / "memory_inventory.json"),
            "prior_refresh_manifest": str(output_dir / "prior_refresh" / "refresh_manifest.json"),
        },
        "fallback_events": list(fallback_events or []),
    }
    (output_dir / "runtime_contract.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _score_proposals_for_delayed_feedback(
    *,
    papers: list[PaperRecord],
    proposals: list[ScoredProposal],
    cutoff_month: str,
    horizon_months: int,
) -> list[dict[str, Any]]:
    """Score proposals against actual future papers for delayed utility updates."""
    if not proposals:
        return []

    train_papers, future_papers, _future_end_month, future_end_date = split_train_future_by_cutoff(
        papers=papers,
        cutoff_month=cutoff_month,
        horizon_months=horizon_months,
    )
    if not future_papers:
        return []

    predictions = [
        proposal_to_idea_prediction(
            proposal_text=proposal.proposal_text,
            innovation=proposal.innovation,
            rank=proposal.rank or index,
        )
        for index, proposal in enumerate(proposals, start=1)
    ]
    scored = score_prediction_list(
        predictions=predictions,
        train_papers=train_papers,
        future_papers=future_papers,
        k=len(predictions),
        cutoff_date=month_start_date(cutoff_month),
        future_end_date=future_end_date,
    )
    match_by_rank = {
        int(match.prediction_rank): match
        for match in scored.matches
    }
    events: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposals, start=1):
        prediction_rank = int(proposal.rank or index)
        match = match_by_rank.get(prediction_rank)
        matched_future_ids = (
            [str(match.paper_id)]
            if match and match.is_match and match.paper_id
            else []
        )
        events.append(
            {
                "proposal_rank": prediction_rank,
                "matched_future_paper_ids": matched_future_ids,
                "future_match_score": float(match.score) if match else 0.0,
                "future_match_lead_time": float(match.lead_time) if match else 0.0,
                "future_match_reasoning": (match.matched_reasoning if match else None) or "",
                "future_support_confirmed": bool(matched_future_ids),
            }
        )
    return events


def _apply_delayed_utility_update(
    memory_store: MemoryStore,
    proposals: list[ScoredProposal],
    future_match_events: list[dict[str, Any]],
    *,
    cutoff_month: str | None = None,
) -> MemoryStore:
    """Apply delayed utility updates to memory based on proposal evaluation.

    Utility is driven by actual proposal-level future support from the evaluation
    backend rather than historical evidence overlap.
    """
    updated = memory_store
    ema_alpha = 0.3
    match_by_rank = {
        int(event.get("proposal_rank", 0)): event
        for event in future_match_events
    }
    for proposal in proposals:
        match_event = match_by_rank.get(int(proposal.rank), {})
        matched_future_ids = [
            str(paper_id)
            for paper_id in match_event.get("matched_future_paper_ids", [])
            if str(paper_id).strip()
        ]
        matched = bool(matched_future_ids)
        utility_delta = 1.0 if matched else -0.1
        for entry in updated.inventory.entries:
            inn = entry.innovation
            if (
                inn.base_direction == proposal.innovation.base_direction
                and inn.operator == proposal.innovation.operator
                and inn.gap == proposal.innovation.gap
            ):
                new_utility = (ema_alpha * utility_delta) + ((1.0 - ema_alpha) * entry.utility_score)
                event = {
                    "cutoff_month": cutoff_month or "",
                    "source_paper_id": entry.source_paper_id,
                    "proposal_rank": int(proposal.rank),
                    "proposal_title": proposal.proposal_text.splitlines()[0].strip() if proposal.proposal_text.strip() else "",
                    "proposal_text": proposal.proposal_text,
                    "proposal_prior_score": float(proposal.prior_score),
                    "proposal_realization_score": float(proposal.realization_score),
                    "proposal_joint_score": float(proposal.joint_score),
                    "evidence_paper_ids": list(proposal.evidence_paper_ids),
                    "matched_future_paper_ids": matched_future_ids,
                    "future_support_confirmed": matched,
                    "future_match_score": float(match_event.get("future_match_score", 0.0) or 0.0),
                    "future_match_lead_time": float(match_event.get("future_match_lead_time", 0.0) or 0.0),
                    "future_match_reasoning": str(match_event.get("future_match_reasoning", "") or ""),
                    "utility_delta": float(utility_delta),
                    "pre_update_utility": float(entry.utility_score),
                    "post_update_utility": float(new_utility),
                    "innovation": {
                        "base_direction": proposal.innovation.base_direction,
                        "operator": proposal.innovation.operator,
                        "gap": proposal.innovation.gap,
                    },
                }
                history = list(entry.metadata.get("delayed_feedback_history", []))
                history.append(event)
                updated = updated.update_utility(
                    entry.source_paper_id,
                    utility_delta,
                    ema_alpha=ema_alpha,
                    metadata={
                        **entry.metadata,
                        "last_delayed_feedback": event,
                        "delayed_feedback_history": history,
                        "last_matched_future_paper_ids": matched_future_ids,
                    },
                )
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
