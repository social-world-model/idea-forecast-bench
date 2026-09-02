from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from forecaster.models import (
    HindsightSample,
    Innovation,
    ScoredProposal,
    innovation_schema_contract,
)
from forecaster.prior.memory import (
    MemoryStore,
    build_memory_store_from_hindsight_samples,
    hindsight_sample_available_by_cutoff,
)
from forecaster.realization.proposal_generator import proposal_to_idea_prediction
from idea_forecast_bench.backtest import split_train_future_by_cutoff
from idea_forecast_bench.models import PaperRecord
from idea_forecast_bench.papers import month_start_date
from idea_forecast_bench.similarity import score_prediction_list

logger = logging.getLogger(__name__)


def _extract_realization_model_path(manifest_path: str | None) -> str | None:
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
        if (
            isinstance(trainer_output_dir, str)
            and trainer_output_dir
            and Path(trainer_output_dir).exists()
        ):
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
            "prior_refresh_manifest": str(
                output_dir / "prior_refresh" / "refresh_manifest.json"
            ),
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

    train_papers, future_papers, _future_end_month, future_end_date = (
        split_train_future_by_cutoff(
            papers=papers,
            cutoff_month=cutoff_month,
            horizon_months=horizon_months,
        )
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
    match_by_rank = {int(match.prediction_rank): match for match in scored.matches}
    events: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposals, start=1):
        prediction_rank = int(proposal.rank or index)
        match = match_by_rank.get(prediction_rank)
        matched_future_ids = (
            [str(match.paper_id)] if match and match.is_match and match.paper_id else []
        )
        events.append(
            {
                "proposal_rank": prediction_rank,
                "matched_future_paper_ids": matched_future_ids,
                "future_match_score": float(match.score) if match else 0.0,
                "future_match_lead_time": float(match.lead_time) if match else 0.0,
                "future_match_reasoning": (match.matched_reasoning if match else None)
                or "",
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
        int(event.get("proposal_rank", 0)): event for event in future_match_events
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
                new_utility = (ema_alpha * utility_delta) + (
                    (1.0 - ema_alpha) * entry.utility_score
                )
                event = {
                    "cutoff_month": cutoff_month or "",
                    "source_paper_id": entry.source_paper_id,
                    "proposal_rank": int(proposal.rank),
                    "proposal_title": proposal.proposal_text.splitlines()[0].strip()
                    if proposal.proposal_text.strip()
                    else "",
                    "proposal_text": proposal.proposal_text,
                    "proposal_prior_score": float(proposal.prior_score),
                    "proposal_realization_score": float(proposal.realization_score),
                    "proposal_joint_score": float(proposal.joint_score),
                    "evidence_paper_ids": list(proposal.evidence_paper_ids),
                    "matched_future_paper_ids": matched_future_ids,
                    "future_support_confirmed": matched,
                    "future_match_score": float(
                        match_event.get("future_match_score", 0.0) or 0.0
                    ),
                    "future_match_lead_time": float(
                        match_event.get("future_match_lead_time", 0.0) or 0.0
                    ),
                    "future_match_reasoning": str(
                        match_event.get("future_match_reasoning", "") or ""
                    ),
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
        base_direction = (
            " ".join(keywords[:3]) if keywords else " ".join(paper.title.split()[:5])
        )
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
