"""Wire the Phase-4 reward into the TRL runner without disturbing legacy.

The TRL runner already wraps a `compute_score(...)` call in a TRL-style
`reward_fn(completions, **kwargs)`. This module exposes a single factory:

    make_reward_fn(config) -> Callable

When `config.reward_mode == "legacy"`, the returned callable is identical
to the existing wrapper. When `config.reward_mode == "foresight"`, we
load a ForesightContext from disk (indices + rubrics + embedder + judge)
and dispatch to compute_score_v2.

Artifact directory layout expected when reward_mode == "foresight":

    <foresight_artifact_dir>/
        indices/
            future_<cutoff>.npz, future_<cutoff>.meta.json
            history_<cutoff>.npz, history_<cutoff>.meta.json
        rubrics/
            <topic>.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _load_indices(indices_dir: Path):
    from forecaster.foresight.indices import FutureIndex, HistoryIndex
    future_indices: dict = {}
    history_indices: dict = {}
    for npz in sorted(indices_dir.glob("future_*.npz")):
        cutoff = npz.stem.split("future_", 1)[1]
        future_indices[cutoff] = FutureIndex.load(npz)
    for npz in sorted(indices_dir.glob("history_*.npz")):
        cutoff = npz.stem.split("history_", 1)[1]
        history_indices[cutoff] = HistoryIndex.load(npz)
    return future_indices, history_indices


def _build_embedder(name: str):
    from forecaster.foresight.indices import HashingEmbedder, SentenceTransformerEmbedder
    if name.startswith("hashing:"):
        try:
            dim = int(name.split(":", 1)[1])
        except ValueError:
            dim = 256
        return HashingEmbedder(dim=dim)
    if name.startswith("sentence-transformer:"):
        return SentenceTransformerEmbedder(model_name=name.split(":", 1)[1])
    return SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")


def _build_judge(mode: str):
    from forecaster.foresight.judge import (
        RubricJudge,
        StubScorer,
        make_live_scorer,
    )
    if mode == "stub":
        # Deterministic constant for smoke runs.
        def fixed(idea: str, candidate: str) -> float:
            return 0.5
        return RubricJudge(scorer=StubScorer(fn=fixed, name="stub-fixed"))
    return RubricJudge(scorer=make_live_scorer())


def build_foresight_context(
    artifact_dir: str | Path,
    *,
    embedder_name: str = "sentence-transformer:all-MiniLM-L6-v2",
    judge_mode: str = "live",
):
    """Construct a ForesightContext from a saved artifact directory."""
    from forecaster.foresight.indices import CutoffIndexBundle
    from forecaster.foresight.reward import ForesightContext
    from forecaster.foresight.rubric import load_rubrics_dir

    root = Path(artifact_dir)
    indices_dir = root / "indices"
    rubrics_dir = root / "rubrics"
    if not indices_dir.exists():
        raise FileNotFoundError(f"missing indices dir: {indices_dir}")
    if not rubrics_dir.exists():
        raise FileNotFoundError(f"missing rubrics dir: {rubrics_dir}")
    future_indices, history_indices = _load_indices(indices_dir)
    rubrics = load_rubrics_dir(rubrics_dir)
    embedder = _build_embedder(embedder_name)
    judge = _build_judge(judge_mode)
    return ForesightContext(
        embedder=embedder,
        judge=judge,
        future_indices=future_indices,
        history_indices=history_indices,
        rubrics=rubrics,
    )


# ---------------------------------------------------------------- TRL reward_fn factory


def make_reward_fn(
    config: Any,
    *,
    trainer_name: str = "grpo",
    reward_config_path: str = "reward.yaml",
    realization_config_path: str = "realization.yaml",
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str = "config.yaml",
    model_name: str | None = None,
) -> Callable[..., list[float]]:
    """Return a TRL-compatible `reward_fn(completions, **kwargs) -> list[float]`.

    Honors `config.reward_mode`:
        legacy    — forecaster.realization.verl.reward_fn.compute_score
        foresight — forecaster.foresight.reward.compute_score_v2
    """
    reward_mode = str(getattr(config, "reward_mode", "legacy") or "legacy").strip().lower()
    num_generations = int(getattr(config, "num_generations", 0) or 0)
    grouping_assert = bool(getattr(config, "grouping_assert", True))
    dedup_penalty = float(getattr(config, "dedup_penalty", 0.0) or 0.0)
    dedup_threshold = float(getattr(config, "dedup_jaccard_threshold", 0.85) or 0.85)

    if reward_mode == "foresight":
        artifact_dir = str(getattr(config, "foresight_artifact_dir", "") or "").strip()
        if not artifact_dir:
            raise ValueError(
                "reward_mode=foresight requires `foresight_artifact_dir` to be set in the GRPO config."
            )
        ctx = build_foresight_context(
            artifact_dir,
            embedder_name=getattr(config, "foresight_embedder",
                                  "sentence-transformer:all-MiniLM-L6-v2"),
            judge_mode=getattr(config, "foresight_judge_mode", "live"),
        )
        logger.info(
            "reward_mode=foresight loaded ctx: future_indices=%d history_indices=%d rubrics=%d",
            len(ctx.future_indices), len(ctx.history_indices), len(ctx.rubrics),
        )

        from forecaster.foresight.grouping import (
            assert_group_invariant,
            compute_dedup_penalties,
            grouping_report,
        )
        from forecaster.foresight.reward import compute_score_v2

        # Tick once on the first batch so the log shows whether grouping is
        # what we expect — and the assert below will tell us if it isn't.
        ticked = {"value": False}

        def _reward_fn(completions: list[str], **kwargs: Any) -> list[float]:
            extra_infos = kwargs.get("extra_info", ["{}"] * len(completions))
            if num_generations > 0 and grouping_assert:
                assert_group_invariant(extra_infos, num_generations=num_generations)
            if not ticked["value"] and num_generations > 0:
                report = grouping_report(extra_infos, num_generations=num_generations)
                logger.info("group invariant first-batch report: %s", report)
                ticked["value"] = True

            out: list[float] = []
            for completion, extra in zip(completions, extra_infos):
                try:
                    out.append(compute_score_v2(
                        data_source=f"live_idea_bench::{trainer_name}",
                        solution_str=completion,
                        ground_truth="",
                        extra_info=extra,
                        ctx=ctx,
                    ))
                except Exception as exc:
                    logger.warning("foresight reward failed: %s", exc, exc_info=True)
                    out.append(0.0)

            if dedup_penalty > 0.0 and num_generations >= 2:
                penalties = compute_dedup_penalties(
                    completions,
                    num_generations=num_generations,
                    threshold=dedup_threshold,
                    penalty=dedup_penalty,
                )
                out = [max(0.0, r - p) for r, p in zip(out, penalties)]
            return out

        _reward_fn.__name__ = "foresight_reward_fn"
        return _reward_fn

    # ---- legacy path ----
    from forecaster.realization.verl.reward_fn import compute_score

    def _reward_fn(completions: list[str], **kwargs: Any) -> list[float]:
        extra_infos = kwargs.get("extra_info", ["{}"] * len(completions))
        out: list[float] = []
        for completion, extra in zip(completions, extra_infos):
            try:
                out.append(compute_score(
                    data_source=f"live_idea_bench::{trainer_name}",
                    solution_str=completion,
                    ground_truth="",
                    extra_info=extra,
                    reward_config_path=reward_config_path,
                    realization_config_path=realization_config_path,
                    similarity_config_path=similarity_config_path,
                    runtime_config_path=runtime_config_path,
                    model_name=model_name,
                ))
            except Exception as exc:
                logger.warning("legacy reward failed: %s", exc, exc_info=True)
                out.append(0.0)
        return out

    _reward_fn.__name__ = "legacy_reward_fn"
    return _reward_fn


__all__ = ["make_reward_fn", "build_foresight_context"]
