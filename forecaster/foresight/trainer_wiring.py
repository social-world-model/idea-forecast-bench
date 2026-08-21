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
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # imported lazily at runtime to keep the heavy deps optional
    from forecaster.foresight.indices import Embedder, FutureIndex, HistoryIndex
    from forecaster.foresight.judge import RubricJudge
    from forecaster.foresight.reward import ForesightContext

logger = logging.getLogger(__name__)


def _load_indices(
    indices_dir: Path,
) -> tuple[dict[str, FutureIndex], dict[str, HistoryIndex]]:
    from forecaster.foresight.indices import FutureIndex, HistoryIndex

    future_indices: dict[str, FutureIndex] = {}
    history_indices: dict[str, HistoryIndex] = {}
    for npz in sorted(indices_dir.glob("future_*.npz")):
        cutoff = npz.stem.split("future_", 1)[1]
        future_indices[cutoff] = FutureIndex.load(npz)
    for npz in sorted(indices_dir.glob("history_*.npz")):
        cutoff = npz.stem.split("history_", 1)[1]
        history_indices[cutoff] = HistoryIndex.load(npz)
    return future_indices, history_indices


def _build_paper_to_topic(hindsight_path: str | Path | None = None) -> dict[str, str]:
    """Map future_paper_id -> topic_id from the hindsight/dz artifacts.

    The GRPO dataset's extra_info drops topic_id (HindsightSample has no such
    field), but the foresight reward needs it to select the rubric. We recover
    it from target_future_paper_id at reward time via this map.
    """
    candidates: list[Path] = []
    if hindsight_path:
        candidates.append(Path(hindsight_path))
    candidates += [
        Path("output/hindsight_samples.jsonl"),
        Path("data/topic_hindsight/dz.jsonl"),
    ]
    mapping: dict[str, str] = {}
    for p in candidates:
        if not p.exists():
            continue
        with p.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = d.get("future_paper_id") or d.get("source_future_id")
                tid = d.get("topic_id")
                if pid and tid and str(pid) not in mapping:
                    mapping[str(pid)] = str(tid)
        if mapping:
            logger.info("paper_to_topic: %d entries from %s", len(mapping), p)
            break
    if not mapping:
        logger.warning("paper_to_topic map is EMPTY — rubric lookup will fail")
    return mapping


def _build_embedder(name: str) -> Embedder:
    from forecaster.foresight.indices import (
        HashingEmbedder,
        SentenceTransformerEmbedder,
    )

    if name.startswith("hashing:"):
        try:
            dim = int(name.split(":", 1)[1])
        except ValueError:
            dim = 256
        return HashingEmbedder(dim=dim)
    if name.startswith("sentence-transformer:"):
        return SentenceTransformerEmbedder(model_name=name.split(":", 1)[1])
    return SentenceTransformerEmbedder(
        model_name="sentence-transformers/allenai-specter"
    )


def _build_judge(mode: str) -> RubricJudge:
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
    embedder_name: str = "sentence-transformer:sentence-transformers/allenai-specter",
    judge_mode: str = "live",
    hindsight_path: str | Path | None = None,
) -> ForesightContext:
    """Construct a ForesightContext from a saved artifact directory."""
    from forecaster.foresight.reward import ForesightContext
    from forecaster.foresight.rubric import load_rubrics_dir

    root = Path(artifact_dir)
    indices_dir = root / "indices"
    rubrics_dir = root / "rubrics"
    _hint = (
        "reward_mode=foresight needs prebuilt artifacts under "
        f"{root}/ (per-cutoff future/history indices + validated rubrics). "
        "Build them first — see forecaster/foresight/README.md and examples/forecaster/build_indices.py "
        "— or set reward_mode: legacy in config/forecaster/grpo_train.yaml to use "
        "the fixed-weight composite reward instead."
    )
    if not indices_dir.exists():
        raise FileNotFoundError(
            f"missing foresight indices dir: {indices_dir}\n{_hint}"
        )
    if not rubrics_dir.exists():
        raise FileNotFoundError(
            f"missing foresight rubrics dir: {rubrics_dir}\n{_hint}"
        )
    future_indices, history_indices = _load_indices(indices_dir)
    rubrics = load_rubrics_dir(rubrics_dir)
    embedder = _build_embedder(embedder_name)
    judge = _build_judge(judge_mode)
    paper_to_topic = _build_paper_to_topic(hindsight_path)
    return ForesightContext(
        embedder=embedder,
        judge=judge,
        future_indices=future_indices,
        history_indices=history_indices,
        rubrics=rubrics,
        paper_to_topic=paper_to_topic,
    )


# ---------------------------------------------------------------- TRL reward_fn factory


#: A handful of transient judge failures is normal; a systematic one is not.
#: Past this many, training stops rather than optimising against zeros.
_MAX_REWARD_FAILURES = 50
_reward_failures = {"n": 0}


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
    reward_mode = (
        str(getattr(config, "reward_mode", "legacy") or "legacy").strip().lower()
    )
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
            embedder_name=getattr(
                config,
                "foresight_embedder",
                "sentence-transformer:sentence-transformers/allenai-specter",
            ),
            judge_mode=getattr(config, "foresight_judge_mode", "live"),
            hindsight_path=getattr(config, "foresight_hindsight_path", None)
            or getattr(config, "hindsight_path", None),
        )
        logger.info(
            "reward_mode=foresight loaded ctx: future_indices=%d history_indices=%d rubrics=%d",
            len(ctx.future_indices),
            len(ctx.history_indices),
            len(ctx.rubrics),
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
            for completion, extra in zip(completions, extra_infos, strict=False):
                # TRL returns conversational completions as a list of message
                # dicts ([{"role":"assistant","content":...}]); the foresight
                # reward/gates expect a plain string. Flatten to the text.
                if isinstance(completion, list):
                    sol = "\n".join(
                        str(m.get("content", ""))
                        for m in completion
                        if isinstance(m, dict)
                    )
                else:
                    sol = completion
                try:
                    out.append(
                        compute_score_v2(
                            data_source=f"live_idea_bench::{trainer_name}",
                            solution_str=sol,
                            ground_truth="",
                            extra_info=extra,
                            ctx=ctx,
                        )
                    )
                except Exception as exc:
                    # 0.0 is a legitimate reward ("bad completion"), so an
                    # infrastructure failure here is indistinguishable from a
                    # genuine low score and the policy trains on the noise.
                    # Tolerate transients, but refuse to keep going once the
                    # failures stop looking transient.
                    logger.warning("foresight reward failed: %s", exc, exc_info=True)
                    _reward_failures["n"] += 1
                    if _reward_failures["n"] > _MAX_REWARD_FAILURES:
                        raise RuntimeError(
                            f"foresight reward failed {_reward_failures['n']} times; "
                            "refusing to continue training on substituted zeros. "
                            "Check the judge endpoint and the foresight artifacts."
                        ) from exc
                    out.append(0.0)

            if dedup_penalty > 0.0 and num_generations >= 2:
                penalties = compute_dedup_penalties(
                    completions,
                    num_generations=num_generations,
                    threshold=dedup_threshold,
                    penalty=dedup_penalty,
                )
                out = [max(0.0, r - p) for r, p in zip(out, penalties, strict=False)]
            return out

        _reward_fn.__name__ = "foresight_reward_fn"
        return _reward_fn

    # ---- legacy path ----
    from forecaster.realization.verl.reward_fn import compute_score

    # Same name as the foresight reward_fn above, but the two definitions sit on
    # mutually exclusive branches (the one above returns before this point).
    def _reward_fn(completions: list[str], **kwargs: Any) -> list[float]:  # type: ignore[no-redef]
        extra_infos = kwargs.get("extra_info", ["{}"] * len(completions))
        out: list[float] = []
        for completion, extra in zip(completions, extra_infos, strict=False):
            try:
                out.append(
                    compute_score(
                        data_source=f"live_idea_bench::{trainer_name}",
                        solution_str=completion,
                        ground_truth="",
                        extra_info=extra,
                        reward_config_path=reward_config_path,
                        realization_config_path=realization_config_path,
                        similarity_config_path=similarity_config_path,
                        runtime_config_path=runtime_config_path,
                        model_name=model_name,
                    )
                )
            except Exception as exc:
                logger.warning("legacy reward failed: %s", exc, exc_info=True)
                out.append(0.0)
        return out

    _reward_fn.__name__ = "legacy_reward_fn"
    return _reward_fn


__all__ = ["make_reward_fn", "build_foresight_context"]
