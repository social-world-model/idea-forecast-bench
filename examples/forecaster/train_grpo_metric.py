"""Single-metric GRPO driver for the soft / coverage / novelty experiments.

This bypasses the Prior SFT phase entirely (no warm-start) so each run starts
from the base Qwen3.5 model and is optimized solely by the chosen eval metric.
The reward mode (soft|coverage|novelty) selects which scorer is wired into
``forecaster.realization.reward.evaluate_rl_reward``.

Usage::

    python examples/forecaster/train_grpo_metric.py \\
        --model qwen3.5-9b \\
        --papers data/csml/raw_markdown \\
        --start-month 2023-01 --end-month 2024-09 \\
        --reward-mode soft \\
        --output-dir output/grpo_soft \\
        --use-vllm-server --vllm-server-port 8765

The wrapper script ``scripts/forecaster/run_three_grpo.sh`` orchestrates the
three sequential runs and manages the local judge vLLM server lifecycle.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path

# Single-GPU per process. Three modes run in parallel on separate GPU pairs;
# Unsloth's loader is single-GPU by design so we never go through torchrun /
# DDP here. The launching script sets CUDA_VISIBLE_DEVICES to the one GPU.
import torch  # noqa: E402

if torch.cuda.is_available():
    torch.cuda.set_device(0)


# TRL's VLLMClient defaults to group_port=51216 for NCCL handshake. Three
# parallel runs collide on this port. Patch the default from env var
# LIB_VLLM_GROUP_PORT so each run uses a unique one. Unsloth regenerates its
# cached GRPOTrainer module at import time (clobbers source-level edits),
# so we patch the underlying class here BEFORE unsloth runs.
def _patch_vllm_client_group_port() -> None:
    """Override the NCCL rendezvous port used by TRL's vLLM weight-sync
    so multiple parallel trainer-vLLM pairs on the same host don't
    collide on port 51216 (the default).

    TRL 0.24-era code path: patches ``VLLMClient.__init__`` default.
    TRL 1.4+ code path: patches ``VLLMGeneration.__init__`` default and
    also the GRPOConfig field default ``vllm_group_port``.
    """
    _group_port = int(os.environ.get("LIB_VLLM_GROUP_PORT", "51216"))

    # TRL 1.4+ path: GRPOConfig has a vllm_group_port field; patch its
    # default so unsloth's compiled trainer reads our env-var value.
    try:
        import dataclasses as _dc

        from trl.trainer.grpo_config import GRPOConfig as _Cfg

        for f in _dc.fields(_Cfg):
            if f.name == "vllm_group_port":
                f.default = _group_port
                # Replace the field's default in __dataclass_fields__ too.
                _Cfg.__dataclass_fields__[f.name].default = _group_port
                break
    except Exception:
        pass

    # TRL 1.4+ path: VLLMGeneration.__init__ also has group_port default.
    try:
        from trl.generation import vllm_generation as _vg

        _orig_vg_init = _vg.VLLMGeneration.__init__

        def _patched_vg_init(self, *args, **kwargs):
            kwargs.setdefault("group_port", _group_port)
            return _orig_vg_init(self, *args, **kwargs)

        _vg.VLLMGeneration.__init__ = _patched_vg_init
    except Exception:
        pass

    # TRL 1.4 / 0.24 path: VLLMClient.__init__ default. Belt and suspenders.
    _vc = None
    for mod_path in ("trl.generation.vllm_client", "trl.extras.vllm_client"):
        try:
            _vc = __import__(mod_path, fromlist=["VLLMClient"])
            break
        except Exception:
            continue
    if _vc is None or not hasattr(_vc, "VLLMClient"):
        return
    _orig_init = _vc.VLLMClient.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault("group_port", _group_port)
        return _orig_init(self, *args, **kwargs)

    _vc.VLLMClient.__init__ = _patched_init


_patch_vllm_client_group_port()


from forecaster.models import HindsightSample  # noqa: E402
from forecaster.realization import (  # noqa: E402
    load_candidate_generation_config,
    load_episode_build_config,
    load_grpo_train_config,
    load_reward_config,
    load_selection_config,
)
from forecaster.realization.episodes import build_rl_episodes  # noqa: E402
from forecaster.realization.model_zoo import resolve_small_model  # noqa: E402
from forecaster.realization.pipeline import (  # noqa: E402
    _derive_episode_innovation,
    run_policy_rl_pipeline,
)
from live_idea_bench.models import PaperRecord  # noqa: E402
from live_idea_bench.papers import load_papers_from_markdown  # noqa: E402


def _build_synthetic_hindsight(
    papers: list[PaperRecord],
    episode_config,
    *,
    max_per_episode: int = 80,
) -> list[HindsightSample]:
    """Generate one HindsightSample per future paper in each episode.

    The Innovation is derived from the paper itself via
    ``pipeline._derive_episode_innovation``. This bypasses the expensive
    LLM-based hindsight-extraction pipeline (``run_topic_hindsight.py``,
    which calls GPT-5.4 once per paper) so we can stand up GRPO training
    rows without any API cost.

    Each row carries a synthetic innovation as the latent ``z`` (only used by
    the composite reward / prompt rendering); the single-metric reward modes
    score the rollout directly against the future-paper corpus and ignore z.
    """
    paper_lookup = {p.paper_id: p for p in papers}
    episodes = build_rl_episodes(papers, episode_config)
    samples: list[HindsightSample] = []
    for episode in episodes:
        for fp_id in episode.future_paper_ids[:max_per_episode]:
            fp = paper_lookup.get(fp_id)
            if fp is None:
                continue
            innovation = _derive_episode_innovation(fp)
            samples.append(
                HindsightSample(
                    context_paper_ids=tuple(episode.train_paper_ids),
                    cutoff_month=episode.cutoff_month,
                    future_paper_id=fp.paper_id,
                    future_paper_published_date=fp.published_date,
                    innovation=innovation,
                )
            )
    return samples


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("forecaster.train_grpo_metric")

_VALID_MODES = {"soft", "coverage", "novelty"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GRPO training with a single eval-metric reward."
    )
    p.add_argument(
        "--model",
        default="qwen3.5-9b",
        help="Model alias from forecaster.realization.model_zoo.",
    )
    p.add_argument("--papers", required=True, help="Directory of markdown papers.")
    p.add_argument("--output-dir", required=True, help="Top-level output directory.")
    p.add_argument(
        "--reward-mode",
        required=True,
        choices=sorted(_VALID_MODES),
        help="Which eval metric to optimize as the GRPO reward.",
    )
    p.add_argument(
        "--start-month", default="2023-01", help="Lower month bound (YYYY-MM)."
    )
    p.add_argument(
        "--end-month", default="2024-09", help="Upper month bound (YYYY-MM)."
    )
    p.add_argument("--max-episodes", type=int, default=None, help="Cap GRPO episodes.")
    p.add_argument(
        "--max-grpo-rows", type=int, default=None, help="Cap GRPO training rows."
    )
    p.add_argument(
        "--num-generations", type=int, default=None, help="GRPO group size G."
    )
    p.add_argument("--max-completion-length", type=int, default=None)
    p.add_argument("--max-prompt-length", type=int, default=None)
    p.add_argument("--grpo-epochs", type=int, default=None)
    p.add_argument("--grpo-lr", type=float, default=None)
    p.add_argument("--use-vllm-server", action="store_true")
    p.add_argument("--vllm-server-host", default="localhost")
    p.add_argument("--vllm-server-port", type=int, default=8765)
    p.add_argument(
        "--skip-alignment-check",
        action="store_true",
        default=True,
        help="Bypass the online reward alignment gate (single-metric rewards "
        "won't correlate with the composite benchmark).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Single-metric GRPO")
    log.info("  model:      %s", args.model)
    log.info("  reward:     %s", args.reward_mode)
    log.info("  papers:     %s", args.papers)
    log.info("  window:     %s -> %s", args.start_month, args.end_month)
    log.info("  output:     %s", output_dir)
    log.info("=" * 60)

    log.info("Loading papers ...")
    papers = load_papers_from_markdown(
        input_dir=Path(args.papers),
        start_month=args.start_month,
        end_month=args.end_month,
    )
    log.info("Loaded %d papers.", len(papers))
    if not papers:
        log.error("No papers loaded from %s", args.papers)
        return 1

    episode_config = load_episode_build_config()
    if args.start_month:
        episode_config = replace(episode_config, start_month=args.start_month)
    if args.end_month:
        episode_config = replace(episode_config, end_month=args.end_month)

    candidate_config = load_candidate_generation_config()
    reward_config_path = f"reward_{args.reward_mode}.yaml"
    reward_config = load_reward_config(reward_config_path)
    selection_config = load_selection_config()
    grpo_config = load_grpo_train_config()

    overrides: dict = {}
    if args.grpo_epochs is not None:
        overrides["num_train_epochs"] = args.grpo_epochs
    if args.grpo_lr is not None:
        overrides["learning_rate"] = args.grpo_lr
    if args.num_generations is not None:
        overrides["num_generations"] = args.num_generations
    if args.max_completion_length is not None:
        overrides["max_completion_length"] = args.max_completion_length
    if args.max_prompt_length is not None:
        overrides["max_prompt_length"] = args.max_prompt_length
    if overrides:
        grpo_config = replace(grpo_config, **overrides)

    log.info("Generating synthetic hindsight samples (no LLM call) ...")
    hindsight_samples = _build_synthetic_hindsight(papers, episode_config)
    log.info(
        "Built %d synthetic hindsight samples across episodes.", len(hindsight_samples)
    )

    grpo_dir = output_dir / "realization_grpo"
    log.info("Launching pipeline with reward_mode=%s ...", args.reward_mode)
    manifest = run_policy_rl_pipeline(
        papers,
        model_name=resolve_small_model(args.model).model_id,
        output_dir=str(grpo_dir),
        episode_config=episode_config,
        candidate_config=candidate_config,
        reward_config=reward_config,
        reward_config_path=reward_config_path,
        selection_config=selection_config,
        trainer_config=grpo_config,
        trainer_config_path="grpo_train.yaml",
        selection_config_path="selection.yaml",
        max_episodes=args.max_episodes,
        init_policy_path=None,
        skip_alignment_check=args.skip_alignment_check,
        hindsight_samples=hindsight_samples,
        max_grpo_rows=args.max_grpo_rows,
        use_vllm_server=args.use_vllm_server,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
    )
    log.info("Done. Manifest: %s", grpo_dir / "pipeline_manifest.json")
    log.info("  checkpoint: %s", manifest.get("trainer_policy_manifest_path", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
