"""LiveIdeaBench unified command-line entrypoint.

This is the single front door to the repository. It maps the paper's four pieces
to a handful of subcommands and forwards all remaining arguments to the underlying
script's ``main()`` unchanged, so every flag documented on the individual scripts
keeps working.

    python -m live_idea_bench <command> [args...]

Commands
--------
Benchmark (paper §3 — LiveIdeaBench):
    benchmark        Run a domain-separated backtest of a forecasting strategy.
    judge-eval       Score saved predictions with the retrieve-then-judge LLM judge.

MDF forecaster — main experiment (paper §4):
    hindsight        Extract latent-innovation training labels from future papers.
    train-prior      SFT the memory-conditioned innovation prior.
    train            GRPO-train the realization policy.
    infer            Joint inference (Algorithm 1): prior -> realization -> select.
    eval             Evaluate a trained forecaster on a held-out test window.

Single-metric GRPO (paper §4.3 ablation):
    ablate           Train the soft / coverage / novelty single-metric GRPO variants.

Supplementary analysis:
    analysis         Evaluation-validity analyses (citation / coauthor / leakage).

Run ``python -m live_idea_bench <command> --help`` to see a command's own flags.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _REPO_ROOT / "examples"

# command -> (relative script path, one-line help). The script is executed as
# __main__ with sys.argv rewritten, so its existing argparse handles the flags.
_COMMANDS: dict[str, tuple[str, str]] = {
    # §3 benchmark
    "benchmark": ("benchmark/run_domain_backtest.py", "Run a domain-separated backtest (paper §3)."),
    "judge-eval": ("benchmark/llm_judge_eval.py", "Retrieve-then-judge LLM evaluation of predictions (§3)."),
    # §4 MDF forecaster
    "hindsight": ("forecaster/run_topic_hindsight.py", "Extract latent-innovation training labels (§4)."),
    "train-prior": ("forecaster/run_prior_sft.py", "SFT the innovation prior (§4)."),
    "train": ("forecaster/run_policy_rl_training.py", "GRPO-train the realization policy (§4)."),
    "infer": ("forecaster/run_joint_inference.py", "Joint inference / Algorithm 1 (§4)."),
    "eval": ("forecaster/eval.py", "Evaluate a trained forecaster (§4)."),
    # §4.3 ablation
    "ablate": ("forecaster/train_grpo_metric.py", "Single-metric GRPO: soft/coverage/novelty (§4.3)."),
    # supplementary
    "analysis": ("analysis/analysis_leakage.py", "Evaluation-validity analyses (citation/coauthor/leakage)."),
}

# `analysis` is a small family; let the user pick which one (default: leakage).
_ANALYSIS_VARIANTS = {
    "leakage": "analysis/analysis_leakage.py",
    "citation": "analysis/analysis_citation.py",
    "coauthor": "analysis/analysis_coauthor.py",
}


def _print_overview() -> None:
    print(__doc__.strip())


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_overview()
        return 0

    command, rest = argv[0], argv[1:]

    if command not in _COMMANDS:
        print(f"Unknown command: {command!r}\n", file=sys.stderr)
        _print_overview()
        return 2

    if command == "analysis":
        # `analysis <variant> [args]`; default variant is leakage.
        variant = "leakage"
        if rest and rest[0] in _ANALYSIS_VARIANTS:
            variant, rest = rest[0], rest[1:]
        script_rel = _ANALYSIS_VARIANTS[variant]
    else:
        script_rel = _COMMANDS[command][0]

    script_path = _EXAMPLES / script_rel
    if not script_path.exists():
        print(f"Entry script not found: {script_path}", file=sys.stderr)
        return 1

    # Forward remaining args to the target script's own argparse, and present a
    # sensible program name in its --help output.
    sys.argv = [f"python -m live_idea_bench {command}", *rest]
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
