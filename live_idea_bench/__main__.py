"""LiveIdeaBench unified command-line entrypoint.

This is the single front door to the repository. It groups the project's work
into a handful of subcommands and forwards all remaining arguments to the
underlying script's ``main()`` unchanged, so every flag documented on the
individual scripts keeps working.

    python -m live_idea_bench <command> [args...]

Commands
--------
Benchmark:
    benchmark        Run a domain-separated backtest of a forecasting strategy.
    judge-eval       Score saved predictions with the retrieve-then-judge LLM judge.

MDF forecaster:
    hindsight        Extract latent-innovation training labels from future papers.
    train-prior      SFT the memory-conditioned innovation prior.
    train            GRPO-train the realization policy.
    infer            Joint inference: sample from the prior -> realize -> select.
    eval             Evaluate a trained forecaster on a held-out test window.

Single-metric ablation:
    ablate           Train the soft / coverage / novelty single-metric GRPO variants.

Analysis:
    analysis         Evaluation-validity analyses (citation / coauthor / leakage).

Run ``python -m live_idea_bench <command> --help`` to see a command's own flags.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _REPO_ROOT / "examples"

# command -> (relative script path, one-line help). The script is executed as
# __main__ with sys.argv rewritten, so its existing argparse handles the flags.
_COMMANDS: dict[str, tuple[str, str]] = {
    # benchmark (examples/live-idea-bench/)
    "benchmark": (
        "live-idea-bench/run_domain_backtest.py",
        "Run a domain-separated backtest.",
    ),
    "judge-eval": (
        "live-idea-bench/llm_judge_eval.py",
        "Retrieve-then-judge LLM evaluation of predictions.",
    ),
    # MDF forecaster
    "hindsight": (
        "forecaster/run_topic_hindsight.py",
        "Extract latent-innovation training labels.",
    ),
    "train-prior": ("forecaster/run_prior_sft.py", "SFT the innovation prior."),
    "train": (
        "forecaster/run_policy_rl_training.py",
        "GRPO-train the realization policy.",
    ),
    "infer": (
        "forecaster/run_joint_inference.py",
        "Joint inference: prior -> realize -> select.",
    ),
    "eval": ("forecaster/eval.py", "Evaluate a trained forecaster."),
    # single-metric ablation
    "ablate": (
        "forecaster/train_grpo_metric.py",
        "Single-metric GRPO: soft/coverage/novelty.",
    ),
    # analysis (examples/live-idea-bench/)
    "analysis": (
        "live-idea-bench/analysis_leakage.py",
        "Evaluation-validity analyses (citation/coauthor/leakage).",
    ),
}

# `analysis` is a small family; let the user pick which one (default: leakage).
_ANALYSIS_VARIANTS = {
    "leakage": "live-idea-bench/analysis_leakage.py",
    "citation": "live-idea-bench/analysis_citation.py",
    "coauthor": "live-idea-bench/analysis_coauthor.py",
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
        # `analysis <variant> [args]`; default variant is leakage. A leading
        # non-flag token must be a known variant (reject typos rather than
        # silently running the wrong analysis).
        variant = "leakage"
        if rest and not rest[0].startswith("-"):
            if rest[0] in _ANALYSIS_VARIANTS:
                variant, rest = rest[0], rest[1:]
            else:
                print(
                    f"Unknown analysis variant {rest[0]!r}; "
                    f"choose one of: {', '.join(_ANALYSIS_VARIANTS)}",
                    file=sys.stderr,
                )
                return 2
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
