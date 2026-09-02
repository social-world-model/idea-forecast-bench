from __future__ import annotations

import runpy
import sys
from pathlib import Path

OVERVIEW = """\
IdeaForecastBench command-line entrypoint.

    idea-forecast-bench <command> [args...]

Benchmark:
    fetch            Download an arXiv corpus the benchmark can read.
    baselines        Run every baseline on one corpus, print a comparison table.
    benchmark        Run a domain-separated backtest of a forecasting strategy.
    judge-eval       Score saved predictions with the retrieve-then-judge LLM judge.
    main-table       Assemble the main results table from judge-eval outputs.

MDF forecaster:
    hindsight        Extract latent-innovation training labels from future papers.
    train-prior      SFT the memory-conditioned innovation prior.
    train            GRPO-train the realization policy.
    infer            Joint inference: sample from the prior -> realize -> select.

Analysis:
    analysis         Evaluation-validity analyses (citation / coauthor / leakage).

Run `idea-forecast-bench <command> --help` to see a command's own flags.
"""

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _REPO_ROOT / "examples"

# command -> (relative script path, one-line help). The script is executed as
# __main__ with sys.argv rewritten, so its existing argparse handles the flags.
_COMMANDS: dict[str, tuple[str, str]] = {
    # benchmark (examples/benchmark/)
    "fetch": (
        "benchmark/fetch_papers.py",
        "Download an arXiv corpus the benchmark can read.",
    ),
    "baselines": (
        "benchmark/run_all_baselines.py",
        "Run every baseline on one corpus and print a comparison table.",
    ),
    "benchmark": (
        "benchmark/run_domain_backtest.py",
        "Run a domain-separated backtest.",
    ),
    "judge-eval": (
        "benchmark/llm_judge_eval.py",
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
    # analysis (examples/benchmark/)
    "analysis": (
        "benchmark/analysis_leakage.py",
        "Evaluation-validity analyses (citation/coauthor/leakage).",
    ),
    "main-table": (
        "benchmark/build_main_table.py",
        "Assemble the main results table from judge-eval outputs.",
    ),
}

#: Modules that live in an optional dependency group. A command that needs one
#: is not broken -- the group just is not installed -- so say which install
#: brings it in instead of printing an import traceback at someone following
#: the README.
_OPTIONAL_MODULES: dict[str, str] = {
    "torch": "forecaster",
    "transformers": "forecaster",
    "trl": "forecaster",
    "peft": "forecaster",
    "datasets": "forecaster",
    "accelerate": "forecaster",
    "sentence_transformers": "forecaster",
    "pandas": "forecaster",
    "flask": "webapp",
    "flask_cors": "webapp",
}


# `analysis` is a small family; let the user pick which one (default: leakage).
_ANALYSIS_VARIANTS = {
    "leakage": "benchmark/analysis_leakage.py",
    "citation": "benchmark/analysis_citation.py",
    "coauthor": "benchmark/analysis_coauthor.py",
}


def _print_overview() -> None:
    print(OVERVIEW.strip())


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
    sys.argv = [f"python -m idea_forecast_bench {command}", *rest]
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except ModuleNotFoundError as exc:
        # Only translate the optional-group case. Anything else is a real
        # import error and deserves its traceback.
        group = _OPTIONAL_MODULES.get((exc.name or "").split(".")[0])
        if group is None:
            raise
        print(
            f"`{command}` needs the optional '{group}' dependencies "
            f"(missing: {exc.name}).\n"
            f"Install them with:  poetry install --with {group}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
