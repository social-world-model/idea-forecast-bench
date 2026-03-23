import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from live_idea_bench.papers import load_papers_from_markdown  # noqa: E402
from forecaster.realization import (  # noqa: E402
    load_candidate_generation_config,
    load_episode_build_config,
    load_grpo_train_config,
    load_ppo_train_config,
    load_rloo_train_config,
    load_reward_config,
    load_selection_config,
)
from forecaster.realization.io import _write_json  # noqa: E402
from forecaster.realization.model_zoo import list_small_model_payloads, resolve_small_model  # noqa: E402
from forecaster.realization.pipeline import run_policy_rl_pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and train RL policy checkpoints for LiveIdeaBench.")
    parser.add_argument("--input-dir", type=str, default="data/arxiv_csml/raw_markdown", help="Directory with markdown papers.")
    parser.add_argument("--output-dir", type=str, default="data/rl_runs/policy_rl", help="Directory for RL artifacts.")
    parser.add_argument("--model-name", type=str, help="Hugging Face model id or local checkpoint path.")
    parser.add_argument("--model-preset", type=str, help="Shortcut alias from the built-in 3B/4B model registry.")
    parser.add_argument("--trainer", type=str, choices=["ppo", "grpo", "rloo"], help="Trainer algorithm to run.")
    parser.add_argument("--trainer-config", type=str, help="Optional trainer config file under config/rl.")
    parser.add_argument("--init-policy-path", type=str, help="Optional checkpoint path used to warm-start PPO, GRPO, or RLOO.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare common artifacts and trainer datasets without training.")
    parser.add_argument(
        "--prepare-split",
        type=str,
        choices=["train", "validation", "test", "all"],
        help="Episode split to materialize when used with --prepare-only.",
    )
    parser.add_argument("--skip-alignment-check", action="store_true", help="Skip the online reward alignment check for PPO/GRPO/RLOO.")
    parser.add_argument("--max-episodes", type=int, help="Optional cap for quick experiments.")
    parser.add_argument("--start-month", type=str, help="Optional lower bound month for loading papers.")
    parser.add_argument("--end-month", type=str, help="Optional upper bound month for loading papers.")
    parser.add_argument("--episode-config", type=str, default="episode_build.yaml", help="RL episode config file under config/rl.")
    parser.add_argument("--candidate-config", type=str, default="candidate_generation.yaml", help="Candidate generation config file under config/rl.")
    parser.add_argument("--reward-config", type=str, default="reward.yaml", help="Reward config file under config/rl.")
    parser.add_argument("--selection-config", type=str, default="selection.yaml", help="Selector config file under config/rl.")
    parser.add_argument("--similarity-config", type=str, default="similarity.yaml", help="Similarity config used for reward evaluation.")
    parser.add_argument("--list-model-presets", action="store_true", help="Print the built-in small-model candidates and exit.")
    return parser


_HF_MODEL_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./")


def _validate_init_policy_path(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute() or "/" in path or "\\" in path:
        resolved = p.resolve()
        if not resolved.exists():
            raise ValueError(f"--init-policy-path does not exist: {resolved}")
        return str(resolved)
    if not all(c in _HF_MODEL_ID_CHARS for c in path):
        raise ValueError(
            f"--init-policy-path {path!r} is not a valid local path or Hugging Face model ID."
        )
    return path


def _resolve_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return str(args.model_name)
    if args.model_preset:
        return resolve_small_model(str(args.model_preset)).model_id
    raise ValueError("Either --model-name or --model-preset is required unless --list-model-presets is used.")


def _resolve_trainer_config(args: argparse.Namespace) -> tuple[str, object]:
    if not args.trainer:
        raise ValueError("--trainer is required unless --list-model-presets is used.")
    explicit = str(args.trainer_config).strip() if args.trainer_config else ""
    if args.trainer == "ppo":
        path = explicit or "ppo_train.yaml"
        return path, load_ppo_train_config(path)
    if args.trainer == "grpo":
        path = explicit or "grpo_train.yaml"
        return path, load_grpo_train_config(path)
    if args.trainer == "rloo":
        path = explicit or "rloo_train.yaml"
        return path, load_rloo_train_config(path)
    raise ValueError(f"Unsupported trainer: {args.trainer}")


def main() -> int:
    parser = build_parser()
    raw_args = sys.argv[1:]
    if any(arg == "--split" or arg.startswith("--split=") for arg in raw_args):
        parser.error(
            "--split was removed from the training CLI. Training always uses the train split; "
            "use --prepare-only with --prepare-split if you need validation/test/all artifacts."
        )
    args = parser.parse_args(raw_args)

    if args.list_model_presets:
        print(json.dumps(list_small_model_payloads(), indent=2, ensure_ascii=False))
        return 0
    if args.prepare_split and not args.prepare_only:
        parser.error("--prepare-split requires --prepare-only.")
    selected_split = args.prepare_split or "train"

    input_dir = Path(args.input_dir)
    papers = load_papers_from_markdown(
        input_dir=input_dir,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    if not papers:
        print(f"No papers loaded from {input_dir}")
        return 1

    episode_config = load_episode_build_config(args.episode_config)
    candidate_config = load_candidate_generation_config(args.candidate_config)
    reward_config = load_reward_config(args.reward_config)
    selection_config = load_selection_config(args.selection_config)
    trainer_config_path, trainer_config = _resolve_trainer_config(args)

    overrides: dict[str, str] = {}
    if args.start_month:
        overrides["start_month"] = args.start_month
    if args.end_month:
        overrides["end_month"] = args.end_month
    if overrides:
        episode_config = replace(episode_config, **overrides)

    manifest = run_policy_rl_pipeline(
        papers,
        trainer=args.trainer,
        model_name=_resolve_model_name(args),
        output_dir=args.output_dir,
        episode_config=episode_config,
        candidate_config=candidate_config,
        reward_config=reward_config,
        reward_config_path=args.reward_config,
        selection_config=selection_config,
        trainer_config=trainer_config,
        trainer_config_path=trainer_config_path,
        selection_config_path=args.selection_config,
        split=selected_split,
        max_episodes=args.max_episodes,
        similarity_config_path=args.similarity_config,
        prepare_only=args.prepare_only,
        init_policy_path=_validate_init_policy_path(args.init_policy_path),
        skip_alignment_check=args.skip_alignment_check,
    )
    manifest_path = Path(args.output_dir)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    _write_json(manifest_path / "run_summary.json", manifest)

    action = "prepared" if args.prepare_only else "finished"
    print(f"RL trainer '{args.trainer}' {action} for {manifest['selected_episode_count']} episodes.")
    print(f"Artifacts saved to {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
