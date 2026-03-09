import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from live_idea_bench.papers import load_papers_from_markdown  # noqa: E402
from live_idea_bench.rl import (  # noqa: E402
    load_candidate_generation_config,
    load_dpo_train_config,
    load_episode_build_config,
    load_grpo_train_config,
    load_reward_config,
)
from live_idea_bench.rl.model_zoo import list_small_model_payloads, resolve_small_model  # noqa: E402
from live_idea_bench.rl.pipeline import run_policy_rl_pipeline  # noqa: E402


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and train RL policy checkpoints for LiveIdeaBench.")
    parser.add_argument("--input-dir", type=str, default="data/arxiv_csml/raw_markdown", help="Directory with markdown papers.")
    parser.add_argument("--output-dir", type=str, default="data/rl_runs/policy_rl", help="Directory for RL artifacts.")
    parser.add_argument("--model-name", type=str, help="Hugging Face model id or local checkpoint path.")
    parser.add_argument("--model-preset", type=str, help="Shortcut alias from the built-in 3B/4B model registry.")
    parser.add_argument("--stage", type=str, default="prepare", choices=["prepare", "dpo", "grpo", "both"], help="Pipeline stage to run.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test", "all"], help="Episode split to use.")
    parser.add_argument("--max-episodes", type=int, help="Optional cap for quick experiments.")
    parser.add_argument("--start-month", type=str, help="Optional lower bound month for loading papers.")
    parser.add_argument("--end-month", type=str, help="Optional upper bound month for loading papers.")
    parser.add_argument("--episode-config", type=str, default="episode_build.yaml", help="RL episode config file under config/rl.")
    parser.add_argument("--candidate-config", type=str, default="candidate_generation.yaml", help="Candidate generation config file under config/rl.")
    parser.add_argument("--reward-config", type=str, default="reward.yaml", help="Reward config file under config/rl.")
    parser.add_argument("--dpo-config", type=str, default="dpo_train.yaml", help="DPO config file under config/rl.")
    parser.add_argument("--grpo-config", type=str, default="grpo_train.yaml", help="GRPO config file under config/rl.")
    parser.add_argument("--similarity-config", type=str, default="similarity.yaml", help="Similarity config used for reward evaluation.")
    parser.add_argument("--list-model-presets", action="store_true", help="Print the built-in small-model candidates and exit.")
    return parser


def _resolve_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return str(args.model_name)
    if args.model_preset:
        return resolve_small_model(str(args.model_preset)).model_id
    raise ValueError("Either --model-name or --model-preset is required unless --list-model-presets is used.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_model_presets:
        print(json.dumps(list_small_model_payloads(), indent=2, ensure_ascii=False))
        return 0

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
    dpo_config = load_dpo_train_config(args.dpo_config)
    grpo_config = load_grpo_train_config(args.grpo_config)

    if args.start_month:
        episode_config.start_month = args.start_month
    if args.end_month:
        episode_config.end_month = args.end_month

    manifest = run_policy_rl_pipeline(
        papers,
        model_name=_resolve_model_name(args),
        output_dir=args.output_dir,
        episode_config=episode_config,
        candidate_config=candidate_config,
        reward_config=reward_config,
        dpo_config=dpo_config,
        grpo_config=grpo_config,
        stage=args.stage,
        split=args.split,
        max_episodes=args.max_episodes,
        similarity_config_path=args.similarity_config,
    )
    manifest_path = Path(args.output_dir)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    _write_json(manifest_path / "run_summary.json", manifest)

    print(f"RL pipeline stage '{args.stage}' finished for {manifest['selected_episode_count']} episodes.")
    print(f"Artifacts saved to {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
