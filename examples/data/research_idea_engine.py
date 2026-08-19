import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from live_idea_bench.backtest import (  # noqa: E402
    BacktestConfig,
    backtest,
    generate,
    load_papers_from_markdown,
)
from live_idea_bench.strategy import create_strategy  # noqa: E402


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strategy-driven framework for idea generation and backtesting.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/arxiv_csml/raw_markdown",
        help="Directory with markdown papers.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="keyword_trend",
        help="Strategy name (keyword_trend, predictor_llm, or policy_rl).",
    )
    parser.add_argument(
        "--recent-months", type=int, default=3, help="Recent window used by strategy."
    )
    parser.add_argument(
        "--min-keyword-freq", type=int, default=2, help="Minimum keyword frequency."
    )
    parser.add_argument(
        "--model-name", type=str, help="Optional model override for predictor_llm."
    )
    parser.add_argument(
        "--predictor-config",
        type=str,
        default="predictor.yaml",
        help="Predictor prompt config.",
    )
    parser.add_argument(
        "--similarity-config",
        type=str,
        default="similarity.yaml",
        help="Similarity prompt config.",
    )
    parser.add_argument(
        "--temperature", type=float, help="Optional temperature override."
    )
    parser.add_argument(
        "--policy-manifest-path",
        type=str,
        help="Optional RL policy manifest for policy_rl.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of ideas to generate."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_generate = subparsers.add_parser(
        "generate", help="Generate ideas at one cutoff month."
    )
    p_generate.add_argument(
        "--cutoff-month", type=str, required=True, help="Cutoff month in YYYY-MM."
    )
    p_generate.add_argument(
        "--start-month", type=str, help="Optional lower bound month for loading papers."
    )
    p_generate.add_argument(
        "--end-month", type=str, help="Optional upper bound month for loading papers."
    )
    p_generate.add_argument(
        "--output",
        type=str,
        default="data/arxiv_csml/strategy_generation.json",
        help="Output JSON path.",
    )

    p_backtest = subparsers.add_parser(
        "backtest", help="Run rolling backtest with strategy."
    )
    p_backtest.add_argument(
        "--horizon-months", type=int, default=3, help="Future horizon length."
    )
    p_backtest.add_argument(
        "--min-train-papers",
        type=int,
        default=6,
        help="Minimum papers before a cutoff.",
    )
    p_backtest.add_argument(
        "--start-month", type=str, help="Optional lower bound month for backtest."
    )
    p_backtest.add_argument(
        "--end-month", type=str, help="Optional upper bound month for backtest."
    )
    p_backtest.add_argument(
        "--output",
        type=str,
        default="data/arxiv_csml/backtest_report.json",
        help="Output JSON path.",
    )
    return parser


def cmd_generate(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    papers = load_papers_from_markdown(
        input_dir=input_dir,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    if not papers:
        print(f"No papers loaded from {input_dir}")
        return 1

    strategy = create_strategy(
        strategy_name=args.strategy,
        recent_months=args.recent_months,
        min_keyword_freq=args.min_keyword_freq,
        model_name=args.model_name,
        predictor_config=args.predictor_config,
        similarity_config=args.similarity_config,
        temperature=args.temperature,
        policy_manifest_path=args.policy_manifest_path,
    )

    predictions = generate(
        papers=papers,
        strategy=strategy,
        cutoff_month=args.cutoff_month,
        top_k=args.top_k,
    )
    payload = {
        "mode": "generate",
        "strategy": args.strategy,
        "cutoff_month": args.cutoff_month,
        "top_k": args.top_k,
        "num_predictions": len(predictions),
        "predictions": [asdict(p) for p in predictions],
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    _write_json(output_path, payload)

    print(f"Generated {len(predictions)} ideas at cutoff {args.cutoff_month}.")
    print(f"Saved to {output_path}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    papers = load_papers_from_markdown(
        input_dir=input_dir,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    if not papers:
        print(f"No papers loaded from {input_dir}")
        return 1

    strategy = create_strategy(
        strategy_name=args.strategy,
        recent_months=args.recent_months,
        min_keyword_freq=args.min_keyword_freq,
        model_name=args.model_name,
        predictor_config=args.predictor_config,
        similarity_config=args.similarity_config,
        temperature=args.temperature,
        policy_manifest_path=args.policy_manifest_path,
    )
    config = BacktestConfig(
        top_k=args.top_k,
        horizon_months=args.horizon_months,
        min_train_papers=args.min_train_papers,
        start_month=args.start_month,
        end_month=args.end_month,
        similarity_config=args.similarity_config,
    )
    report_obj = backtest(papers=papers, strategy=strategy, config=config)
    if not isinstance(report_obj, dict):
        raise ValueError("backtest must return a mapping")
    report = cast(dict[str, object], report_obj)
    payload: dict[str, object] = {
        "mode": "backtest",
        "strategy": args.strategy,
        "config": asdict(config),
        "report": report,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    _write_json(output_path, payload)

    summary_obj = report.get("summary", {})
    summary = (
        cast(dict[str, object], summary_obj) if isinstance(summary_obj, dict) else {}
    )
    print("Backtest finished: {} windows".format(summary.get("windows", 0)))
    print(
        "avg_hit_at_k={}, avg_recall_at_k={}, avg_mrr={}".format(
            summary.get("avg_hit_at_k", 0.0),
            summary.get("avg_recall_at_k", 0.0),
            summary.get("avg_mrr", 0.0),
        )
    )
    print(f"Saved to {output_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "backtest":
        return cmd_backtest(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
