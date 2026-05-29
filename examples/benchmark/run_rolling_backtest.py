from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from live_idea_bench.backtest import BacktestRunner, generate_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run backtests over rolling time windows and persist resumable artifacts. "
            "Use placeholders in --command-template: {window_start}, {window_end}, "
            "{window_start_yymm}, {window_end_yymm}, {window_id}, {window_index}, "
            "{artifacts_dir}, {window_dir}."
        )
    )
    parser.add_argument("--start", required=True, help="Start month, format YYYY-MM or YYMM.")
    parser.add_argument("--end", required=True, help="End month, format YYYY-MM or YYMM.")
    parser.add_argument(
        "--window-months",
        type=int,
        default=3,
        help="Window size in months (default: 3).",
    )
    parser.add_argument(
        "--step-months",
        type=int,
        default=1,
        help="Step size in months between window starts (default: 1).",
    )
    parser.add_argument(
        "--command-template",
        required=True,
        help="Command template to execute for each window.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="data/backtests/default_run",
        help="Artifact directory for state, manifest, and per-window outputs.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Optional cap on number of generated windows (0 means all).",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing state.json (default: true).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing state and start fresh.",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="When resuming, rerun windows previously marked failed.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop execution after first failed window.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate artifacts and commands without executing subprocesses.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    windows = generate_windows(
        start=args.start,
        end=args.end,
        window_months=args.window_months,
        step_months=args.step_months,
    )
    if args.max_windows > 0:
        windows = windows[: args.max_windows]

    artifacts_dir = Path(args.artifacts_dir)
    runner = BacktestRunner(
        artifacts_dir=artifacts_dir,
        command_template=args.command_template,
        resume=args.resume,
        rerun_failed=args.rerun_failed,
        stop_on_error=args.stop_on_error,
        dry_run=args.dry_run,
    )

    print(
        f"Running backtest: {len(windows)} windows, artifacts='{artifacts_dir}', "
        f"resume={args.resume}, dry_run={args.dry_run}"
    )
    state = runner.run(windows)

    completed = sum(1 for v in state["windows"].values() if v.get("status") == "completed")
    failed = sum(1 for v in state["windows"].values() if v.get("status") == "failed")
    print(
        f"Finished. completed={completed}, failed={failed}, "
        f"state='{artifacts_dir / 'state.json'}'"
    )


if __name__ == "__main__":
    main()
