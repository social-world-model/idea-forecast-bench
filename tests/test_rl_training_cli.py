from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "examples" / "run_policy_rl_training.py"


def test_rl_training_cli_help_hides_split_and_exposes_prepare_split() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--prepare-split" in completed.stdout
    assert "--split" not in completed.stdout
    assert "ppo" in completed.stdout
    assert "grpo" in completed.stdout
    assert "rloo" in completed.stdout
    assert "dpo" not in completed.stdout


def test_rl_training_cli_rejects_legacy_split_flag() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--split", "train"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "--split was removed from the training CLI" in completed.stderr


def test_rl_training_cli_rejects_removed_dpo_trainer() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--trainer", "dpo"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "invalid choice" in completed.stderr
