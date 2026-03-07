from __future__ import annotations

from pathlib import Path

import live_idea_bench


def test_core_package_imports() -> None:
    assert live_idea_bench.BacktestConfig is not None
    assert live_idea_bench.PredictorLLMStrategy is not None
    assert callable(live_idea_bench.generate_predictions)


def test_repo_layout_is_src_free_and_prompt_lives_in_package() -> None:
    project_root = Path(__file__).resolve().parents[1]
    assert not (project_root / "src").exists()
    assert (project_root / "live_idea_bench" / "prompt" / "predictor.yaml").exists()
    assert (project_root / "live_idea_bench" / "prompt" / "similarity.yaml").exists()
