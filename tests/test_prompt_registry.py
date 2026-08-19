from __future__ import annotations

import pytest

from live_idea_bench.config import Config, load_predictor_config, load_similarity_config


def test_load_predictor_config_happy_path() -> None:
    cfg = load_predictor_config("predictor.yaml")
    assert cfg.default_model == "gpt-4o"
    assert cfg.parser == "json"
    assert "JSON only" in cfg.user_template


def test_load_similarity_config_happy_path() -> None:
    cfg = load_similarity_config("similarity.yaml")
    # Default engine is the Voyage embedding matcher (see similarity.yaml).
    assert cfg.engine == "embedding"
    assert cfg.semantic_threshold == pytest.approx(0.5)
    # embedding_threshold must be read from YAML, not silently defaulted.
    assert cfg.embedding_threshold == pytest.approx(0.4)
    assert "similarity" in cfg.system_prompt.lower()


def test_load_predictor_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_predictor_config("missing-predictor.yaml")


def test_load_similarity_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_similarity_config("missing-similarity.yaml")


def test_config_load_config_attaches_similarity_prompt() -> None:
    runtime = Config.load_config()
    assert runtime.prompt_template is not None
    assert runtime.prompt_template.similarity_prompt.system_prompt
    assert "{idea}" in runtime.prompt_template.similarity_prompt.user_prompt_template
