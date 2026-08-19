from pathlib import Path
import sys

import pytest

if sys.version_info < (3, 8):
    pytestmark = pytest.mark.skip(reason="Config module requires Python 3.8+ (typing.Literal)")
else:
    from live_idea_bench.config import Config


def test_load_config_with_embedding_and_prompt(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    prompt_dir = tmp_path / "prompt"
    prompt_dir.mkdir()

    config_file.write_text(
        """
model_name: gpt-4o-mini
max_context_chars: 1000
temperature: 0.2
openai_api_key: test-key
embedding:
  model_name: all-MiniLM-L6-v2
  use_section_weights: false
""".strip(),
        encoding="utf-8",
    )

    (prompt_dir / "similarity.yaml").write_text(
        """
system_prompt: You are a judge.
user_prompt_template: "Idea: {idea}\\nContext: {context}"
""".strip(),
        encoding="utf-8",
    )

    cfg = Config.load_config(str(config_file), str(prompt_dir))

    assert cfg.model_name == "gpt-4o-mini"
    assert cfg.max_context_chars == 1000
    assert cfg.embedding.use_section_weights is False
    assert len(cfg.topics) == 5
    assert cfg.prompt_template is not None
    assert cfg.prompt_template.similarity_prompt.system_prompt == "You are a judge."


def test_load_yaml_file_raises_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        Config._load_yaml_file(str(missing), model_class=type("Dummy", (), {}))  # type: ignore[arg-type]


def test_load_config_reads_custom_topics(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    prompt_dir = tmp_path / "prompt"
    prompt_dir.mkdir()

    config_file.write_text(
        """
topics:
  - id: optimizer
    name: Optimizer
    keywords:
      - adamw
""".strip(),
        encoding="utf-8",
    )
    (prompt_dir / "similarity.yaml").write_text(
        """
system_prompt: Judge.
user_prompt_template: "Idea: {idea}"
""".strip(),
        encoding="utf-8",
    )

    cfg = Config.load_config(str(config_file), str(prompt_dir))

    assert [topic.id for topic in cfg.topics] == ["optimizer"]
    assert cfg.topics[0].keywords == ["adamw"]
