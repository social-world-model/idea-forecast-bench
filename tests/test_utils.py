from pathlib import Path

from src.utils import (
    filter_by_arxiv_date,
    find_markdown_files,
    group_by_keywords,
    load_json,
    save_json,
    truncate,
)


def test_find_markdown_files_returns_sorted_files(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.md").write_text("a", encoding="utf-8")
    (nested / "ignore.txt").write_text("x", encoding="utf-8")

    files = find_markdown_files(tmp_path)

    assert [f.suffix for f in files] == [".md", ".md"]
    assert files == sorted(files)


def test_truncate_behaviour() -> None:
    assert truncate("hello", 10) == "hello"
    assert truncate("hello", 0) == "hello"
    assert truncate("abcdef", 3) == "abc\n...(truncated)"


def test_load_and_save_json_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "data" / "keywords.json"
    payload = {"paper.md": ["agents", "llm"]}

    save_json(target, payload)
    loaded = load_json(target)

    assert loaded == payload


def test_load_json_invalid_content_returns_empty_dict(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not-valid-json}", encoding="utf-8")

    assert load_json(broken) == {}


def test_filter_by_arxiv_date() -> None:
    raw = {
        "/x/2301.00001.md": ["a"],
        "/x/2312.00001.md": ["b"],
        "/x/readme.md": ["c"],
    }

    filtered = filter_by_arxiv_date(raw, "2302", "2312")

    assert "/x/2312.00001.md" in filtered
    assert "/x/2301.00001.md" not in filtered
    assert "/x/readme.md" not in filtered


def test_group_by_keywords_with_target_and_threshold() -> None:
    results = {
        "paper1.md": ["LLM agents", "optimization"],
        "paper2.md": ["llm agent", "optimization"],
        "paper3.md": ["very long" + "x" * 110],
    }

    grouped = group_by_keywords(
        results,
        target_categories=["LLM Agents", "Optimization"],
        fuzzy_threshold=0.5,
        min_papers=1,
    )

    assert "LLM Agents" in grouped
    assert grouped["LLM Agents"] == ["paper1.md", "paper2.md"]
    assert grouped["Optimization"] == ["paper1.md", "paper2.md"]


def test_group_by_keywords_without_target_respects_min_papers() -> None:
    results = {
        "paper1.md": ["A", "B"],
        "paper2.md": ["a"],
    }

    grouped = group_by_keywords(results, target_categories=None, min_papers=2)

    assert grouped == {"a": ["paper1.md", "paper2.md"]}
