import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.backtest.models import PaperRecord
from src.utils import find_markdown_files, read_file_content


def normalize_month(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        raise ValueError("Empty month string")

    if re.match(r"^\d{4}-\d{2}$", value):
        return value
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value[:7]
    if re.match(r"^\d{4}$", value):
        return "20{}-{}".format(value[:2], value[2:])
    raise ValueError("Unsupported month format: {}".format(raw))


def month_to_index(month: str) -> int:
    y, m = normalize_month(month).split("-")
    return int(y) * 12 + (int(m) - 1)


def index_to_month(index: int) -> str:
    year = index // 12
    month = (index % 12) + 1
    return "{:04d}-{:02d}".format(year, month)


def add_months(month: str, delta: int) -> str:
    return index_to_month(month_to_index(month) + delta)


def _extract_front_matter_and_body(text: str) -> Tuple[Dict[str, object], str]:
    if not text.startswith("---\n"):
        return {}, text

    lines = text.splitlines()
    end_idx = -1
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break

    if end_idx == -1:
        return {}, text

    header_text = "\n".join(lines[1:end_idx])
    body_text = "\n".join(lines[end_idx + 1 :])
    try:
        metadata = yaml.safe_load(header_text) or {}
        if not isinstance(metadata, dict):
            metadata = {}
    except Exception:
        metadata = {}
    return metadata, body_text


def _extract_section(body: str, section_name: str) -> str:
    pattern = re.compile(
        r"^#{{1,6}}\s*{}\s*$".format(re.escape(section_name)),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(body)
    if not match:
        return ""

    start = match.end()
    next_heading = re.search(r"^#{1,6}\s+.+$", body[start:], flags=re.MULTILINE)
    if next_heading:
        return body[start : start + next_heading.start()].strip()
    return body[start:].strip()


def _extract_month(metadata: Dict[str, object], path: Path) -> str:
    raw_date = metadata.get("date")
    if isinstance(raw_date, str) and raw_date.strip():
        return normalize_month(raw_date)

    name = path.name
    if len(name) >= 4 and name[:4].isdigit():
        return normalize_month(name[:4])
    raise ValueError("Cannot determine month for {}".format(path))


def _extract_keywords(metadata: Dict[str, object]) -> List[str]:
    raw = metadata.get("keywords")
    if isinstance(raw, list):
        values = [str(v).strip().lower() for v in raw if str(v).strip()]
        return sorted(set(values))
    if isinstance(raw, str):
        values = [v.strip().lower() for v in raw.split(",") if v.strip()]
        return sorted(set(values))
    return []


def _extract_title(metadata: Dict[str, object], body: str, path: Path) -> str:
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()

    first_heading = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
    if first_heading:
        return first_heading.group(1).strip()
    return path.stem


def parse_markdown_paper(path: Path) -> Optional[PaperRecord]:
    text = read_file_content(path)
    if not text.strip():
        return None

    metadata, body = _extract_front_matter_and_body(text)
    month = _extract_month(metadata, path)

    summary = _extract_section(body, "Summary")
    if not summary:
        summary = _extract_section(body, "Abstract")
    if not summary:
        summary = body[:1500].strip()

    paper_id = str(metadata.get("paper_id") or path.stem)
    title = _extract_title(metadata, body, path)
    keywords = _extract_keywords(metadata)

    return PaperRecord(
        paper_id=paper_id,
        title=title,
        month=month,
        summary=summary,
        keywords=keywords,
        source_path=str(path),
        metadata={k: str(v) for k, v in metadata.items() if k not in {"keywords"}},
    )


def load_papers_from_markdown(
    input_dir: Path,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
) -> List[PaperRecord]:
    files = find_markdown_files(input_dir)
    records: List[PaperRecord] = []

    start_idx = month_to_index(start_month) if start_month else None
    end_idx = month_to_index(end_month) if end_month else None

    for file_path in files:
        if file_path.name.lower() == "readme.md":
            continue
        paper = parse_markdown_paper(file_path)
        if not paper:
            continue

        month_idx = month_to_index(paper.month)
        if start_idx is not None and month_idx < start_idx:
            continue
        if end_idx is not None and month_idx > end_idx:
            continue
        records.append(paper)

    records.sort(key=lambda p: (month_to_index(p.month), p.paper_id))
    return records
