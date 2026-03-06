import calendar
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.backtest.models import PaperRecord
from src.utils import find_markdown_files, read_file_content


def normalize_month(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        raise ValueError("Empty month string")

    if re.match(r"^\d{4}-\d{2}$", value):
        return value
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value[:7]
    if re.match(r"^\d{4}$", value):
        year = 2000 + int(value[:2])
        month = int(value[2:])
        if month < 1 or month > 12:
            raise ValueError("Unsupported month format: {}".format(raw))
        return "{:04d}-{:02d}".format(year, month)
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


def month_start_date(month: str) -> str:
    normalized = normalize_month(month)
    return f"{normalized}-01"


def month_end_date(month: str) -> str:
    normalized = normalize_month(month)
    year, month_part = normalized.split("-")
    day = calendar.monthrange(int(year), int(month_part))[1]
    return f"{normalized}-{day:02d}"


def normalize_date(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        raise ValueError("Empty date string")

    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        # Validate calendar date.
        return date.fromisoformat(value).isoformat()

    if re.match(r"^\d{4}-\d{2}$", value):
        return month_end_date(value)

    if re.match(r"^\d{4}$", value):
        return month_end_date(normalize_month(value))

    raise ValueError(f"Unsupported date format: {raw}")


def date_to_ordinal(raw: str) -> int:
    return date.fromisoformat(normalize_date(raw)).toordinal()


def add_months_keep_month(raw_date: str, delta: int) -> str:
    base = date.fromisoformat(normalize_date(raw_date))
    month_index = (base.year * 12 + (base.month - 1)) + delta
    target_year = month_index // 12
    target_month = (month_index % 12) + 1
    max_day = calendar.monthrange(target_year, target_month)[1]
    target_day = min(base.day, max_day)
    return date(target_year, target_month, target_day).isoformat()


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


def _to_date_text(raw: object) -> str:
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()
    return str(raw or "").strip()


def _extract_published_date(metadata: Dict[str, object], path: Path) -> str:
    raw_date = metadata.get("date")
    date_text = _to_date_text(raw_date)
    if date_text:
        return normalize_date(date_text)

    name = path.name
    if len(name) >= 4 and name[:4].isdigit():
        return month_end_date(normalize_month(name[:4]))
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
    published_date = _extract_published_date(metadata, path)
    month = normalize_month(published_date)

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
        published_date=published_date,
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

    records.sort(key=lambda p: (date_to_ordinal(get_paper_published_date(p)), p.paper_id))
    return records


def get_paper_published_date(paper: PaperRecord) -> str:
    raw = str(getattr(paper, "published_date", "") or "").strip()
    if raw:
        return normalize_date(raw)
    return month_end_date(paper.month)
