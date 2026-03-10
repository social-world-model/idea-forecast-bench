from __future__ import annotations

import calendar
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import yaml

from live_idea_bench.models import PaperRecord

PathLike = Union[str, Path]


def find_markdown_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.md"))


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Read error: {path} ({exc})")
        return ""


def read_file_content(path: PathLike) -> str:
    return read_text(Path(path))


def truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(truncated)"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Load error: {path} ({exc})")
        return {}
    return payload if isinstance(payload, dict) else {}


def save_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def filter_by_arxiv_date(
    results: Dict[str, List[str]],
    start_yymm: str,
    end_yymm: str,
) -> Dict[str, List[str]]:
    filtered = {}
    for file_path, keywords in results.items():
        filename = Path(file_path).name
        if filename[:4].isdigit():
            date_prefix = filename[:4]
            if start_yymm <= date_prefix <= end_yymm:
                filtered[file_path] = keywords
    return filtered


def group_by_keywords(
    results: Dict[str, List[str]],
    target_categories: List[str] | None = None,
    fuzzy_threshold: float = 0.6,
    min_papers: int = 1,
) -> Dict[str, List[str]]:
    import difflib
    from collections import defaultdict

    keyword_map = defaultdict(set)
    targets_lower = [item.lower() for item in target_categories] if target_categories else []

    for file_path, keywords in results.items():
        for keyword in keywords:
            clean_keyword = keyword.strip()
            if not clean_keyword or len(clean_keyword) > 100:
                continue

            matched_keyword = clean_keyword
            if target_categories:
                matches = difflib.get_close_matches(
                    clean_keyword.lower(),
                    targets_lower,
                    n=1,
                    cutoff=fuzzy_threshold,
                )
                if matches:
                    idx = targets_lower.index(matches[0])
                    matched_keyword = target_categories[idx]
                else:
                    continue

            normalized_keyword = (
                matched_keyword.strip().lower()
                if not target_categories
                else matched_keyword
            )
            if normalized_keyword:
                keyword_map[normalized_keyword].add(file_path)

    return {
        keyword: sorted(list(paths))
        for keyword, paths in keyword_map.items()
        if len(paths) >= min_papers
    }


def clean_paper_content(content: str) -> str:
    ref_patterns = [
        r"^#+\s*REFERENCES\s*$",
        r"^#+\s*References\s*$",
        r"^REFERENCES\s*$",
        r"^References\s*$",
    ]
    lines = content.split("\n")
    for idx, line in enumerate(lines):
        stripped = line.strip()
        for pattern in ref_patterns:
            if re.match(pattern, stripped):
                return "\n".join(lines[:idx])
    return content


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
            raise ValueError(f"Unsupported month format: {raw}")
        return f"{year:04d}-{month:02d}"
    raise ValueError(f"Unsupported month format: {raw}")


def month_to_index(month: str) -> int:
    y, m = normalize_month(month).split("-")
    return int(y) * 12 + (int(m) - 1)


def index_to_month(index: int) -> str:
    year = index // 12
    month = (index % 12) + 1
    return f"{year:04d}-{month:02d}"


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
        return date.fromisoformat(value).isoformat()
    if re.match(r"^\d{4}-\d{2}$", value):
        return month_end_date(value)
    if re.match(r"^\d{4}$", value):
        return month_end_date(normalize_month(value))
    raise ValueError(f"Unsupported date format: {raw}")


def date_to_ordinal(raw: str) -> int:
    return date.fromisoformat(normalize_date(raw)).toordinal()


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
        rf"^#{{1,6}}\s*{re.escape(section_name)}\s*$",
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
    raise ValueError(f"Cannot determine month for {path}")


def _extract_keywords(metadata: Dict[str, object]) -> List[str]:
    raw = metadata.get("keywords")
    if isinstance(raw, list):
        values = [str(v).strip().lower() for v in raw if str(v).strip()]
        return sorted(set(values))
    if isinstance(raw, str):
        values = [v.strip().lower() for v in raw.split(",") if v.strip()]
        return sorted(set(values))
    return []


_TITLE_STOP_WORDS = frozenset({
    "a", "an", "the", "of", "for", "in", "on", "with", "and", "or", "to",
    "is", "by", "from", "at", "its", "via", "are", "we", "our", "can",
    "be", "has", "have", "this", "that", "it", "not", "but", "as", "do",
    "how", "what", "which", "into", "over", "new", "more", "than",
})


def _keywords_from_title(title: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]+", title)
    seen: set = set()
    keywords: List[str] = []
    for token in tokens:
        low = token.lower()
        if low in _TITLE_STOP_WORDS or len(low) < 3 or low in seen:
            continue
        seen.add(low)
        keywords.append(low)
    return keywords


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
    try:
        published_date = _extract_published_date(metadata, path)
    except ValueError:
        return None
    month = normalize_month(published_date)

    summary = _extract_section(body, "Summary")
    if not summary:
        summary = _extract_section(body, "Abstract")
    if not summary:
        summary = clean_paper_content(body)[:1500].strip()

    paper_id = str(metadata.get("paper_id") or path.stem)
    title = _extract_title(metadata, body, path)
    keywords = _extract_keywords(metadata)
    if not keywords:
        keywords = _keywords_from_title(title)

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
