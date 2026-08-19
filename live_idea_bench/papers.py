from __future__ import annotations

import calendar
import json
import re
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import IO, Any

from live_idea_bench.models import PaperRecord

PathLike = str | Path


def find_markdown_files(root: Path) -> list[Path]:
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


def load_json(path: Path) -> dict[str, Any]:
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
    results: dict[str, list[str]],
    start_yymm: str,
    end_yymm: str,
) -> dict[str, list[str]]:
    filtered = {}
    for file_path, keywords in results.items():
        filename = Path(file_path).name
        if filename[:4].isdigit():
            date_prefix = filename[:4]
            if start_yymm <= date_prefix <= end_yymm:
                filtered[file_path] = keywords
    return filtered


def group_by_keywords(
    results: dict[str, list[str]],
    target_categories: list[str] | None = None,
    fuzzy_threshold: float = 0.6,
    min_papers: int = 1,
) -> dict[str, list[str]]:
    import difflib
    from collections import defaultdict

    keyword_map = defaultdict(set)
    targets_lower = (
        [item.lower() for item in target_categories] if target_categories else []
    )

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
        keyword: sorted(paths)
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


def to_yymm(month: str) -> str:
    """Convert a YYYY-MM month string to the 4-char arXiv YYMM form (e.g. 2401)."""
    normalized = normalize_month(month)
    year_str, month_str = normalized.split("-", maxsplit=1)
    return f"{int(year_str) % 100:02d}{month_str}"


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


def _extract_title_and_body(text: str, path: Path) -> tuple[str, str]:
    match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if not match:
        return path.stem, text
    title = match.group(1).strip()
    return title, text[match.end() :].lstrip()


def _normalize_metadata_key(raw: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    if key == "paperid":
        return "paper_id"
    return key


def _extract_preamble_metadata(body: str) -> tuple[dict[str, object], str]:
    lines = body.splitlines()
    metadata: dict[str, object] = {}
    consumed = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            consumed = idx + 1
            continue
        if re.match(r"^#{1,6}\s+\S", stripped):
            break
        match = re.match(r"^([A-Za-z][A-Za-z0-9 _/-]{0,60}):\s*(.+?)\s*$", stripped)
        if not match:
            break
        metadata[_normalize_metadata_key(match.group(1))] = match.group(2).strip()
        consumed = idx + 1

    return metadata, "\n".join(lines[consumed:]).lstrip()


def _infer_month_from_parent_dirs(path: Path) -> str:
    for parent in path.parents:
        name = parent.name.strip()
        if re.match(r"^\d{4}-\d{2}$", name):
            return normalize_month(name)
    raise ValueError(f"Cannot determine month for {path}")


def _extract_published_date(metadata: dict[str, object], path: Path) -> str:
    raw_date = metadata.get("date")
    date_text = _to_date_text(raw_date)
    if date_text:
        return normalize_date(date_text)

    stem = path.stem
    if len(stem) >= 4 and stem[:4].isdigit():
        return month_end_date(normalize_month(stem[:4]))

    return month_end_date(_infer_month_from_parent_dirs(path))


def _extract_keywords(metadata: dict[str, object]) -> list[str]:
    raw = metadata.get("keywords")
    if isinstance(raw, list):
        values = [str(v).strip().lower() for v in raw if str(v).strip()]
        return sorted(set(values))
    if isinstance(raw, str):
        values = [v.strip().lower() for v in raw.split(",") if v.strip()]
        return sorted(set(values))
    return []


def _normalize_metadata_value(value: object) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            str(key): _normalize_metadata_value(inner_value)
            for key, inner_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_metadata_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


_TITLE_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "for",
        "in",
        "on",
        "with",
        "and",
        "or",
        "to",
        "is",
        "by",
        "from",
        "at",
        "its",
        "via",
        "are",
        "we",
        "our",
        "can",
        "be",
        "has",
        "have",
        "this",
        "that",
        "it",
        "not",
        "but",
        "as",
        "do",
        "how",
        "what",
        "which",
        "into",
        "over",
        "new",
        "more",
        "than",
    }
)


def _keywords_from_title(title: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]+", title)
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        low = token.lower()
        if low in _TITLE_STOP_WORDS or len(low) < 3 or low in seen:
            continue
        seen.add(low)
        keywords.append(low)
    return keywords


def _split_preface_and_sections(body: str) -> tuple[str, str]:
    match = re.search(r"^#{1,6}\s+\S.+$", body, flags=re.MULTILINE)
    if not match:
        return body.strip(), ""
    return body[: match.start()].strip(), body[match.start() :].strip()


def _clean_summary_text(text: str, *, max_chars: int = 1500) -> str:
    normalized = re.sub(r"[ \t]+", " ", text).strip()
    return normalized[:max_chars].strip()


def _extract_summary(body: str) -> str:
    summary = _extract_section(body, "Summary")
    if summary:
        return _clean_summary_text(summary)

    abstract = _extract_section(body, "Abstract")
    if abstract:
        return _clean_summary_text(abstract)

    preface, _sections = _split_preface_and_sections(body)
    if preface:
        paragraphs = [
            part.strip() for part in re.split(r"\n\s*\n", preface) if part.strip()
        ]
        for index, paragraph in enumerate(paragraphs):
            match = re.match(
                r"^Abstract\s*[—–:-]\s*(.*)$",
                paragraph,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not match:
                continue
            abstract_parts = [match.group(1).strip()] + paragraphs[index + 1 :]
            return _clean_summary_text(
                "\n\n".join(part for part in abstract_parts if part)
            )
        return _clean_summary_text(preface)

    return _clean_summary_text(clean_paper_content(body))


def _is_reference_entry_start(line: str) -> bool:
    return bool(re.match(r"^(?:\[\d+\]|\d+[.)])\s+", line))


def _strip_reference_prefix(line: str) -> str:
    return re.sub(r"^(?:\[\d+\]|\d+[.)])\s+", "", line).strip()


def _extract_bibliography(body: str) -> list[dict[str, Any]]:
    references_text = _extract_section(body, "References")
    if not references_text:
        return []

    entries: list[str] = []
    current_parts: list[str] = []
    saw_blank = False
    for raw_line in references_text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            saw_blank = True
            continue
        if _is_reference_entry_start(line):
            if current_parts:
                entries.append(" ".join(current_parts).strip())
            current_parts = [_strip_reference_prefix(line)]
            saw_blank = False
            continue
        if saw_blank and current_parts:
            entries.append(" ".join(current_parts).strip())
            current_parts = [line]
            saw_blank = False
            continue
        if current_parts:
            current_parts.append(line)
        else:
            current_parts = [line]
        saw_blank = False

    if current_parts:
        entries.append(" ".join(current_parts).strip())

    return [{"text": entry} for entry in entries if entry]


def parse_markdown_paper(path: Path) -> PaperRecord | None:
    text = read_file_content(path)
    if not text.strip():
        return None

    title, body = _extract_title_and_body(text, path)
    metadata, body = _extract_preamble_metadata(body)
    try:
        published_date = _extract_published_date(metadata, path)
    except ValueError:
        return None
    month = normalize_month(published_date)

    summary = _extract_summary(body)

    paper_id = str(metadata.get("paper_id") or path.stem)
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
        metadata={
            str(k): _normalize_metadata_value(v)
            for k, v in metadata.items()
            if k not in {"paper_id", "date", "keywords"}
        },
        references=_extract_bibliography(body),
        citations=[],
    )


def _arxiv_dir_in_range(name: str, start_idx: int | None, end_idx: int | None) -> bool:
    """Fast pre-filter: arxiv IDs start with YYMM (e.g. 2401 = 2024-01).

    Returns True if the directory name could contain papers within [start_idx, end_idx].
    Returns True when uncertain (non-standard naming) to avoid false negatives.
    """
    if len(name) < 4 or not name[:4].isdigit():
        return True  # can't tell — let it through
    yymm = name[:4]
    yy, mm = int(yymm[:2]), int(yymm[2:])
    if mm < 1 or mm > 12:
        return True
    year = 2000 + yy
    month_str = f"{year:04d}-{mm:02d}"
    try:
        idx = month_to_index(month_str)
    except Exception:
        return True
    if start_idx is not None and idx < start_idx:
        return False
    return not (end_idx is not None and idx > end_idx)


def _parse_and_filter(
    file_path: Path,
    start_idx: int | None,
    end_idx: int | None,
) -> PaperRecord | None:
    """Parse a single markdown file and apply month filter. Thread-safe."""
    if file_path.name.lower() == "readme.md":
        return None
    paper = parse_markdown_paper(file_path)
    if not paper:
        return None
    idx = month_to_index(paper.month)
    if start_idx is not None and idx < start_idx:
        return None
    if end_idx is not None and idx > end_idx:
        return None
    # Strip heavy fields not used by evidence retrieval or GRPO reward
    paper.references = []
    paper.citations = []
    paper.metadata = {}
    return paper


def _default_workers() -> int:
    """Use all available CPU cores, capped at 32 to avoid fd exhaustion."""
    import os

    return min(os.cpu_count() or 4, 32)


def _discover_files_for_dir(args: tuple[str, str]) -> list[Path]:
    """Discover .md files in a single paper directory. Thread-safe."""
    child_path, child_name = args
    child = Path(child_path)
    # Fast: try known structure {id}/auto/{id}.md first
    direct = child / "auto" / f"{child_name}.md"
    if direct.is_file():
        return [direct]
    # Fallback: rglob for non-standard layouts
    return [p for p in child.rglob("*.md") if p.name.lower() != "readme.md"]


def _cache_path_for(
    input_dir: Path, start_month: str | None, end_month: str | None
) -> Path:
    """Return a deterministic pickle cache path for the given query."""
    import hashlib

    key = f"{Path(input_dir).resolve()}|{start_month}|{end_month}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    cache_dir = Path(input_dir) / ".paper_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"papers_{h}.pkl"


def load_papers_from_markdown(
    input_dir: Path,
    start_month: str | None = None,
    end_month: str | None = None,
    *,
    workers: int | None = None,
    use_cache: bool = True,
) -> list[PaperRecord]:
    import os
    import pickle
    from concurrent.futures import ThreadPoolExecutor

    # --- Cache layer: load from pickle if available ---
    cache_file = _cache_path_for(input_dir, start_month, end_month)
    if use_cache and cache_file.is_file():
        try:
            # Declared up front: the same name is rebound to a writer below.
            f: IO[bytes]
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, list) and cached:
                print(
                    f"[papers] Loaded {len(cached)} papers from cache ({cache_file.name})"
                )
                return cached
        except Exception:
            pass  # stale/corrupt cache — rebuild

    if workers is None:
        workers = _default_workers()

    start_idx = month_to_index(start_month) if start_month else None
    end_idx = month_to_index(end_month) if end_month else None

    # Fast path: single os.listdir + prefix set filter — avoids slow iterdir/glob
    # on 239k-entry directories. Then parallel file discovery + parsing.
    if start_idx is not None or end_idx is not None:
        input_dir = Path(input_dir)
        s_idx = start_idx if start_idx is not None else 0
        # month_to_index is year*12 + month, so 2024-01 is 24288 -- a 9999
        # sentinel made range(s_idx, 9999 + 1) empty for every real date and
        # silently returned zero papers whenever end_month was omitted.
        e_idx = end_idx if end_idx is not None else month_to_index("2100-12")
        # Build YYMM prefix set for O(1) lookup
        prefixes: set[str] = set()
        for idx in range(s_idx, e_idx + 1):
            yy = (idx // 12) % 100
            mm = (idx % 12) + 1
            prefixes.add(f"{yy:02d}{mm:02d}")
        # Single listdir (1-2s for 239k entries) + in-memory filter (~0.02s).
        # Accept both the legacy arXiv "YYMM" prefix layout (e.g. "2603") and the
        # canonical "YYYY-MM" month-directory layout (e.g. "2026-03").
        all_names = os.listdir(input_dir)
        dir_args: list[tuple[str, str]] = []
        for name in all_names:
            keep = len(name) >= 4 and name[:4] in prefixes
            if not keep and re.match(r"^\d{4}-\d{2}$", name):
                try:
                    keep = s_idx <= month_to_index(name) <= e_idx
                except ValueError:
                    keep = False
            if keep:
                dir_args.append((str(input_dir / name), name))
        print(
            f"[papers] Found {len(dir_args)} dirs, discovering .md files ({workers} workers)..."
        )
        effective_disc = min(max(1, workers), len(dir_args)) if dir_args else 1
        if effective_disc <= 1:
            nested = [_discover_files_for_dir(a) for a in dir_args]
        else:
            with ThreadPoolExecutor(max_workers=effective_disc) as pool:
                nested = list(pool.map(_discover_files_for_dir, dir_args))
        files: list[Path] = []
        for batch in nested:
            files.extend(batch)
    else:
        files = find_markdown_files(input_dir)

    print(f"[papers] Parsing {len(files)} files ({workers} workers)...")
    effective_workers = min(max(1, workers), len(files)) if files else 1
    if effective_workers <= 1:
        records = [_parse_and_filter(f, start_idx, end_idx) for f in files]
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            records = list(
                pool.map(
                    lambda f: _parse_and_filter(f, start_idx, end_idx),
                    files,
                )
            )

    results = [r for r in records if r is not None]
    results.sort(
        key=lambda p: (date_to_ordinal(get_paper_published_date(p)), p.paper_id)
    )

    # --- Save to cache ---
    if use_cache and results:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[papers] Cached {len(results)} papers to {cache_file.name}")
        except Exception:
            pass  # non-fatal

    return results


def get_paper_published_date(paper: PaperRecord) -> str:
    raw = str(getattr(paper, "published_date", "") or "").strip()
    if raw:
        return normalize_date(raw)
    return month_end_date(paper.month)
