#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATE_RE = re.compile(r"^\d{4}-\d{2}$")
FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
SECTION_RE = re.compile(r"^#\s+(.*?)\n(.*?)(?=\n#\s+|\Z)", re.DOTALL | re.MULTILINE)


@dataclass
class ValidationIssue:
    level: str
    file: str
    message: str


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_front_matter(raw: str) -> dict[str, Any]:
    """
    Parse a small YAML subset used by this dataset:
    - key: value
    - key:
      - item
      - item
    - quoted strings are supported and unwrapped.
    """
    parsed: dict[str, Any] = {}
    lines = raw.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        if not line.strip():
            continue
        if ":" not in line or line.startswith((" ", "\t")):
            continue

        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if val:
            parsed[key] = val.strip('"').strip("'")
            continue

        items: list[str] = []
        while i < len(lines):
            nxt = lines[i].rstrip()
            if not nxt.strip():
                i += 1
                continue
            if nxt.startswith((" ", "\t")) and nxt.strip().startswith("-"):
                item = nxt.strip()[1:].strip().strip('"').strip("'")
                if item:
                    items.append(item)
                i += 1
                continue
            break
        parsed[key] = items

    return parsed


def parse_markdown(md_path: Path) -> dict[str, Any]:
    content = md_path.read_text(encoding="utf-8")
    match = FRONT_MATTER_RE.search(content)
    if not match:
        raise ValueError("Missing YAML front matter")

    front_matter = parse_front_matter(match.group(1))

    body = content[match.end() :]
    sections: dict[str, str] = {}
    for heading, text in SECTION_RE.findall(body):
        sections[_clean_text(heading).lower()] = text.strip()

    ideas_raw = sections.get("next-stage ideas", "")
    next_stage_ideas = [
        _clean_text(line.lstrip("-"))
        for line in ideas_raw.splitlines()
        if line.strip().startswith("-")
    ]

    return {
        "paper_id": front_matter.get("paper_id"),
        "date": front_matter.get("date"),
        "track": front_matter.get("track"),
        "venue": front_matter.get("venue"),
        "title": front_matter.get("title"),
        "authors": front_matter.get("authors") or [],
        "keywords": front_matter.get("keywords") or [],
        "summary": sections.get("summary", ""),
        "method": sections.get("method", ""),
        "findings": sections.get("findings", ""),
        "limitations": sections.get("limitations", ""),
        "next_stage_ideas": next_stage_ideas,
        "source_file": str(md_path),
        "ingested_at": datetime.now(UTC).isoformat(),
    }


def validate_record(record: dict[str, Any], file_label: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required_str_fields = ["paper_id", "date", "track", "venue", "title", "summary"]

    for field in required_str_fields:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append(
                ValidationIssue(
                    "error",
                    file_label,
                    f"Missing/empty required field '{field}'",
                )
            )

    date_value = record.get("date")
    if isinstance(date_value, str) and not DATE_RE.match(date_value):
        issues.append(
            ValidationIssue("error", file_label, "Invalid date format; expected YYYY-MM")
        )

    for list_field in ["authors", "keywords", "next_stage_ideas"]:
        value = record.get(list_field)
        if not isinstance(value, list):
            issues.append(
                ValidationIssue("error", file_label, f"Field '{list_field}' must be a list")
            )
            continue
        if not all(isinstance(item, str) and item.strip() for item in value):
            issues.append(
                ValidationIssue(
                    "error",
                    file_label,
                    f"Field '{list_field}' must contain non-empty strings",
                )
            )

    for text_field in ["summary", "method", "findings", "limitations"]:
        value = record.get(text_field)
        if isinstance(value, str) and value.strip() and len(value.strip()) < 20:
            issues.append(
                ValidationIssue("warning", file_label, f"Field '{text_field}' is very short")
            )

    return issues


def write_validation_log(
    path: Path, all_issues: list[ValidationIssue], totals: dict[str, int]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Markdown Ingestion Validation Report",
        f"generated_at: {datetime.now(UTC).isoformat()}",
        f"files_scanned: {totals['files_scanned']}",
        f"valid_records: {totals['valid_records']}",
        f"invalid_records: {totals['invalid_records']}",
        f"warnings: {totals['warnings']}",
        "",
    ]

    if not all_issues:
        lines.append("No validation issues found.")
    else:
        for issue in all_issues:
            lines.append(f"[{issue.level.upper()}] {issue.file} :: {issue.message}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index markdown papers into canonical JSONL.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/arxiv_csml/raw_markdown"),
        help="Directory containing markdown files",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/arxiv_csml/papers.jsonl"),
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--validation-log",
        type=Path,
        default=Path("data/arxiv_csml/validation.log"),
        help="Path to validation log report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any record is invalid",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    md_files = sorted(
        [p for p in args.input_dir.glob("*.md") if p.name.lower() != "readme.md"]
    )
    records: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []
    totals = {
        "files_scanned": len(md_files),
        "valid_records": 0,
        "invalid_records": 0,
        "warnings": 0,
    }

    for md_file in md_files:
        label = md_file.name
        try:
            record = parse_markdown(md_file)
        except Exception as exc:  # noqa: BLE001
            issues.append(ValidationIssue("error", label, str(exc)))
            totals["invalid_records"] += 1
            continue

        record_issues = validate_record(record, label)
        issues.extend(record_issues)
        totals["warnings"] += sum(1 for issue in record_issues if issue.level == "warning")

        if any(issue.level == "error" for issue in record_issues):
            totals["invalid_records"] += 1
            continue

        records.append(record)
        totals["valid_records"] += 1

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_validation_log(args.validation_log, issues, totals)
    print(f"Scanned {totals['files_scanned']} markdown files.")
    print(f"Wrote {totals['valid_records']} valid records to {args.output_jsonl}")
    print(f"Validation log written to {args.validation_log}")

    if args.strict and totals["invalid_records"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
