from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from idea_forecast_bench.atomic import atomic_write_text
from idea_forecast_bench.combinatorial.types import (
    RECORD_FAILED,
    RECORD_OK,
    UNKNOWN_MOVE,
    ExtractionRecord,
)

MANIFEST_NAME = "manifest.json"
RECORDS_DIR = "records"
VECTORS_DIR = "vectors"
SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tuple_of_str(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(str(x) for x in raw if str(x).strip())


def record_from_dict(raw: Mapping[str, Any]) -> ExtractionRecord | None:
    paper_id = str(raw.get("paper_id") or "").strip()
    if not paper_id:
        return None
    status = str(raw.get("status") or RECORD_FAILED)
    if status not in (RECORD_OK, RECORD_FAILED):
        status = RECORD_FAILED
    return ExtractionRecord(
        paper_id=paper_id,
        published_date=str(raw.get("published_date") or ""),
        status=status,
        themes=_tuple_of_str(raw.get("themes")),
        domains=_tuple_of_str(raw.get("domains")),
        methods=_tuple_of_str(raw.get("methods")),
        frames=_tuple_of_str(raw.get("frames")),
        template=str(raw.get("template") or ""),
        move=str(raw.get("move") or UNKNOWN_MOVE),
        model=str(raw.get("model") or ""),
        fingerprint=str(raw.get("fingerprint") or ""),
        extracted_at=str(raw.get("extracted_at") or ""),
        error=str(raw.get("error") or ""),
    )


class RecordWriter:
    """Append-only JSONL writer. Each process owns one shard file, so
    concurrent processes never interleave; a crash can at worst leave one
    truncated final line, which the loader skips."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        self._handle = path.open("a", encoding="utf-8")
        self._pending = 0

    def append(self, record: ExtractionRecord) -> None:
        line = json.dumps(dataclasses.asdict(record), ensure_ascii=False)
        with self._lock:
            self._handle.write(line + "\n")
            self._pending += 1
            if self._pending >= 20:
                self._handle.flush()
                self._pending = 0

    def close(self) -> None:
        with self._lock:
            self._handle.flush()
            self._handle.close()


class ElementCache:
    """On-disk store of per-paper extraction records.

    Layout::

        <root>/<fingerprint>/manifest.json
        <root>/<fingerprint>/records/<pid>-<ts>.jsonl
        <root>/<fingerprint>/vectors/<embed-model>.json
    """

    def __init__(self, directory: Path, manifest: Mapping[str, Any]) -> None:
        self.directory = directory
        self.manifest = dict(manifest)

    @property
    def fingerprint(self) -> str:
        return str(self.manifest.get("fingerprint", ""))

    @property
    def records_dir(self) -> Path:
        return self.directory / RECORDS_DIR

    def vectors_path(self, embed_model: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in embed_model)
        return self.directory / VECTORS_DIR / f"{safe}.json"

    # ---- open / locate -----------------------------------------------------
    @classmethod
    def open(
        cls,
        root: Path,
        fingerprint: str,
        *,
        manifest_extra: Mapping[str, Any] | None = None,
        allow_mismatch: bool = False,
    ) -> ElementCache:
        directory = root / fingerprint
        manifest_path = directory / MANIFEST_NAME
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stored = str(manifest.get("fingerprint", ""))
            if stored != fingerprint and not allow_mismatch:
                raise ValueError(
                    f"Element cache at {directory} was produced under fingerprint "
                    f"{stored!r}, but the current prompt/model/config fingerprint is "
                    f"{fingerprint!r}. Use a fresh --cache-dir or "
                    "--allow-fingerprint-mismatch (analysis only)."
                )
            return cls(directory, manifest)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RECORDS_DIR).mkdir(exist_ok=True)
        (directory / VECTORS_DIR).mkdir(exist_ok=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "created_at": _utc_now(),
            **dict(manifest_extra or {}),
        }
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
        return cls(directory, manifest)

    @classmethod
    def locate(cls, path: Path) -> ElementCache:
        """Open an existing cache given either its directory or its root
        (a root holding exactly one fingerprint directory)."""
        if (path / MANIFEST_NAME).exists():
            manifest = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
            return cls(path, manifest)
        if path.is_dir():
            candidates = [p for p in path.iterdir() if (p / MANIFEST_NAME).exists()]
            if len(candidates) == 1:
                return cls.locate(candidates[0])
            if len(candidates) > 1:
                raise ValueError(
                    f"{path} holds several element caches; pass one of: "
                    + ", ".join(sorted(str(c) for c in candidates))
                )
        raise FileNotFoundError(
            f"No element cache found at {path} (expected {MANIFEST_NAME}). "
            "Run `idea-forecast-bench extract-elements` first."
        )

    # ---- records -----------------------------------------------------------
    def writer(self) -> RecordWriter:
        stamp = f"{os.getpid()}-{int(time.time())}"
        return RecordWriter(self.records_dir / f"{stamp}.jsonl")

    def load(self) -> dict[str, ExtractionRecord]:
        """All records, last write per paper_id wins. Malformed lines are
        skipped and counted in ``self.manifest['_skipped_lines']``."""
        records: dict[str, ExtractionRecord] = {}
        skipped = 0
        if not self.records_dir.exists():
            return records
        for shard in sorted(self.records_dir.glob("*.jsonl")):
            with shard.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    record = record_from_dict(raw) if isinstance(raw, dict) else None
                    if record is None:
                        skipped += 1
                        continue
                    records[record.paper_id] = record
        self.manifest["_skipped_lines"] = skipped
        return records

    def summary(self, records: Mapping[str, ExtractionRecord]) -> dict[str, Any]:
        ok = sum(1 for r in records.values() if r.ok)
        failed = len(records) - ok
        unknown_move = sum(
            1 for r in records.values() if r.ok and r.move == UNKNOWN_MOVE
        )
        return {
            "records": len(records),
            "ok": ok,
            "failed": failed,
            "failure_rate": round(failed / len(records), 4) if records else 0.0,
            "unknown_move_rate": round(unknown_move / ok, 4) if ok else 0.0,
            "skipped_lines": int(self.manifest.get("_skipped_lines", 0)),
        }
