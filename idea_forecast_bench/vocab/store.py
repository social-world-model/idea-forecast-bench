"""Append-only JSONL store of ConceptRecords, one directory per extraction
fingerprint. Records are keyed by paper id with last-write-wins, so a rerun
resumes where the previous one stopped and never edits a line in place."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from idea_forecast_bench.atomic import atomic_write_text
from idea_forecast_bench.vocab.extract import dumps_record, record_from_dict
from idea_forecast_bench.vocab.types import ConceptRecord

MANIFEST_NAME = "manifest.json"
RECORDS_NAME = "records.jsonl"
VECTORS_NAME = "vectors"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ConceptStore:
    """``<root>/<fingerprint>/records.jsonl`` plus a manifest describing the
    prompt and model that produced it. Mixing fingerprints in one directory
    is refused: two prompts produce two vocabularies, not one."""

    def __init__(self, root: Path, fingerprint: str) -> None:
        self.dir = Path(root) / fingerprint
        self.fingerprint = fingerprint
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / VECTORS_NAME).mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._path = self.dir / RECORDS_NAME

    @property
    def vectors_dir(self) -> Path:
        return self.dir / VECTORS_NAME

    def write_manifest(self, payload: Mapping[str, Any]) -> None:
        manifest_path = self.dir / MANIFEST_NAME
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") not in (None, self.fingerprint):
                raise ValueError(
                    f"{self.dir} was written under fingerprint "
                    f"{existing.get('fingerprint')!r}, not {self.fingerprint!r}"
                )
        merged = {
            **payload,
            "fingerprint": self.fingerprint,
            "updated_at": _utc_now(),
        }
        atomic_write_text(manifest_path, json.dumps(merged, indent=2, sort_keys=True))

    def load(self) -> dict[str, ConceptRecord]:
        records: dict[str, ConceptRecord] = {}
        if not self._path.exists():
            return records
        with open(self._path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record = record_from_dict(raw) if isinstance(raw, dict) else None
                if record is not None:
                    records[record.paper_id] = record
        return records

    def append(self, records: Iterable[ConceptRecord]) -> int:
        lines = [dumps_record(r) for r in records]
        if not lines:
            return 0
        with self._lock, open(self._path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return len(lines)
