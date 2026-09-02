from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from idea_forecast_bench.atomic import atomic_write_text


class RunState:
    """Persists embeddings and judge decisions across runs."""

    def __init__(
        self,
        path: Path,
        *,
        judge_fingerprint: str,
        embed_fingerprint: str,
    ) -> None:
        self.path = path
        self.judge_fp = judge_fingerprint
        self.embed_fp = embed_fingerprint
        # Split-flush: embeddings live in a sidecar file because they grow to
        # ~99% of state bytes but only change during the embedding phase. The
        # main state (decisions + windows, ~30MB) flushes every 30s; the
        # sidecar (~1.5GB) flushes every 300s. Without this, a single 866MB
        # flush takes ~10s under lock and drops throughput from ~12 c/s to
        # under 2 c/s at scale.
        self.embeddings_path = path.with_name(path.stem + ".embeddings.json")
        self._lock = threading.Lock()
        self._last_flush_time = 0.0
        self._last_emb_flush_time = 0.0
        if path.exists():
            self._data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        else:
            self._data = {"version": 2}
        # Forward-compat: backfill expected keys regardless of which schema
        # version produced the loaded file.
        self._data.setdefault("version", 2)
        self._data.setdefault("paper_embeddings", {})
        self._data.setdefault("pred_embeddings", {})
        self._data.setdefault("judge_decisions", {})
        self._data.setdefault("completed_windows", [])
        self._data.setdefault("window_outputs", {})
        # Fingerprint guard: a resumed state file must have been produced under
        # the same judge config (model + rubric) and embedding model, otherwise
        # cached decisions / vectors are stale and incomparable. Fail loud
        # rather than silently serving them. A fresh file (no fingerprints key)
        # is stamped with the current fingerprints.
        stored_fp = self._data.get("fingerprints")
        if stored_fp is None:
            self._data["fingerprints"] = {
                "judge": self.judge_fp,
                "embed": self.embed_fp,
            }
        else:
            mismatches = []
            if stored_fp.get("judge") != self.judge_fp:
                mismatches.append(
                    f"judge ({stored_fp.get('judge')} != {self.judge_fp}): "
                    f"--judge-model or JUDGE_SYSTEM changed"
                )
            if stored_fp.get("embed") != self.embed_fp:
                mismatches.append(
                    f"embed ({stored_fp.get('embed')} != {self.embed_fp}): "
                    f"--embed-model changed"
                )
            if mismatches:
                raise ValueError(
                    "State file "
                    f"{path} was produced under a different config:\n  - "
                    + "\n  - ".join(mismatches)
                    + "\nUse a fresh --state-file (or delete the existing one) "
                    "so decisions/embeddings are not silently reused."
                )
        # If a sidecar exists, merge its embeddings in (authoritative when
        # both sources have the same key — sidecar is written more recently).
        if self.embeddings_path.exists():
            emb = json.loads(self.embeddings_path.read_text(encoding="utf-8"))
            self._data["paper_embeddings"].update(emb.get("paper_embeddings", {}))
            self._data["pred_embeddings"].update(emb.get("pred_embeddings", {}))

    # ---- embeddings --------------------------------------------------------
    def get_paper_vec(self, paper_id: str) -> list[float] | None:
        with self._lock:
            vec: list[float] | None = self._data["paper_embeddings"].get(
                f"{self.embed_fp}__{paper_id}"
            )
            return vec

    def set_paper_vecs(self, id_vec_pairs: list[tuple[str, list[float]]]) -> None:
        with self._lock:
            for paper_id, vec in id_vec_pairs:
                self._data["paper_embeddings"][f"{self.embed_fp}__{paper_id}"] = vec
            self._flush_embeddings()

    def get_pred_vec(self, pred_hash: str) -> list[float] | None:
        with self._lock:
            vec: list[float] | None = self._data["pred_embeddings"].get(
                f"{self.embed_fp}__{pred_hash}"
            )
            return vec

    def set_pred_vec(self, pred_hash: str, vec: list[float]) -> None:
        with self._lock:
            self._data["pred_embeddings"][f"{self.embed_fp}__{pred_hash}"] = vec
            self._flush_embeddings()

    # ---- judge decisions ---------------------------------------------------
    def get_decision(self, pred_hash: str, paper_id: str) -> dict[str, Any] | None:
        key = f"{self.judge_fp}__{pred_hash}__{paper_id}"
        decision: dict[str, Any] | None = self._data["judge_decisions"].get(key)
        return decision

    def set_decision(
        self, pred_hash: str, paper_id: str, decision: dict[str, Any]
    ) -> None:
        key = f"{self.judge_fp}__{pred_hash}__{paper_id}"
        with self._lock:
            self._data["judge_decisions"][key] = decision
            self._flush()

    # ---- window completion -------------------------------------------------
    def is_window_done(self, topic_id: str, cutoff: str) -> bool:
        return f"{topic_id}__{cutoff}" in self._data["completed_windows"]

    def get_window_output(self, topic_id: str, cutoff: str) -> dict[str, Any] | None:
        key = f"{topic_id}__{cutoff}"
        with self._lock:
            out: dict[str, Any] | None = self._data.get("window_outputs", {}).get(key)
            return out

    def mark_window_done(
        self, topic_id: str, cutoff: str, window_result: dict[str, Any]
    ) -> None:
        key = f"{topic_id}__{cutoff}"
        with self._lock:
            if key not in self._data["completed_windows"]:
                self._data["completed_windows"].append(key)
            if "window_outputs" not in self._data:
                self._data["window_outputs"] = {}
            self._data["window_outputs"][key] = window_result
            # Force a flush at window boundaries so the completed_windows
            # marker is durable for resume.
            self._flush(force=True)

    # ---- persistence -------------------------------------------------------
    def _flush(self, *, force: bool = False) -> None:
        """Must be called with self._lock held. Writes only the main state
        (decisions + windows + version, ~30MB at full scale). Embeddings
        live in a sidecar with its own throttle. Throttled to 1 write per
        30s unless force=True."""
        now = time.time()
        if not force and now - self._last_flush_time < 30:
            return
        main = {
            "version": self._data.get("version", 2),
            "fingerprints": self._data.get("fingerprints", {}),
            "judge_decisions": self._data["judge_decisions"],
            "completed_windows": self._data["completed_windows"],
            "window_outputs": self._data["window_outputs"],
        }
        atomic_write_text(self.path, json.dumps(main, ensure_ascii=False))
        self._last_flush_time = now

    def _flush_embeddings(self, *, force: bool = False) -> None:
        """Must be called with self._lock held. Writes the embeddings sidecar
        (paper_embeddings + pred_embeddings, ~1.5GB at full scale). Throttled
        to 1 write per 300s — embeddings are append-mostly and stabilize early
        in the run."""
        now = time.time()
        if not force and now - self._last_emb_flush_time < 300:
            return
        emb = {
            "paper_embeddings": self._data["paper_embeddings"],
            "pred_embeddings": self._data["pred_embeddings"],
        }
        atomic_write_text(self.embeddings_path, json.dumps(emb, ensure_ascii=False))
        self._last_emb_flush_time = now

    def force_flush(self) -> None:
        """Force-write both the main state and the embeddings sidecar."""
        with self._lock:
            self._flush(force=True)
            self._flush_embeddings(force=True)
