"""Local sentence embedder used as a drop-in replacement for Voyage in
the eval pipeline and as the embedding backend for cluster-coverage and
novelty rewards during GRPO training.

Selection rationale:
  * `BAAI/bge-large-en-v1.5` (1024-dim) is the strongest open en-text retriever
    in the 0.3-1.5B size range, fits on a single A6000 alongside training
    workloads, and produces vectors whose cosine geometry matches what the
    KMeans / novelty code in `examples/llm_judge_eval.py` already expects.

Caching is process-local: callers passing identical texts (same hash) skip
re-encoding. This matters for the soft-reward path where each prediction is
embedded once per training step and reused across candidate retrieval.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Iterable
from typing import Any

import numpy as np

_DEFAULT_MODEL = os.environ.get("LIB_LOCAL_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
_DEFAULT_DEVICE = os.environ.get("LIB_LOCAL_EMBED_DEVICE", "")
_DEFAULT_BATCH = int(os.environ.get("LIB_LOCAL_EMBED_BATCH", "64"))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class LocalEmbedder:
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = _DEFAULT_BATCH,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        if device is None or device == "":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        # `sentence_transformers` is an optional, unstubbed dependency imported
        # lazily in `_ensure_model`, so the model handle is only ever `Any`.
        self._model: Any = None
        self._lock = threading.Lock()
        self._cache: dict[str, list[float]] = {}

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = 512

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = [str(t or "") for t in texts]
        if not text_list:
            return []

        # Cache lookup
        keys = [_hash_text(t) for t in text_list]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._cache]

        if missing_idx:
            self._ensure_model()
            to_encode = [text_list[i] for i in missing_idx]
            vecs = self._model.encode(
                to_encode,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vecs = np.asarray(vecs, dtype=np.float32)
            for j, i in enumerate(missing_idx):
                self._cache[keys[i]] = vecs[j].tolist()

        return [self._cache[k] for k in keys]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


_DEFAULT: LocalEmbedder | None = None
_DEFAULT_LOCK = threading.Lock()


def get_default_embedder() -> LocalEmbedder:
    """Process-wide default embedder. Lazily instantiated."""
    global _DEFAULT
    if _DEFAULT is not None:
        return _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = LocalEmbedder()
    return _DEFAULT
