from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from idea_forecast_bench.atomic import atomic_write_text

# Importing anything from idea_forecast_bench.judge here would pull in
# judge/__init__ -> judge.topics -> backtest -> strategy, which is a cycle
# when this module is reached through strategy/__init__. The Voyage URL is
# duplicated in similarity.py for the same reason; embed_batch is imported
# lazily inside the method.
from idea_forecast_bench.similarity import VOYAGE_BASE_URL

HASH_BACKEND = "hash"
VOYAGE_BACKEND = "voyage"


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class VoyageEmbedder:
    """Voyage embeddings through the same OpenAI-compatible client the judge
    uses, so element vectors share the judge's geometry and API key."""

    def __init__(self, model: str) -> None:
        import openai

        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "VOYAGE_API_KEY is required for the voyage embedding backend "
                "(use --embed-backend hash for an offline dry run)."
            )
        base_url = os.environ.get("VOYAGE_BASE_URL") or VOYAGE_BASE_URL
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        from idea_forecast_bench.judge.embeddings import embed_batch

        return embed_batch(list(texts), self._client, model=self._model)


class HashEmbedder:
    """Deterministic character-trigram hashing. Similar strings get similar
    vectors, which is all a dry run needs; never use it for real results."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return f"hash-{self._dim}"

    def _one(self, text: str) -> list[float]:
        padded = f"  {text.lower()}  "
        vec = [0.0] * self._dim
        for i in range(len(padded) - 2):
            gram = padded[i : i + 3]
            digest = hashlib.sha1(gram.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


def make_embedder(backend: str, model: str) -> Embedder:
    if backend == HASH_BACKEND:
        return HashEmbedder()
    if backend == VOYAGE_BACKEND:
        return VoyageEmbedder(model)
    raise ValueError(f"unknown embed backend {backend!r}; use voyage or hash")


class VectorStore:
    """Element-label vectors persisted as one JSON file per embedding model.

    Embedding a label depends only on the label text, so vectors may be
    computed over the whole corpus once without leaking any time information
    into a window."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._vectors: dict[str, list[float]] = {}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._vectors = {
                    str(k): [float(x) for x in v]
                    for k, v in raw.items()
                    if isinstance(v, list)
                }

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, key: object) -> bool:
        return key in self._vectors

    def missing(self, keys: Iterable[str]) -> list[str]:
        with self._lock:
            return sorted({k for k in keys if k not in self._vectors})

    def view(self) -> Mapping[str, Sequence[float]]:
        return MappingProxyType(self._vectors)

    def ensure(
        self,
        keys: Iterable[str],
        texts: Mapping[str, str],
        embedder: Embedder,
        *,
        batch_size: int = 512,
    ) -> int:
        """Embed every key not yet stored; returns how many were added."""
        todo = self.missing(keys)
        if not todo:
            return 0
        added = 0
        for start in range(0, len(todo), batch_size):
            chunk = todo[start : start + batch_size]
            vectors = embedder.embed([texts[k] for k in chunk])
            if len(vectors) != len(chunk):
                raise RuntimeError(
                    f"embedder returned {len(vectors)} vectors for {len(chunk)} texts"
                )
            with self._lock:
                for key, vec in zip(chunk, vectors, strict=True):
                    self._vectors[key] = [float(x) for x in vec]
            added += len(chunk)
            self.save()
        return added

    def save(self) -> None:
        with self._lock:
            payload = json.dumps(self._vectors)
        atomic_write_text(self.path, payload)
