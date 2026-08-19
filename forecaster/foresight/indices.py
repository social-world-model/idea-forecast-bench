"""Per-cutoff vector indices for the Foresight reward + grounding gate.

Two index types per training cutoff t:
  * FutureIndex  — over Y_{t+1} (papers published in (t, t+horizon])
                   used by the new reward to retrieve true-future candidates.
  * HistoryIndex — over X_{<=t} (papers available at the cutoff)
                   used by the grounding gate (does the rollout's cited
                   evidence retrieve close enough to anything that *did* exist?).

Index storage is intentionally minimal (numpy arrays + JSON metadata).
A faiss/hnsw backend can be swapped in later behind the `search()` API.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from live_idea_bench.models import PaperRecord
from forecaster.foresight.cutoffs import (
    FUTURE_WINDOW_HARD_LIMIT,
    assert_no_test_window_leakage,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- embedder protocol


class Embedder(Protocol):
    """Minimal embedder interface. `encode(texts) -> (N, D) float32 array`."""

    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local embedder for the foresight indices (SPECTER, the scientific-paper
    embedder used for the per-cutoff future/history indices)."""

    def __init__(self, model_name: str = "sentence-transformers/allenai-specter"):
        from sentence_transformers import SentenceTransformer  # local import; heavy

        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformer:{model_name}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        arr = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float32)


class HashingEmbedder:
    """Cheap deterministic fallback embedder (no models, no network).

    Only suitable for tests + smoke runs. Uses a fixed-dim hashed
    bag-of-tokens with L2 normalization. Drop in when SentenceTransformer
    weights aren't available locally.
    """

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = int(dim)
        self.seed = int(seed)
        self.name = f"hashing:{self.dim}d"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in (text or "").lower().split():
                # cheap stable hash → index
                h = hash((token, self.seed)) & 0xFFFFFFFF
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        out /= norms
        return out


# --------------------------------------------------------------------------- text helpers


def _paper_text(p: PaperRecord) -> str:
    title = (p.title or "").strip()
    abstract = (p.summary or "").strip()
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract


# --------------------------------------------------------------------------- index dataclasses


@dataclass
class _BaseIndex:
    paper_ids: tuple[str, ...]
    published_dates: tuple[str, ...]
    embeddings: np.ndarray  # (N, D) L2-normalized float32
    embedder_name: str
    cutoff_date: str
    kind: str  # "future" or "history"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_ids = len(self.paper_ids)
        n_dates = len(self.published_dates)
        if n_ids != n_dates:
            raise ValueError(
                f"index inconsistent: {n_ids} paper_ids vs {n_dates} dates"
            )
        if self.embeddings.shape[0] != n_ids:
            raise ValueError(
                f"index inconsistent: {self.embeddings.shape[0]} rows vs {n_ids} ids"
            )

    @property
    def size(self) -> int:
        return len(self.paper_ids)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Cosine-similarity top-k. Returns (paper_id, score) pairs."""
        if self.size == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        # embeddings are already L2-normalized; normalize q to be safe.
        n = float(np.linalg.norm(q))
        if n > 0.0:
            q = q / n
        scores = self.embeddings @ q
        k = min(top_k, self.size)
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.paper_ids[i], float(scores[i])) for i in top_idx]

    # ------------- persistence -------------

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta_path = p.with_suffix(".meta.json")
        np.savez(p, embeddings=self.embeddings)
        meta_path.write_text(
            json.dumps(
                {
                    "kind": self.kind,
                    "cutoff_date": self.cutoff_date,
                    "embedder": self.embedder_name,
                    "paper_ids": list(self.paper_ids),
                    "published_dates": list(self.published_dates),
                    "meta": self.meta,
                },
                indent=2,
            )
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "_BaseIndex":
        p = Path(path)
        meta = json.loads(p.with_suffix(".meta.json").read_text())
        emb = np.load(p)["embeddings"].astype(np.float32)
        return cls(
            paper_ids=tuple(meta["paper_ids"]),
            published_dates=tuple(meta["published_dates"]),
            embeddings=emb,
            embedder_name=meta["embedder"],
            cutoff_date=meta["cutoff_date"],
            kind=meta["kind"],
            meta=meta.get("meta", {}),
        )


class FutureIndex(_BaseIndex):
    """Y_{t+1} retrieval index for the new reward."""


class HistoryIndex(_BaseIndex):
    """X_{<=t} retrieval index for the grounding gate."""


# --------------------------------------------------------------------------- builders


def build_index_from_papers(
    papers: Sequence[PaperRecord],
    embedder: Embedder,
    *,
    kind: str,
    cutoff_date: str,
    meta: dict[str, Any] | None = None,
) -> _BaseIndex:
    if kind not in {"future", "history"}:
        raise ValueError(f"unsupported index kind: {kind!r}")
    if not papers:
        cls = FutureIndex if kind == "future" else HistoryIndex
        return cls(
            paper_ids=(),
            published_dates=(),
            embeddings=np.zeros((0, 1), dtype=np.float32),
            embedder_name=embedder.name,
            cutoff_date=cutoff_date,
            kind=kind,
            meta=dict(meta or {}),
        )
    texts = [_paper_text(p) for p in papers]
    emb = embedder.encode(texts)
    full_meta = dict(meta or {})
    full_meta.setdefault(
        "paper_texts",
        {p.paper_id: t for p, t in zip(papers, texts)},
    )
    cls = FutureIndex if kind == "future" else HistoryIndex
    return cls(
        paper_ids=tuple(p.paper_id for p in papers),
        published_dates=tuple(p.published_date for p in papers),
        embeddings=emb,
        embedder_name=embedder.name,
        cutoff_date=cutoff_date,
        kind=kind,
        meta=full_meta,
    )


def build_future_index(
    future_papers: Sequence[PaperRecord],
    embedder: Embedder,
    *,
    cutoff_date: str,
    assert_no_leakage: bool = True,
) -> FutureIndex:
    """Build Y_{t+1} index. Refuses to include any test-window paper."""
    if assert_no_leakage:
        assert_no_test_window_leakage(
            (p.published_date for p in future_papers if p.published_date),
            context=f"future_index@{cutoff_date}",
        )
    idx = build_index_from_papers(
        future_papers, embedder, kind="future", cutoff_date=cutoff_date,
    )
    return idx  # type: ignore[return-value]


def build_history_index(
    history_papers: Sequence[PaperRecord],
    embedder: Embedder,
    *,
    cutoff_date: str,
) -> HistoryIndex:
    """Build X_{<=t} index. Used by the grounding gate."""
    return build_index_from_papers(  # type: ignore[return-value]
        history_papers, embedder, kind="history", cutoff_date=cutoff_date,
    )


# --------------------------------------------------------------------------- bulk orchestrator


@dataclass
class CutoffIndexBundle:
    """The pair of indices for one cutoff."""

    cutoff_date: str
    future: FutureIndex
    history: HistoryIndex


def build_cutoff_indices(
    papers: Sequence[PaperRecord],
    cutoff_dates: Sequence[str],
    horizon_months: int,
    embedder: Embedder,
    *,
    save_dir: str | Path | None = None,
) -> dict[str, CutoffIndexBundle]:
    """Build future + history indices for every cutoff.

    Uses live_idea_bench.backtest.split_train_future_by_cutoff to slice
    the corpus, so the temporal semantics exactly match what the
    existing trainer dataset assumes.
    """
    from live_idea_bench.backtest import split_train_future_by_cutoff

    bundles: dict[str, CutoffIndexBundle] = {}
    save_path: Path | None = Path(save_dir) if save_dir else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    for cutoff_date in cutoff_dates:
        cutoff_month = cutoff_date[:7]
        train_papers, future_papers, _future_end_month, future_end_date = (
            split_train_future_by_cutoff(
                papers=papers,
                cutoff_month=cutoff_month,
                cutoff_date=cutoff_date,
                horizon_months=horizon_months,
            )
        )
        future_index = build_future_index(
            future_papers,
            embedder,
            cutoff_date=cutoff_date,
        )
        history_index = build_history_index(
            train_papers,
            embedder,
            cutoff_date=cutoff_date,
        )
        bundles[cutoff_date] = CutoffIndexBundle(
            cutoff_date=cutoff_date,
            future=future_index,
            history=history_index,
        )
        if save_path is not None:
            future_index.save(save_path / f"future_{cutoff_date}.npz")
            history_index.save(save_path / f"history_{cutoff_date}.npz")
        logger.info(
            "cutoff=%s future=%d history=%d (horizon=%dmo, future_end=%s)",
            cutoff_date, future_index.size, history_index.size,
            horizon_months, future_end_date,
        )
    return bundles


# --------------------------------------------------------------------------- public re-exports

__all__ = [
    "Embedder",
    "SentenceTransformerEmbedder",
    "HashingEmbedder",
    "FutureIndex",
    "HistoryIndex",
    "CutoffIndexBundle",
    "build_future_index",
    "build_history_index",
    "build_cutoff_indices",
    "build_index_from_papers",
    "FUTURE_WINDOW_HARD_LIMIT",
]
