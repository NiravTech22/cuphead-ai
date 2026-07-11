"""knowledge/embedder.py — CPU-ONLY embedding for the knowledge base.

Deliberately separate from moment_finder's embedder (which may sit on the GPU):
the KB must add zero VRAM. all-MiniLM-L6-v2 is ~80MB and embeds a query in
tens of milliseconds on CPU.
"""
from __future__ import annotations

import functools

import numpy as np

from ..logging_setup import get_logger

log = get_logger(__name__)

DIM = 384


@functools.lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    log.info("loading KB embedder all-MiniLM-L6-v2 on CPU (no VRAM)")
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    """Return a (N, 384) float32 array of L2-normalized embeddings (CPU)."""
    vecs = _model().encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False, device="cpu")
    return vecs.astype(np.float32)


def to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
