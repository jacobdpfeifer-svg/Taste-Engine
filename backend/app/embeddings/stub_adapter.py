from __future__ import annotations

import hashlib
import re

import numpy as np

from app.embeddings.base import EmbeddingAdapter

# How much id-specific noise to mix into a text-derived embedding. Small enough that stimuli
# sharing tags stay close; large enough that no two stimuli are identical.
_ID_NOISE = 0.35


def _seeded_vec(token: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).normal(size=dim)


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9-]+", text.lower()) if t]


class StubAdapter(EmbeddingAdapter):
    """Deterministic pseudo-embeddings, no external dependencies or model downloads.

    If `text` (caption/tags) is provided, the embedding is the normalized sum of per-token
    vectors plus a little id-specific noise. Stimuli that share tags therefore land near each
    other — a bag-of-words stand-in for real semantics — so a person choosing consistently by
    visible style produces a learnable signal even on the stub backend.

    Without text it falls back to a pure id hash (same id -> same vector, no semantics).
    Swap for OpenClipAdapter (EMBEDDING_BACKEND=openclip) for real image signal.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(
        self, stimulus_id: str, image_path: str | None = None, text: str | None = None
    ) -> np.ndarray:
        v = _seeded_vec(f"id:{stimulus_id}", self.dim)
        if text:
            toks = _tokens(text)
            if toks:
                sem = np.sum([_seeded_vec(f"tok:{t}", self.dim) for t in toks], axis=0)
                v = sem + _ID_NOISE * v
        n = np.linalg.norm(v)
        return v / n if n else v
