from __future__ import annotations

import numpy as np

from app.geometry.fit import Choice, Embeddings


class ExemplarModel:
    def __init__(self, pos: np.ndarray, neg: np.ndarray):
        self.pos = pos
        self.neg = neg

    def score(self, e: np.ndarray) -> float:
        """Mean cosine to chosen exemplars minus mean cosine to rejected exemplars."""
        p = float(np.mean(self.pos @ e)) if len(self.pos) else 0.0
        n = float(np.mean(self.neg @ e)) if len(self.neg) else 0.0
        return p - n

    def prefers(self, ea: np.ndarray, eb: np.ndarray) -> bool:
        return self.score(ea) >= self.score(eb)


def fit_exemplar(choices: list[Choice], emb: Embeddings) -> ExemplarModel:
    pos = np.array([emb[c] for c, _ in choices])
    neg = np.array([emb[r] for _, r in choices])
    return ExemplarModel(pos, neg)
