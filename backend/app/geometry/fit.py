from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

# A choice is (chosen_id, rejected_id).
Choice = tuple[str, str]
Embeddings = dict[str, np.ndarray]


@dataclass
class TasteModel:
    """A linear taste model. `score(item) = weights . emb_item`; higher = more preferred."""

    weights: np.ndarray

    def score(self, emb: np.ndarray) -> float:
        return float(emb @ self.weights)

    def prefers(self, emb_a: np.ndarray, emb_b: np.ndarray) -> bool:
        return self.score(emb_a) >= self.score(emb_b)


def build_dataset(choices: list[Choice], emb: Embeddings) -> tuple[np.ndarray, np.ndarray]:
    """Each chosen>rejected pair becomes a difference vector with label 1, plus its
    antisymmetric counterpart with label 0 (so both classes are present for the classifier)."""
    X: list[np.ndarray] = []
    y: list[int] = []
    for chosen, rejected in choices:
        d = emb[chosen] - emb[rejected]
        X.append(d)
        y.append(1)
        X.append(-d)
        y.append(0)
    return np.asarray(X), np.asarray(y)


def fit_taste(choices: list[Choice], emb: Embeddings, l2: float = 1.0) -> TasteModel:
    """Bradley-Terry-style linear utility: learn weights w s.t. w.(emb_chosen - emb_rejected) > 0.

    Implemented as logistic regression over difference vectors with no intercept. `l2` is the
    regularization strength (higher = smoother/less overfit); tune during Phase 1.
    """
    if not choices:
        raise ValueError("need at least one choice to fit")
    X, y = build_dataset(choices, emb)
    clf = LogisticRegression(C=1.0 / l2, fit_intercept=False, max_iter=1000)
    clf.fit(X, y)
    return TasteModel(weights=clf.coef_[0].astype(np.float64))


def rank(items: list[str], model: TasteModel, emb: Embeddings) -> list[str]:
    """Return item ids sorted most-preferred first."""
    return sorted(items, key=lambda i: model.score(emb[i]), reverse=True)
