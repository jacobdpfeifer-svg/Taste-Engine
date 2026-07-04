from __future__ import annotations

import numpy as np

from app.geometry.fit import Choice, Embeddings, TasteModel, fit_taste


def fit_hierarchical(
    choices_by_domain: dict[str, list[Choice]],
    emb: Embeddings,
    l2_shared: float = 1.0,
    l2_offset: float = 4.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (g, {domain: d_k}) where preference_direction(domain) = g + d_k.

    Phase 2 target: proper partial pooling / multilevel logistic regression so sparse domains
    borrow strength from the shared vector g.

    Minimal placeholder implementation (REPLACE): fit g on all choices pooled, then fit each
    domain and take the residual as an offset, shrunk by l2_offset. This is a stand-in so the
    interface exists and returns the right shapes — it is NOT proper partial pooling.
    """
    all_choices = [c for cs in choices_by_domain.values() for c in cs]
    g = fit_taste(all_choices, emb, l2=l2_shared).weights

    offsets: dict[str, np.ndarray] = {}
    for domain, cs in choices_by_domain.items():
        if len(cs) < 5:  # too sparse: lean entirely on g
            offsets[domain] = np.zeros_like(g)
            continue
        w_domain = fit_taste(cs, emb, l2=l2_offset).weights
        offsets[domain] = (w_domain - g) / l2_offset  # shrink the residual toward zero
    return g, offsets


def domain_model(g: np.ndarray, offset: np.ndarray) -> TasteModel:
    return TasteModel(weights=g + offset)
