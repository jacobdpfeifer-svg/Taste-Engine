from __future__ import annotations

import numpy as np

from app.geometry.exemplar import ExemplarModel, fit_exemplar


def test_exemplar_score_uses_mean_dot_products():
    model = ExemplarModel(
        pos=np.array([[1.0, 0.0], [0.0, 1.0]]),
        neg=np.array([[-1.0, 0.0]]),
    )
    assert model.score(np.array([1.0, 0.0])) == 1.5


def test_fit_exemplar_prefers_chosen_exemplar():
    emb = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([-1.0, 0.0]),
    }
    model = fit_exemplar([("a", "b")], emb)
    assert model.prefers(emb["a"], emb["b"])
