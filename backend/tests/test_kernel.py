from __future__ import annotations

import numpy as np

from app.geometry.fit import TasteModel
from app.kernel import build_kernel, kernel_model, load_kernel, save_kernel


def test_kernel_roundtrip(tmp_path):
    dim = 8
    rng = np.random.default_rng(0)
    ids = [f"s{i}" for i in range(6)]
    emb = {i: rng.normal(size=dim) for i in ids}
    model = TasteModel(weights=rng.normal(size=dim))
    choices = [("s0", "s1"), ("s2", "s3")]
    kernel = build_kernel(
        person_id="p_test",
        model=model,
        choices=choices,
        choices_by_domain={"interior": choices},
        ids_by_domain={"interior": ids},
        emb=emb,
        holdout_accuracy=0.7,
        per_domain_confidence={"interior": 0.7},
    )

    path = save_kernel(kernel, str(tmp_path))
    loaded = load_kernel("p_test", str(tmp_path))
    assert path.endswith("p_test.json")
    assert loaded is not None
    assert loaded["provenance"]["n_choices"] == 2
    assert np.allclose(kernel_model(loaded).weights, model.weights)

    # exemplars are the top-scoring items for the fitted direction
    scores = {i: model.score(emb[i]) for i in ids}
    best = max(scores, key=scores.get)
    assert loaded["exemplars"]["interior"][0] == best


def test_load_missing_returns_none(tmp_path):
    assert load_kernel("nobody", str(tmp_path)) is None
