from __future__ import annotations

import numpy as np

from app.embeddings.stub_adapter import StubAdapter


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def test_deterministic_and_normalized():
    ad = StubAdapter(dim=64)
    v1 = ad.embed("x_001", text="warm minimal room warm minimal")
    v2 = ad.embed("x_001", text="warm minimal room warm minimal")
    assert np.allclose(v1, v2)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-9


def test_shared_tags_are_closer_than_disjoint():
    ad = StubAdapter(dim=64)
    a = ad.embed("a", text="warm minimal geometric")
    b = ad.embed("b", text="warm minimal organic")
    c = ad.embed("c", text="cool ornate glossy")
    assert _cos(a, b) > _cos(a, c), "stimuli sharing tags should be closer in stub space"


def test_no_text_falls_back_to_id_hash():
    ad = StubAdapter(dim=32)
    v = ad.embed("only_id")
    assert v.shape == (32,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9
    assert not np.allclose(v, ad.embed("other_id"))
