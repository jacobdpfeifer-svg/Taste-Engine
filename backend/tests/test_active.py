from __future__ import annotations

import numpy as np
import pytest

from app.elicitation.active import diverse_sample, next_pair, next_triad, should_stop
from app.geometry.fit import TasteModel


def _emb(ids: list[str], dim: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {i: rng.normal(size=dim) for i in ids}


def test_diverse_sample_uses_farthest_points():
    emb = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.9, 0.1]),
        "c": np.array([-1.0, 0.0]),
        "d": np.array([0.0, 1.0]),
    }
    assert diverse_sample(list(emb), emb, 3, seed=0) == ["d", "a", "c"]


def test_cold_start_returns_fresh_pair():
    ids = ["a", "b", "c"]
    emb = _emb(ids)
    shown = {frozenset(("a", "b")), frozenset(("a", "c"))}
    pair = next_pair(ids, emb, model=None, already_shown=shown)
    assert frozenset(pair) == frozenset(("b", "c"))


def test_exhaustion_raises():
    ids = ["a", "b"]
    emb = _emb(ids)
    with pytest.raises(ValueError):
        next_pair(ids, emb, model=None, already_shown={frozenset(("a", "b"))})


def test_uncertainty_sampling_picks_closest_scores():
    ids = ["a", "b", "c"]
    emb = {"a": np.array([1.0, 0.0]), "b": np.array([0.9, 0.0]), "c": np.array([-1.0, 0.0])}
    model = TasteModel(weights=np.array([1.0, 0.0]))
    pair = next_pair(ids, emb, model=model, already_shown=set(), exploration_rate=0.0, mode="refine")
    assert frozenset(pair) == frozenset(("a", "b")), "should pick the least-decided pair"


def test_triad_cold_start_returns_three_distinct():
    ids = ["a", "b", "c", "d"]
    triad = next_triad(ids, _emb(ids), model=None)
    assert len(set(triad)) == 3
    assert set(triad) <= set(ids)


def test_triad_needs_three():
    ids = ["a", "b"]
    with pytest.raises(ValueError):
        next_triad(ids, _emb(ids))


def test_triad_uncertainty_picks_tightest_spread():
    # Three items score nearly the same; one is far away. The informative triad is the
    # close-scoring three.
    emb = {
        "a": np.array([1.00, 0.0]),
        "b": np.array([1.01, 0.0]),
        "c": np.array([0.99, 0.0]),
        "far": np.array([-5.0, 0.0]),
    }
    model = TasteModel(weights=np.array([1.0, 0.0]))
    triad = next_triad(list(emb), emb, model=model, n_samples=500, exploration_rate=0.0)
    assert set(triad) == {"a", "b", "c"}


def test_should_stop_transitions():
    assert should_stop(5, None) == (False, "collecting")
    assert should_stop(30, 0.55) == (False, "collecting")
    assert should_stop(30, 0.75) == (True, "converged")
    assert should_stop(96, None) == (True, "budget")
    # accuracy alone is not enough before min_choices
    assert should_stop(10, 0.9) == (False, "collecting")
