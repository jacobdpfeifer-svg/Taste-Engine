from __future__ import annotations

import numpy as np

from app.validation.holdout import (
    ablation_metric,
    cross_domain_metric,
    holdout_metric,
    holdout_report_exemplar,
    wilson_interval,
)


def test_wilson_interval_bounds_and_width():
    low, high = wilson_interval(0.5, 10)
    assert 0.0 <= low < 0.5 < high <= 1.0
    low_big, high_big = wilson_interval(0.5, 1000)
    assert (high_big - low_big) < (high - low), "more evidence must narrow the interval"
    # extremes stay inside [0, 1]
    assert wilson_interval(1.0, 4)[1] <= 1.0
    assert wilson_interval(0.0, 4)[0] >= 0.0


def test_wilson_interval_degenerate():
    low, high = wilson_interval(float("nan"), 10)
    assert np.isnan(low) and np.isnan(high)
    low, high = wilson_interval(0.7, 0)
    assert np.isnan(low) and np.isnan(high)


def _synthetic(n_stim=60, dim=16, n_choices=200, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n_stim)]
    emb = {i: rng.normal(size=dim) for i in ids}
    true = rng.normal(size=dim)
    choices = []
    for _ in range(n_choices):
        a, b = rng.choice(ids, size=2, replace=False)
        chosen, rejected = (a, b) if (emb[a] - emb[b]) @ true > 0 else (b, a)
        choices.append((chosen, rejected))
    return emb, choices


def test_holdout_metric_shape():
    emb, choices = _synthetic()
    m = holdout_metric(choices, emb)
    assert m is not None
    assert m.n_test == int(len(choices) * 0.3)
    assert 0.0 <= m.low <= m.accuracy <= m.high <= 1.0
    assert m.accuracy > 0.6  # deterministic chooser: should be easy


def test_holdout_report_exemplar_shape():
    emb, choices = _synthetic()
    m = holdout_report_exemplar(choices, emb)
    assert m is not None
    assert m.n_test == int(len(choices) * 0.3)
    assert 0.0 <= m.low <= m.accuracy <= m.high <= 1.0


def test_cross_domain_and_ablation_metrics():
    emb, choices = _synthetic()
    half = len(choices) // 2
    by_domain = {"A": choices[:half], "B": choices[half:]}

    cm = cross_domain_metric(by_domain, emb, "B")
    assert cm is not None and cm.n_test == len(by_domain["B"])

    am = ablation_metric(by_domain, emb, "A")
    assert am is not None
    assert set(am) == {"target_only", "all_domains"}
    for m in am.values():
        assert 0.0 <= m.low <= m.accuracy <= m.high <= 1.0
