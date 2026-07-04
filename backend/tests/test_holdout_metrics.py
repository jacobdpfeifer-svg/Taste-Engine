from __future__ import annotations

import math

from app.validation.holdout import Metric, wilson_interval, _metric


def test_wilson_interval_bounds():
    low, high = wilson_interval(0.5, 20)
    assert 0 <= low <= 0.5 <= high <= 1.0


def test_wilson_interval_perfect():
    low, high = wilson_interval(1.0, 10)
    assert low > 0.7 and high == 1.0


def test_metric_wraps():
    m = _metric(0.6, 18)
    assert m is not None
    assert m.n_test == 18
    assert m.low <= m.accuracy <= m.high
    assert not math.isnan(m.low)
