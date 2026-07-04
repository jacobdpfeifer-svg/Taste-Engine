from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.geometry.exemplar import fit_exemplar
from app.geometry.fit import Choice, Embeddings, fit_taste


@dataclass(frozen=True)
class Metric:
    """An accuracy plus what it's worth: how many held-out choices it was measured on and a
    Wilson 95% interval. Small sessions make single numbers wildly overconfident — the
    interval is the honest report."""

    accuracy: float
    n_test: int
    low: float
    high: float


def wilson_interval(acc: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (better than normal approx at small n)."""
    if n <= 0 or math.isnan(acc):
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _metric(acc: float, n_test: int) -> Metric | None:
    if math.isnan(acc) or n_test <= 0:
        return None
    low, high = wilson_interval(acc, n_test)
    return Metric(accuracy=acc, n_test=n_test, low=low, high=high)


def _accuracy(train: list[Choice], test: list[Choice], emb: Embeddings, l2: float) -> float:
    if not train or not test:
        return float("nan")
    model = fit_taste(train, emb, l2=l2)
    correct = sum(1 for chosen, rejected in test if model.prefers(emb[chosen], emb[rejected]))
    return correct / len(test)


def _accuracy_exemplar(train: list[Choice], test: list[Choice], emb: Embeddings) -> float:
    if not train or not test:
        return float("nan")
    model = fit_exemplar(train, emb)
    correct = sum(1 for chosen, rejected in test if model.prefers(emb[chosen], emb[rejected]))
    return correct / len(test)


def _split_choices(
    choices: list[Choice], test_frac: float, seed: int
) -> tuple[list[Choice], list[Choice]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(choices))
    n_test = max(1, int(len(choices) * test_frac))
    test = [choices[i] for i in idx[:n_test]]
    train = [choices[i] for i in idx[n_test:]]
    return train, test


def holdout_accuracy(
    choices: list[Choice], emb: Embeddings, test_frac: float = 0.3, seed: int = 0, l2: float = 1.0
) -> float:
    """TEST 1. Fit on a training split, predict held-out choices. Chance = 0.5; target > ~0.65."""
    train, test = _split_choices(choices, test_frac, seed)
    return _accuracy(train, test, emb, l2)


def repeated_holdout_accuracy(
    choices: list[Choice],
    emb: Embeddings,
    n_splits: int = 5,
    test_frac: float = 0.3,
    l2: float = 1.0,
) -> float:
    """Hold-out accuracy averaged over several random splits. A single split is noisy at
    session sizes (a few dozen choices); the mean is a steadier signal for the stop condition
    and the reported confidence."""
    accs = [holdout_accuracy(choices, emb, test_frac=test_frac, seed=s, l2=l2) for s in range(n_splits)]
    return float(np.mean(accs))


def holdout_metric(
    choices: list[Choice],
    emb: Embeddings,
    n_splits: int = 5,
    test_frac: float = 0.3,
    l2: float = 1.0,
) -> Metric | None:
    """TEST 1 with error bars: mean accuracy over repeated splits. The interval uses a single
    split's test size — conservative, since repeated splits overlap and aren't independent."""
    acc = repeated_holdout_accuracy(choices, emb, n_splits=n_splits, test_frac=test_frac, l2=l2)
    n_test = max(1, int(len(choices) * test_frac))
    return _metric(acc, n_test)


def holdout_report_exemplar(
    choices: list[Choice],
    emb: Embeddings,
    repeats: int = 25,
    test_frac: float = 0.3,
) -> Metric | None:
    """Hold-out report for the exemplar model using the same repeated split logic."""
    accs = []
    for seed in range(repeats):
        train, test = _split_choices(choices, test_frac, seed)
        accs.append(_accuracy_exemplar(train, test, emb))
    acc = float(np.mean(accs))
    n_test = max(1, int(len(choices) * test_frac))
    return _metric(acc, n_test)


def cross_domain_metric(
    choices_by_domain: dict[str, list[Choice]], emb: Embeddings, held_out_domain: str, l2: float = 1.0
) -> Metric | None:
    """TEST 2 with error bars. Train/test sets are fixed (whole domains), so there is no split
    randomness to average over — the interval reflects the held-out domain's size."""
    acc = cross_domain_accuracy(choices_by_domain, emb, held_out_domain, l2=l2)
    return _metric(acc, len(choices_by_domain.get(held_out_domain, [])))


def ablation_metric(
    choices_by_domain: dict[str, list[Choice]],
    emb: Embeddings,
    target_domain: str,
    n_splits: int = 5,
    test_frac: float = 0.3,
    l2: float = 1.0,
) -> dict[str, Metric] | None:
    """TEST 3 with error bars: both arms averaged over repeated splits (a single split of a
    small domain tests on ~4 choices, where 0.25 vs 1.00 is pure noise)."""
    per_arm: dict[str, list[float]] = {"target_only": [], "all_domains": []}
    for seed in range(n_splits):
        res = ablation(choices_by_domain, emb, target_domain, test_frac=test_frac, seed=seed, l2=l2)
        for arm, acc in res.items():
            per_arm[arm].append(acc)
    n_test = max(1, int(len(choices_by_domain[target_domain]) * test_frac))
    out = {arm: _metric(float(np.mean(accs)), n_test) for arm, accs in per_arm.items()}
    if any(m is None for m in out.values()):
        return None
    return out  # type: ignore[return-value]


def cross_domain_accuracy(
    choices_by_domain: dict[str, list[Choice]], emb: Embeddings, held_out_domain: str, l2: float = 1.0
) -> float:
    """TEST 2. Fit the shared vector on all OTHER domains, predict the held-out domain. Above
    chance => cross-domain taste transfer is real (the 'general taste' premise)."""
    train = [c for d, cs in choices_by_domain.items() if d != held_out_domain for c in cs]
    test = choices_by_domain.get(held_out_domain, [])
    return _accuracy(train, test, emb, l2)


def ablation(
    choices_by_domain: dict[str, list[Choice]],
    emb: Embeddings,
    target_domain: str,
    test_frac: float = 0.3,
    seed: int = 0,
    l2: float = 1.0,
) -> dict[str, float]:
    """TEST 3. Compare predicting target-domain hold-out using (a) target domain only vs
    (b) all domains. If (b) > (a), the multi-medium design earns its complexity."""
    rng = np.random.default_rng(seed)
    target = choices_by_domain[target_domain]
    idx = rng.permutation(len(target))
    n_test = max(1, int(len(target) * test_frac))
    test = [target[i] for i in idx[:n_test]]
    train_target = [target[i] for i in idx[n_test:]]
    others = [c for d, cs in choices_by_domain.items() if d != target_domain for c in cs]

    return {
        "target_only": _accuracy(train_target, test, emb, l2),
        "all_domains": _accuracy(train_target + others, test, emb, l2),
    }
