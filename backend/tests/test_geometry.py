"""Synthetic proof that the core mechanism works.

We invent a ground-truth taste vector `t`, simulate a person choosing between random stimuli
according to it (Bradley-Terry), then check that (1) hold-out prediction beats chance and
(2) the fitted vector aligns with the true one. If these pass, the geometry premise is sound;
what remains is whether REAL embeddings + REAL choices behave the same (that's Phase 1's gate).
"""
from __future__ import annotations

import numpy as np

from app.geometry.fit import fit_taste
from app.validation.holdout import (
    ablation,
    cross_domain_accuracy,
    holdout_accuracy,
)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _make_person(n_stim=240, dim=32, n_choices=1500, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n_stim)]
    emb = {i: rng.normal(size=dim) for i in ids}
    for i in ids:
        emb[i] /= np.linalg.norm(emb[i])
    true = rng.normal(size=dim)
    true /= np.linalg.norm(true)
    choices = []
    for _ in range(n_choices):
        a, b = rng.choice(ids, size=2, replace=False)
        p = _sigmoid(float((emb[a] - emb[b]) @ true) * 8.0)  # temperature sharpens signal
        chosen, rejected = (a, b) if rng.random() < p else (b, a)
        choices.append((chosen, rejected))
    return emb, true, choices


def test_holdout_beats_chance():
    emb, _true, choices = _make_person()
    acc = holdout_accuracy(choices, emb, test_frac=0.3, seed=1)
    assert acc > 0.65, f"hold-out accuracy {acc:.3f} not above chance"


def test_recovers_true_direction():
    emb, true, choices = _make_person()
    w = fit_taste(choices, emb).weights
    cos = float(w @ true) / (np.linalg.norm(w) * np.linalg.norm(true))
    assert cos > 0.5, f"fitted vector cosine to truth only {cos:.3f}"


def test_cross_domain_transfer_when_taste_is_shared():
    # Same underlying taste across two synthetic "domains" -> transfer should beat chance.
    emb, _true, choices = _make_person(seed=2)
    half = len(choices) // 2
    by_domain = {"A": choices[:half], "B": choices[half:]}
    acc = cross_domain_accuracy(by_domain, emb, held_out_domain="B")
    assert acc > 0.6, f"cross-domain accuracy {acc:.3f} not above chance"


def test_ablation_runs():
    emb, _true, choices = _make_person(seed=3)
    third = len(choices) // 3
    by_domain = {"A": choices[:third], "B": choices[third:2 * third], "C": choices[2 * third:]}
    res = ablation(by_domain, emb, target_domain="A")
    assert set(res) == {"target_only", "all_domains"}
