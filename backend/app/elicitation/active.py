from __future__ import annotations

import itertools
import random

import numpy as np

from app.geometry.fit import Embeddings, TasteModel

Pair = tuple[str, str]


def diverse_sample(ids: list[str], emb: dict, k: int, seed: int | None = None) -> list[str]:
    """Select a diverse subset by farthest-point sampling over cosine distance."""
    if k >= len(ids):
        return ids
    if k <= 0:
        return []

    rng = random.Random(seed) if seed is not None else random
    chosen = [rng.choice(ids)]
    remaining = [sid for sid in ids if sid != chosen[0]]

    while len(chosen) < k and remaining:
        best_id = max(
            remaining,
            key=lambda sid: min(1.0 - float(np.dot(emb[sid], emb[cid])) for cid in chosen),
        )
        chosen.append(best_id)
        remaining.remove(best_id)
    return chosen


def next_pair(
    candidate_ids: list[str],
    emb: Embeddings,
    model: TasteModel | None = None,
    already_shown: set[frozenset[str]] | None = None,
    max_candidates: int = 400,
    exploration_rate: float = 0.3,
    mode: str = "diverse",
) -> Pair:
    """Pick the most informative next pair.

    Default / cold start: show a visibly different pair using farthest-point sampling.
    Refine mode with a model: uncertainty sampling — choose the pair the model is least sure
    about (smallest |score(a) - score(b)|), since that choice tells us the most. Sampling a
    subset of candidate pairs keeps this cheap.

    `exploration_rate` (ε-greedy) mixes in random pairs even once a model exists. Without it,
    the session collects ONLY maximally ambiguous pairs, and since hold-out accuracy is measured
    on the collected choices, the estimate decays toward chance as the sample becomes
    adversarially hard — pure uncertainty sampling quietly breaks the Phase 1 measurement.
    """
    already_shown = already_shown or set()

    def fresh(a: str, b: str) -> bool:
        return frozenset((a, b)) not in already_shown and a != b

    pairs = [(a, b) for a, b in itertools.combinations(candidate_ids, 2) if fresh(a, b)]
    if not pairs:
        raise ValueError("no fresh pairs remain")

    if model is None or mode != "refine":
        a, b = diverse_sample(candidate_ids, emb, 2)
        pair = (a, b)
        if fresh(a, b):
            return pair
        return random.choice(pairs)

    if random.random() < exploration_rate:
        return random.choice(pairs)

    random.shuffle(pairs)
    pairs = pairs[:max_candidates]
    return min(pairs, key=lambda p: abs(model.score(emb[p[0]]) - model.score(emb[p[1]])))


Triad = tuple[str, str, str]


def next_triad(
    candidate_ids: list[str],
    emb: Embeddings,
    model: TasteModel | None = None,
    n_samples: int = 200,
    exploration_rate: float = 0.3,
) -> Triad:
    """Pick three stimuli for an odd-one-out prompt ("which is least you?").

    The answer decomposes into two pairwise choices (each kept item > the odd one), so triads
    are denser signal per screen than pairs; they also surface which dimensions the person
    sorts on, which Phase 2's axis discovery will exploit.

    Cold start / exploration: random triple. With a model: sample triples and take the one
    with the smallest score spread — three items the model can barely rank are the most
    informative to have disambiguated.
    """
    if len(candidate_ids) < 3:
        raise ValueError("need at least 3 stimuli for a triad")
    if model is None or random.random() < exploration_rate:
        return tuple(random.sample(candidate_ids, 3))

    best: Triad | None = None
    best_spread = float("inf")
    for _ in range(n_samples):
        trio = random.sample(candidate_ids, 3)
        scores = sorted(model.score(emb[i]) for i in trio)
        spread = scores[-1] - scores[0]
        if spread < best_spread:
            best, best_spread = tuple(trio), spread
    assert best is not None
    return best


def should_stop(
    n_choices: int,
    holdout_accuracy: float | None,
    min_choices: int = 30,
    max_choices: int = 96,
    target_accuracy: float = 0.70,
) -> tuple[bool, str]:
    """Warmer/colder stop condition: end the session once the model predicts held-out choices
    well enough, or once the choice budget is spent (whichever comes first).

    Returns (done, reason). Reasons: "converged" | "budget" | "collecting".
    """
    if n_choices >= max_choices:
        return True, "budget"
    if (
        n_choices >= min_choices
        and holdout_accuracy is not None
        and holdout_accuracy >= target_accuracy
    ):
        return True, "converged"
    return False, "collecting"
