"""End-to-end API test: a simulated person with a consistent tag-based taste runs a full
session (next-pair -> choice -> ... -> fit -> validate) and the fitted model must predict
their held-out choices above chance. This is the Phase 1 loop exercised on the stub backend.

Env isolation (db path, kernels dir, stimuli bank) is set up in conftest.py.
"""
from __future__ import annotations

import os
import random

from fastapi.testclient import TestClient

import app.main as main
from app.main import app, bank

client = TestClient(app)

# A person who consistently prefers one pole of each style axis (the generated bank encodes
# three axes; see scripts/generate_stimuli.py), caring about some axes more than others.
# Graded weights keep pairs from tying in utility — ties would be decided arbitrarily and read
# as noise, which uncertainty sampling then fixates on.
TAG_UTILITY = {
    "warm": 3.0, "cool": -3.0,
    "minimal": 2.5, "ornate": -2.5,
    "organic": 2.0, "geometric": -2.0,
}

DOMAINS = ["interior", "exterior", "apparel", "palette"]


def _utility(stimulus_id: str) -> float:
    return sum(TAG_UTILITY.get(t, 0.0) for t in bank.items[stimulus_id].tags)


def _run_session(n_choices: int) -> str:
    session_id = client.post("/sessions", json={"person_id": "p_test"}).json()["session_id"]
    for i in range(n_choices):
        domain = DOMAINS[i % len(DOMAINS)]
        pair = client.get(f"/sessions/{session_id}/next-pair", params={"domain": domain}).json()
        a, b = pair["a"], pair["b"]
        chosen, rejected = (a, b) if _utility(a["id"]) >= _utility(b["id"]) else (b, a)
        r = client.post(
            "/choices",
            json={
                "session_id": session_id,
                "chosen_id": chosen["id"],
                "rejected_id": rejected["id"],
                "domain": domain,
            },
        )
        assert r.status_code == 200
    return session_id


def test_stimuli_listing_has_no_embeddings():
    r = client.get("/stimuli")
    assert r.status_code == 200
    items = r.json()
    assert len(items) >= 8
    assert "embedding" not in items[0]


def test_unknown_session_is_404():
    assert client.get("/sessions/nope/next-pair").status_code == 404
    assert client.post("/sessions/nope/fit").status_code == 404


def test_unknown_stimulus_is_400():
    session_id = client.post("/sessions", json={"person_id": "p_x"}).json()["session_id"]
    r = client.post(
        "/choices",
        json={"session_id": session_id, "chosen_id": "nope", "rejected_id": "int_000"},
    )
    assert r.status_code == 400


def test_grid_returns_distinct_varied_stimuli():
    session_id = client.post("/sessions", json={"person_id": "p_grid"}).json()["session_id"]
    responses = [
        client.get(f"/sessions/{session_id}/grid", params={"k": 9})
        for _ in range(5)
    ]
    assert all(r.status_code == 200 for r in responses)
    grids = []
    for r in responses:
        items = r.json()["items"]
        ids = [item["id"] for item in items]
        assert len(ids) == 9
        assert len(set(ids)) == 9
        grids.append(tuple(ids))
    assert len(set(grids)) > 1


def test_choices_batch_counts_in_validate():
    session_id = client.post("/sessions", json={"person_id": "p_batch"}).json()["session_id"]
    ids = list(bank.items.keys())[:6]
    records = [
        {"chosen_id": ids[i], "rejected_id": ids[i + 1], "domain": "interior"}
        for i in range(5)
    ]
    r = client.post("/choices/batch", json={"session_id": session_id, "records": records})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "n": 5}
    val = client.get(f"/sessions/{session_id}/validate").json()
    assert val["n_choices"] == 5
    assert val["per_domain_counts"]["interior"] == 5


def test_reflect_after_session():
    session_id = client.post("/sessions", json={"person_id": "p_reflect"}).json()["session_id"]
    # Prefer warm/minimal stimuli consistently so reflect-back has a named pattern.
    warm_minimal = [
        sid for sid, s in bank.items.items()
        if "warm" in s.tags and "minimal" in s.tags
    ][:6]
    other = [
        sid for sid, s in bank.items.items()
        if "cool" in s.tags and "ornate" in s.tags
    ][:6]
    records = [
        {"chosen_id": warm_minimal[i % len(warm_minimal)],
         "rejected_id": other[i % len(other)], "domain": "interior"}
        for i in range(8)
    ]
    client.post("/choices/batch", json={"session_id": session_id, "records": records})

    r = client.get(f"/sessions/{session_id}/reflect")
    assert r.status_code == 200
    body = r.json()
    assert len(body["hero_ids"]) <= 5
    assert len(body["hero_ids"]) >= 1
    assert isinstance(body["summary"], str)
    assert body["summary"].endswith(".")
    assert "warm" in body["top_tags"] or "minimal" in body["top_tags"]


def test_full_session_beats_chance():
    # Pair selection is stochastic (cold start + ε-greedy exploration); seed for a
    # deterministic session so this test can't flake on an unlucky sample.
    random.seed(7)
    session_id = _run_session(48)

    progress = client.get(f"/sessions/{session_id}/progress").json()
    assert progress["n_choices"] == 48

    fit = client.post(f"/sessions/{session_id}/fit").json()
    assert fit["n_choices"] == 48
    assert fit["kernel_path"] is not None and os.path.isfile(fit["kernel_path"])

    val = client.get(f"/sessions/{session_id}/validate").json()
    h = val["holdout"]
    he = val["holdout_exemplar"]
    assert h is not None
    assert he is not None
    assert val["best_model"] in {"linear", "exemplar"}
    assert h["accuracy"] > 0.6, f"holdout {h['accuracy']:.2f} not above chance"
    assert 0.0 <= h["low"] <= h["accuracy"] <= h["high"] <= 1.0
    assert 0.0 <= he["low"] <= he["accuracy"] <= he["high"] <= 1.0
    assert h["n_test"] > 0
    assert set(val["per_domain_counts"]) == set(DOMAINS)
    # cross-domain transfer should be above chance too for a consistent tag-based person
    assert val["cross_domain"], "expected cross-domain results"
    mean_cross = sum(m["accuracy"] for m in val["cross_domain"].values()) / len(val["cross_domain"])
    assert mean_cross > 0.5, f"mean cross-domain {mean_cross:.2f} at or below chance"


def test_triad_decomposes_into_two_pairwise_rows():
    session_id = client.post("/sessions", json={"person_id": "p_triad"}).json()["session_id"]

    r = client.get(f"/sessions/{session_id}/next-triad", params={"domain": "interior"})
    assert r.status_code == 200
    triad = r.json()
    ids = [triad["a"]["id"], triad["b"]["id"], triad["c"]["id"]]
    assert len(set(ids)) == 3

    # The odd one out is the lowest-utility item; the other two were each "chosen" over it.
    odd = min(ids, key=_utility)
    others = [i for i in ids if i != odd]
    r = client.post(
        "/triads",
        json={"session_id": session_id, "odd_id": odd, "other_ids": others, "domain": "interior"},
    )
    assert r.status_code == 200

    progress = client.get(f"/sessions/{session_id}/progress").json()
    assert progress["n_choices"] == 2, "one triad answer must yield two pairwise rows"


def test_triad_rejects_malformed_answers():
    session_id = client.post("/sessions", json={"person_id": "p_triad2"}).json()["session_id"]
    base = {"session_id": session_id, "domain": "interior"}
    # odd_id repeated in other_ids
    r = client.post("/triads", json={**base, "odd_id": "int_000", "other_ids": ["int_000", "int_001"]})
    assert r.status_code == 400
    # wrong count
    r = client.post("/triads", json={**base, "odd_id": "int_000", "other_ids": ["int_001"]})
    assert r.status_code == 400
    # unknown stimulus
    r = client.post("/triads", json={**base, "odd_id": "nope", "other_ids": ["int_001", "int_002"]})
    assert r.status_code == 400


def test_pair_exhaustion_allows_repeats(monkeypatch):
    # With a 3-stimulus domain (3 distinct pairs), asking for many pairs must keep working:
    # the route falls back to repeats once fresh pairs run out.
    from app.stimuli.loader import Stimulus, StimulusBank

    small = StimulusBank(
        items={
            s.id: s
            for s in (
                Stimulus(id=f"t_{i}", domain="tiny", path="", caption=f"tiny {i}",
                         embedding=bank.items[list(bank.items)[i]].embedding)
                for i in range(3)
            )
        }
    )
    monkeypatch.setattr(main, "bank", small)

    session_id = client.post("/sessions", json={"person_id": "p_exhaust"}).json()["session_id"]
    for _ in range(6):
        r = client.get(f"/sessions/{session_id}/next-pair", params={"domain": "tiny"})
        assert r.status_code == 200
        pair = r.json()
        client.post(
            "/choices",
            json={
                "session_id": session_id,
                "chosen_id": pair["a"]["id"],
                "rejected_id": pair["b"]["id"],
                "domain": "tiny",
            },
        )
