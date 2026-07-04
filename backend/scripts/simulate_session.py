"""Drive a full elicitation session against a running backend with a simulated chooser.

Useful as a live smoke test and as a demo of the Phase 1 loop:
    python scripts/simulate_session.py [--base http://localhost:8000]

The simulated person consistently prefers one pole of each style axis, so the fitted model
should predict their held-out choices well above chance.
"""
from __future__ import annotations

import argparse
import json
import urllib.request

# Graded per-tag utilities (some axes matter more); no two-tag ties, so choices are consistent.
TAG_UTILITY = {
    "warm": 3.0, "cool": -3.0,
    "minimal": 2.5, "ornate": -2.5,
    "organic": 2.0, "geometric": -2.0,
}
DOMAINS = ["interior", "exterior", "apparel", "palette"]


def call(base: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def utility(stim: dict) -> float:
    return sum(TAG_UTILITY.get(t, 0.0) for t in stim["tags"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    args = ap.parse_args()

    session_id = call(args.base, "/sessions", {"person_id": "p_sim"})["session_id"]
    print(f"session {session_id}")

    i = 0
    while True:
        domain = DOMAINS[i % len(DOMAINS)]
        if (i + 1) % 3 == 0:  # every 3rd round: odd-one-out triad, like the frontend
            triad = call(args.base, f"/sessions/{session_id}/next-triad?domain={domain}")
            items = [triad["a"], triad["b"], triad["c"]]
            odd = min(items, key=utility)
            call(args.base, "/triads", {
                "session_id": session_id, "odd_id": odd["id"],
                "other_ids": [s["id"] for s in items if s["id"] != odd["id"]],
                "domain": domain,
            })
        else:
            pair = call(args.base, f"/sessions/{session_id}/next-pair?domain={domain}")
            a, b = pair["a"], pair["b"]
            chosen, rejected = (a, b) if utility(a) >= utility(b) else (b, a)
            call(args.base, "/choices", {
                "session_id": session_id, "chosen_id": chosen["id"],
                "rejected_id": rejected["id"], "domain": domain,
            })
        i += 1
        p = call(args.base, f"/sessions/{session_id}/progress")
        acc = f"{p['holdout_accuracy']:.2f}" if p["holdout_accuracy"] is not None else "--"
        print(f"round {i:2d}  choices={p['n_choices']:2d}  holdout={acc}  {p['reason']}")
        if p["done"]:
            break

    fit = call(args.base, f"/sessions/{session_id}/fit", {})
    val = call(args.base, f"/sessions/{session_id}/validate")
    h = val["holdout"]
    print(f"\nstopped after {fit['n_choices']} choices ({p['reason']})")
    print(f"kernel: {fit['kernel_path']}")
    if h:
        print(f"holdout: {h['accuracy']:.2f}  [{h['low']:.2f}–{h['high']:.2f}], n={h['n_test']}  (chance 0.50)")
    print("cross-domain:", {d: round(m["accuracy"], 2) for d, m in val["cross_domain"].items()})
    print("ablation:", {
        d: {k: round(m["accuracy"], 2) for k, m in arms.items()}
        for d, arms in val["ablation"].items()
    })


if __name__ == "__main__":
    main()
