# TASKS

Ordered build plan. Check items off as you go. Respect the phase gates in
`.cursor/rules/030-build-order.mdc`. Acceptance criteria are in **bold**.

## Phase 0 — Runnable skeleton  ✅ mostly scaffolded

- [x] Repo structure, Cursor rules, docs.
- [x] `EmbeddingAdapter` interface + deterministic `stub_adapter`.
- [x] `geometry/fit.py` — logistic fit of taste direction from pairwise choices.
- [x] `validation/holdout.py` — hold-out, cross-domain, ablation.
- [x] `elicitation/active.py` — uncertainty-sampling next-pair (random on cold start).
- [x] `tests/test_geometry.py` — synthetic proof: recovers a known taste vector, beats chance.
- [x] `db.py` — create sqlite tables; helpers to insert/list sessions and choices.
- [x] `stimuli/loader.py` — load the stimulus bank, attach embeddings via the adapter
  (caption/tags are passed to the adapter so the stub embeds semantics, not just id hashes).
- [x] `main.py` — wire the API in `DATA_MODEL.md`. **`uvicorn app.main:app` boots; `GET /stimuli`
  returns the sample bank; `pytest` passes.**
- [x] Frontend: Vite app renders `PairChoice`, posts a choice, loads the next pair.
  **Clicking an image records a choice and advances.**

## Phase 1 — Validation (the point)

- [x] Wire the full loop: session → repeated next-pair/choice → fit → validate.
  (`tests/test_api.py` runs it end-to-end with a simulated chooser and beats chance.)
- [x] Expand the bank to ≥ 40 stimuli/domain across ≥ 4 domains: `scripts/generate_stimuli.py`
  writes `data/stimuli.json` + tag-encoding placeholder SVGs to `data/stimuli/` (gitignored).
- [ ] Replace placeholder SVGs with a real curated photo set and switch
  `EMBEDDING_BACKEND=openclip`. Tooling ready: drop photos into `data/stimuli/<domain>/`, run
  `scripts/ingest_images.py`, verify with `scripts/verify_openclip.py`. Only the photo curation
  itself remains.
- [x] Elicitation shows images only (no captions) — words invite choosing by description
  (verbal overshadowing) and the pixels are what gets embedded.
- [x] Validation metrics carry error bars: Wilson 95% intervals + held-out sample sizes on all
  three tests; hold-out and ablation averaged over repeated splits; dashboard withholds
  verdicts when intervals include chance.
- [x] Domain-focus start screen (pick which mediums to assess), triads every 3rd round, choice
  budget raised to 96 — more signal per domain for the per-domain tests.
- [x] Placeholder SVGs re-rendered to encode 3 salient axes (warm/cool, minimal/ornate,
  geometric/organic) instead of 6 subtle ones, so human visual choices are learnable by the
  stub in dry runs.
- [x] Add a "warmer/colder" style stop condition: `GET /sessions/{id}/progress` ends the session
  when hold-out accuracy converges or after a choice budget (`elicitation/active.py:should_stop`).
- [x] Odd-one-out triads: `next_triad` (smallest score spread), `GET .../next-triad`,
  `POST /triads` (decomposed into two pairwise rows), and a `TriadChoice` UI mixed into the
  session flow every 5th round.
- [x] Build the results/validation dashboard (frontend) showing the three test outputs.
- [ ] **GATE: run a real session (real person, real images, openclip); hold-out accuracy >
  chance with the Wilson interval clearing 0.50. If not, stop and report.** (Passes with
  simulated choosers on the stub backend; the real-human/real-image run is still open.)
- [x] Elicitation shows images only (captions hidden) to avoid verbal overshadowing.
- [x] Validation dashboard reports Wilson 95% intervals + n_test; ablation averaged over
  repeated splits; verdicts only when intervals clear chance.
- [x] Domain picker (focus on 2 mediums), triads every 3rd round, budget raised to 96.
- [x] `scripts/ingest_images.py` + `scripts/verify_openclip.py` for the real-photo path.

## Phase 2 — Hierarchy + articulation

- [ ] `geometry/hierarchy.py` — partial-pooling fit of shared `g` + per-domain `d_k`.
- [ ] `geometry/axes.py` — PCA/linear-probe axis discovery; label axes by nearest caption/tag
  anchors.
- [ ] Confirmation step: AI states discovered axes/principles; user confirms/vetoes; vetoes
  re-weight the model. **Words are AI-authored, user-ratified — never free-text taste input.**
- [ ] **GATE: cross-domain transfer > chance, else downgrade to per-domain assessments.**

## Phase 3 — Color-affect

- [ ] Color-affect elicitation UI (palette/scene → emotion pick).
- [ ] Build the personal `color_map`; expose it as a generation constraint.

## Phase 4 — Generation (do not start before Phase 1–2 gates pass)

- [ ] `GenerationAdapter` interface (mirror the embedding-adapter pattern).
- [ ] Brief composer: `g + d_domain + exemplars + color constraints + request` → generator.
- [ ] Per-domain routing: photoreal scene model for interior/exterior/apparel; deterministic
  palette generator for color schemes.
- [ ] Every output ships with a rationale (which axes/principles drove it) + warmer/colder
  feedback that refines the kernel.

## Cross-cutting (any phase)

- [x] Persist kernels to `kernels/<person_id>.json` (gitignored) — `app/kernel.py`, written on
  `POST /sessions/{id}/fit`.
- [ ] Confidence surfaced everywhere; never present predictions as certainty.
- [ ] Keep `DATA_MODEL.md` authoritative — update it before changing shapes.
