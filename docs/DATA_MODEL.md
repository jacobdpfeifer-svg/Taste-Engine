# DATA_MODEL

Single source of truth for shapes. Extend this file first if a new need arises.

## Stimulus

```json
{
  "id": "int_001",
  "domain": "interior",
  "path": "data/stimuli/int_001.jpg",
  "caption": "warm minimal living room, oak, low contrast",
  "tags": ["warm", "minimal", "wood", "low-contrast"],
  "embedding": [0.01, -0.22, "... EMBEDDING_DIM floats ..."]
}
```

Domains (extensible): `interior`, `exterior`, `apparel`, `object`, `art`, `landscape`,
`typography`, `palette`.

## Choice

The atomic training signal. Triads and board-selects are decomposed into pairwise rows.

```json
{
  "id": 42,
  "session_id": "sess_abc",
  "kind": "pair",              // "pair" | "triad" | "board"
  "chosen_id": "int_001",
  "rejected_id": "int_014",
  "domain": "interior",       // domain of the pair; "cross" if the pair mixes domains
  "created_at": "2026-07-01T12:00:00Z"
}
```

## Session

```json
{ "id": "sess_abc", "person_id": "p_1", "created_at": "..." }
```

## Taste kernel (Phase 1 = shared_vector only; later phases add the rest)

```json
{
  "person_id": "p_1",
  "shared_vector": [/* EMBEDDING_DIM floats: g */],
  "domain_offsets": { "interior": [/* d_k */], "apparel": [/* d_k */] },
  "axes": [
    { "label": "warm ↔ cool", "vector": [/* ... */], "polarity": "prefers warm" }
  ],
  "color_map": {
    "calm":    { "hue_range": [180, 240], "sat": [0.1, 0.4], "val": [0.6, 0.9] },
    "focused": { "hue_range": [200, 260], "sat": [0.2, 0.5], "val": [0.3, 0.6] }
  },
  "exemplars": { "interior": ["int_001", "int_007"] },
  "provenance": {
    "n_choices": 120,
    "per_domain_confidence": { "interior": 0.82, "apparel": 0.55 },
    "fitted_at": "...",
    "holdout_accuracy": 0.71
  }
}
```

## SQLite tables (Phase 1)

- `sessions(id TEXT PK, person_id TEXT, created_at TEXT)`
- `choices(id INTEGER PK, session_id TEXT, kind TEXT, chosen_id TEXT, rejected_id TEXT,
   domain TEXT, created_at TEXT)`

Stimuli + embeddings are loaded into memory at startup (Phase 1) from `data/stimuli.json` if it
exists (generate with `python scripts/generate_stimuli.py`), else `data/stimuli.sample.json`.
Override with `STIMULI_PATH`. Move to a table only if the bank grows large.

Stimulus images live in `data/stimuli/` (gitignored) and are served statically by the backend at
`/data/stimuli/<file>`.

## API (Phase 1)

- `GET  /stimuli` → list stimuli (no embeddings in the payload)
- `POST /sessions` → `{ person_id }` → `{ session_id }`
- `GET  /sessions/{id}/next-pair?domain=` → `{ a: Stimulus, b: Stimulus }` (active learning;
  falls back to repeats if every pair in the domain has been shown)
- `GET  /sessions/{id}/next-triad?domain=` → `{ a, b, c }` (odd-one-out prompt; with a model,
  picks the triple with the smallest score spread)
- `POST /choices` → Choice row → `{ ok: true }`
- `POST /triads` → `{ session_id, odd_id, other_ids: [x, y], domain }` → decomposed into two
  pairwise choice rows (kind `"triad"`: each kept item chosen over the odd one) → `{ ok: true }`
- `GET  /sessions/{id}/progress` → `{ n_choices, holdout_accuracy, done, reason }` — the
  stop-condition signal; `reason` is `"converged" | "budget" | "collecting"`. The frontend polls
  this between choices and ends the session when `done`.
- `POST /sessions/{id}/fit` → fits `w`, persists the kernel to `kernels/<person_id>.json`,
  returns `{ n_choices, holdout_accuracy, kernel_path }`
- `GET  /sessions/{id}/validate` →
  `{ holdout: Metric, cross_domain: {domain: Metric}, ablation: {domain: {target_only, all_domains}}, n_choices, per_domain_counts }`
  where `Metric = { accuracy, n_test, low, high }` (Wilson 95% interval on the held-out sample). where every accuracy is a
  **Metric**: `{ accuracy, n_test, low, high }` (Wilson 95% interval over the held-out sample —
  small sessions make bare accuracies overconfident; hold-out and ablation are averaged over
  repeated splits)
