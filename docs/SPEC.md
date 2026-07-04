# SPEC — Taste Engine

Build-oriented condensation of the system plan. See `.cursor/rules/` for how to build; this is
*what* to build and why.

## Premise (the thing we are testing)

A person's aesthetic preferences are partly governed by a stable disposition representable as a
location/direction in a learned embedding space, and choices in some mediums predict choices in
others above chance. **Phase 1 exists to falsify or confirm this.** If confirmed, one assessment
generalizes across domains; if not, we fall back to independent per-domain assessments.

## The four things the system does

1. **Elicit** preferences by choice (pairwise, odd-one-out, board-select) across mediums —
   interiors, exteriors/architecture, apparel, objects, art, landscape, typography, palettes.
2. **Represent** taste as geometry: fit a taste direction `w` from choices where, for a chosen-
   over-rejected pair, `w · (emb_chosen − emb_rejected) > 0`. Hierarchical: `w_domain = g + d_k`.
3. **Store** a taste kernel: shared vector, per-domain offsets, discovered+labeled axes, a
   personalized color→emotion map, exemplars, and provenance/confidence.
4. **Apply** (later): compose a brief from the kernel + a request and generate a design for a
   requested domain (room, façade, outfit, palette).

## Interaction philosophy (hard constraint)

The human reacts and chooses; the AI proposes and articulates. Forcing verbal self-description
degrades aesthetic judgment (verbal overshadowing / "thinking too much"), so the primary signal
is always a choice. Words appear only when the AI names a discovered pattern and the user
confirms it.

## Representation (the key decision)

- `g`: shared taste vector, learned from all choices across mediums — the "general taste."
- `d_k`: a small per-domain offset. `preference_direction(domain) = g + d_k`.
- Personal **axes**: the dimensions the person's choices actually vary on, discovered (PCA /
  linear probes) then labeled by nearest concept anchors.

Rationale: cross-domain taste transfer is real but partial. A single point is too crude; fully
independent per-domain models waste the shared signal. Hierarchy captures both, and `g` gives new
domains a warm start so each needs only a few choices.

## Color-affect module

Color→emotion is personal and culturally variable — do not assume "blue = calm." Elicit it:
"which palette feels [calm/focused/energized/serious/playful] to you?" and store the person's own
emotion→palette-region map for use as a generation constraint.

## Falsifiable tests (Phase 1 definition of done)

1. **Hold-out accuracy** — fit on most of a person's choices, predict the rest; target > ~0.65
   (chance = 0.5).
2. **Cross-domain transfer** — fit `g` on domains {A,B,C}, predict held-out domain D; above
   chance validates the general-taste ambition (expect partial).
3. **Ablation** — does adding non-target domains improve prediction in a target domain vs that
   domain alone? Justifies the multi-medium design.

`app/validation/holdout.py` implements these; `backend/tests/test_geometry.py` already proves the
mechanism works on synthetic data with a known ground-truth taste vector.

## Out of scope for the MVP

Real generation, per-person LoRA training, auth, multi-user accounts, mobile. Ship the
elicitation + geometry + validation loop first.

## Honest limits

Preference model, not a personality test. Cross-domain transfer is partial by design. Taste
drifts with mood/context/time — keep the kernel living and recent-weighted. Cold start needs a
minimum number of choices before predictions are trustworthy; always surface confidence.
