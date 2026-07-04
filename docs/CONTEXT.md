# CONTEXT — Project origin, research dossier, and design rationale

A hand-off record of where this project came from and the research behind its design decisions.
Read `SPEC.md` for *what* to build and `TASKS.md` for *in what order*; read this for *why*.

---

## 1. One-paragraph summary

The project began as a tool to generate **clothing-brand mockups** from plain-English requests
run through a pre-encoded "creative system," optimized to be token-efficient and image-first. It
evolved through three moves: (a) reframing the encoding problem as a **"creative director in a
box"**; (b) a pivot to the psychology of **capturing subjective taste without forcing verbal
description**; and (c) **generalizing beyond fashion** into a domain-agnostic "taste engine" that
learns a person's aesthetic preferences from *choices*, represents them as geometry, and applies
them across mediums (interiors, exteriors, apparel, color). It ended with a buildable, test-first
plan and this repository, whose core runs and proves its own premise on synthetic data.

## 2. How the idea evolved

1. **Original ask.** A user describes a garment in plain English; an AI produces a downloadable
   mockup by working through a pre-designed system encoding the user's creative taste — captured
   primarily via images, moods, clothing pieces, and life preferences — using as little context
   (tokens) as possible.
2. **Fashion research + first architecture.** How creative directors communicate vision, how
   tech packs and mood boards encode intent, how AI style systems stay consistent. Produced a
   three-layer model (Identity → Spec → Output) and a token-efficient "brand bible" scaffold
   using progressive disclosure. Key feasibility fork: the AI composes briefs/specs; photoreal
   pixels require an external image model.
3. **The taste-capture pivot.** How to make subjective taste reproducible *without* forcing
   verbal description. Researched verbal overshadowing, music/personality, Openness/aesthetics,
   embeddings, and pairwise/active-learning elicitation. Result: the principle **"the human
   reacts and chooses; the AI proposes and articulates,"** and the "taste kernel" concept.
4. **Generalization.** Leave fashion; test whether the system works *in general*. Centered on a
   hierarchical representation (shared taste vector + per-domain offsets) and a validation-first
   stance.
5. **This repo.** Rules files, ordered task list, reference docs, and a runnable backend whose
   tests prove the core mechanism.

## 3. Core concepts and the evidence behind them

### 3.1 Interaction philosophy — choice over description
Forcing people to verbalize an aesthetic judgment can *degrade* it. Grounded in **verbal
overshadowing** (Schooler & Engstler-Schooler 1990: describing a face/color/wine impairs later
recognition/judgment — a disruption of non-reportable holistic processing) and **"thinking too
much"** (Wilson & Schooler 1991, *JPSP*: introspecting about reasons reduces preference quality;
Wilson et al. 1993: and post-choice satisfaction). Design consequence: the primary signal is
always a **choice between shown options**; any words are AI-authored and user-ratified, never
user free-text.

### 3.2 Representation — taste as geometry
Images become vectors (CLIP-style embeddings; Radford et al. 2021); a person's taste is a
**location + direction** in that space, not a paragraph. Similar styles cluster in latent space;
linear projections can disentangle style from content and expose interpretable axes.
Reproducibility = a coordinate + exemplars, applied identically to any new item.

### 3.3 Elicitation mechanics — adaptive pairwise + triads
Ratings are cognitively unreliable and inconsistent; **pairwise comparison** is lower-load and
more reliable (Bradley & Terry 1952; Bradley-Terry-Luce). Each choice constrains the preference
point; **active learning** picks the most informative next comparison so the model converges in
few questions. **Triads ("odd one out")** are denser signal per screen and reveal which
dimensions the person sorts on — input for Phase 2's axis discovery.

### 3.4 Cross-domain triangulation
A shared aesthetic disposition — largely **Openness to Experience**, plus preferences for
complexity/warmth/valence — underlies taste across mediums. Music-preference research (Rentfrow &
Gosling 2003; Rentfrow, Goldberg & Levitin 2011; Nave et al. 2018) shows revealed preference has
stable structure and generalizes. Sampling multiple domains captures the *generator* of taste.

### 3.5 The key architectural decision — hierarchy
Cross-domain transfer is **real but partial**. Model taste as shared vector `g` plus per-domain
offset `d_k`: `preference(domain) = g + d_k`. `g` gives each new domain a warm start; offsets
respect genuine domain independence (loving brutalist architecture and soft clothing).

### 3.6 The taste kernel
Per person: shared vector, per-domain offsets, discovered-and-labeled axes, a personalized
color→emotion map, exemplar sets (image refs + captions), AI-written user-confirmed principles,
and provenance/confidence. Stored with progressive disclosure; images live as references, never
inlined into context.

### 3.7 Affective color mapping
Color→emotion has broad tendencies but strong individual/cultural variance, so it is **learned
per person** rather than assumed, and used as a generation constraint. (Phase 3.)

### 3.8 Output / generation (Phase 4)
Compose a brief from the kernel + a request, then route to a generator. Two pipelines:
**deterministic compositing** (graphic on a template — pixel-exact, repeatable, cheap) and
**generative** (photoreal scenes conditioned on exemplars; per-person LoRA strongest at scale).
Lesson from tech packs: images get misinterpreted without text callouts — every generated output
ships with a rationale.

### 3.9 Epistemics (non-negotiable)
A **preference model, not a personality diagnosis**. Transfer is partial by design; taste drifts
with mood/context/time (keep the kernel living and recent-weighted); cold start needs a minimum
number of choices. Always surface confidence and validate on the person's own held-out choices.

### 3.10 Validation-first
Three falsifiable tests gate the build: hold-out accuracy, cross-domain transfer, ablation.
Build the test before the product. The synthetic pass validates the *mechanism*, not the *claim
about real people* — the first real milestone is the three tests on genuine embeddings + choices.

## 4. Open technical choices

- Embedding model: CLIP vs SigLIP vs DINOv2 (adapter interface keeps this swappable).
- Generative backend: hosted API vs Flux/SD + per-person LoRA.
- Proper partial pooling for the hierarchical fit (`geometry/hierarchy.py` is a stub).
- Cold-start threshold: how many choices before the kernel is trustworthy.

## 5. Source dossier (selected, by theme)

**[P]** = peer-reviewed/foundational; **[S]** = industry/illustrative (verify before relying on).

### Verbal overshadowing & introspection harming preference
- [P] Schooler Lab (UCSB) — overview: https://labs.psych.ucsb.edu/schooler/jonathan/research/verbal-overshadowing
- [P] Dodson, Johnson & Schooler (1997), *Memory & Cognition*: https://memlab.yale.edu/sites/default/files/files/1997_Dodson_Johnson_Schooler_MemCog.pdf
- [P] Ryan & Schooler (1998) — individual differences in VO: https://faculty.kutztown.edu/rryan/RESEARCH/pubs/Ryan%20&%20Schooler%201998%20Indiv%20diffs%20in%20VO.pdf
- [S] Wikipedia — Verbal overshadowing: https://en.wikipedia.org/wiki/Verbal_overshadowing

### Music preference & personality (cross-domain signal)
- [P] Rentfrow & Gosling (2003), *JPSP*: https://gosling.psy.utexas.edu/wp-content/uploads/2014/09/JPSP03musicdimensions.pdf
- [P] STOMP scale: https://gosling.psy.utexas.edu/scales-weve-developed/short-test-of-music-preferences-stomp/
- [P] Rentfrow et al. (2011), five-factor MUSIC model: https://pmc.ncbi.nlm.nih.gov/articles/PMC3138530/

### Openness to Experience & aesthetic sensitivity
- [P] Myszkowski et al., *PAID* — aesthetic sensitivity: https://www.sciencedirect.com/science/article/abs/pii/S0191886913013342
- [P] *Frontiers in Psychology* — Openness predicts novelty sensitivity: https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.01877/full
- [P] PMC — Big Five, aesthetic judgment styles, art interest: https://pmc.ncbi.nlm.nih.gov/articles/PMC6266521/

### Embeddings / CLIP / latent style space
- [P] arXiv — Does CLIP perceive art the same way we do?: https://arxiv.org/pdf/2505.05229
- [P] arXiv — Everyone Can Be Picasso (CLIP latent + aesthetics): https://arxiv.org/pdf/2304.07999
- [P] arXiv — Zero-shot visual concept blending (linear disentanglement): https://arxiv.org/pdf/2503.21277
- [P] arXiv — SCFlow: style/content disentanglement: https://arxiv.org/pdf/2508.03402

### Preference elicitation / pairwise comparison / active learning
- [P] arXiv — Bayes-optimal entropy pursuit for active choice-based preference learning: https://arxiv.org/pdf/1702.07694
- [P] arXiv — Active utility-based pairwise sampling: https://arxiv.org/html/2508.14911v1
- [P] arXiv — Robust ordinal regression for subset comparisons: https://arxiv.org/pdf/2308.03376
- [P] USPTO 12,499,178 — preference point via paired comparisons: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12499178

### Context engineering & progressive disclosure
- [P] Anthropic — Effective context engineering for AI agents: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

### Fashion practice (first instantiation; industry sources)
- [S] Tech packs: https://techpacker.com/blog/design/what-is-a-tech-pack/ — key lesson: never
  let an image travel without words (label every component).
- [S] Mood boards: https://glamobserver.com/how-to-build-a-fashion-mood-board/
- [S] AI style consistency (reference conditioning, style elements, design tokens):
  https://getimg.ai/features/custom-ai-styles

## 6. Caveats for whoever continues

- Source quality varies; [S] sources illustrate industry practice, they are not evidence.
- Several sources carry 2025–2026 datestamps; verify currency if precise figures matter.
- The premise is unproven on real data until the Phase 1 gate is run with a real person, real
  images, and real embeddings.
- Ethical guardrail: preference model, never a personality diagnosis; surface confidence.
