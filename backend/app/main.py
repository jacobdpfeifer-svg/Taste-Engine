from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import db
from app.config import config
from app.elicitation.active import diverse_sample, next_pair, next_triad, should_stop
from app.embeddings.base import get_adapter
from app.geometry.fit import Choice, TasteModel, fit_taste
from app.kernel import build_kernel, save_kernel
from app.schemas import (
    BatchIn,
    ChoiceIn,
    MetricOut,
    CreateSessionIn,
    CreateSessionOut,
    FitOut,
    GridOut,
    MetaOut,
    PairOut,
    ProgressOut,
    ReflectOut,
    StimulusOut,
    TriadChoiceIn,
    TriadOut,
    ValidateOut,
)
from app.stimuli.loader import load_bank
from app.validation.holdout import (
    ablation_metric,
    cross_domain_accuracy,
    cross_domain_metric,
    holdout_metric,
    holdout_report_exemplar,
    repeated_holdout_accuracy,
)

MIN_CHOICES_TO_FIT = 8

app = FastAPI(title="Taste Engine")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

adapter = get_adapter()
bank = load_bank(config.stimuli_path, adapter)
db.init_db()

# Serve stimulus images so the frontend can render them at /api/data/stimuli/<file>.
_stimuli_dir = Path("data/stimuli")
if _stimuli_dir.is_dir():
    app.mount("/data/stimuli", StaticFiles(directory=str(_stimuli_dir)), name="stimuli")


def _stim_out(sid: str) -> StimulusOut:
    s = bank.items[sid]
    return StimulusOut(id=s.id, domain=s.domain, path=s.path, caption=s.caption, tags=s.tags)


def _require_session(session_id: str) -> str:
    row = db.get_session(session_id)
    if row is None:
        raise HTTPException(404, f"unknown session {session_id!r}")
    return row["person_id"]


def _current_model(session_id: str) -> tuple[TasteModel | None, list[Choice], list]:
    rows = db.get_choices(session_id)
    choices = [(r["chosen_id"], r["rejected_id"]) for r in rows]
    model = fit_taste(choices, bank.embeddings) if len(choices) >= MIN_CHOICES_TO_FIT else None
    return model, choices, rows


def _holdout(choices: list[Choice]) -> float | None:
    if len(choices) < MIN_CHOICES_TO_FIT:
        return None
    return repeated_holdout_accuracy(choices, bank.embeddings)


def _reflect(session_id: str) -> ReflectOut:
    """Read-only mirror of the person's choices: which stimuli they kept vs rejected, and
    which tags those heroes share. AI-authored articulation the user can ratify (Phase 2)."""
    rows = db.get_choices(session_id)
    scores: dict[str, float] = defaultdict(float)
    for r in rows:
        w = float(r["weight"])
        scores[r["chosen_id"]] += w
        scores[r["rejected_id"]] -= w

    hero_ids = [
        sid for sid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:5]
    ]

    tag_counts: Counter[str] = Counter()
    for hid in hero_ids:
        stim = bank.items.get(hid)
        if stim:
            tag_counts.update(stim.tags)
    top_tags = [t for t, _ in tag_counts.most_common(5)]

    if top_tags:
        summary = "You keep choosing " + ", ".join(top_tags[:3]) + "."
    else:
        summary = "Not enough signal yet to name a pattern."

    return ReflectOut(hero_ids=hero_ids, top_tags=top_tags, summary=summary)


@app.get("/stimuli", response_model=list[StimulusOut])
def list_stimuli() -> list[StimulusOut]:
    return [_stim_out(sid) for sid in bank.ids()]


@app.get("/meta", response_model=MetaOut)
def get_meta() -> MetaOut:
    return MetaOut(
        provenance_banner=(
            "You're choosing, not describing. This builds a preference model from your picks — "
            "not a personality diagnosis. Confidence is measured, never assumed."
        ),
        embedding_backend=config.embedding_backend,
    )


@app.post("/sessions", response_model=CreateSessionOut)
def create_session(body: CreateSessionIn) -> CreateSessionOut:
    sid = f"sess_{uuid.uuid4().hex[:12]}"
    db.create_session(sid, body.person_id)
    return CreateSessionOut(session_id=sid)


@app.get("/sessions/{session_id}/next-pair", response_model=PairOut)
def get_next_pair(session_id: str, domain: str | None = None) -> PairOut:
    _require_session(session_id)
    model, _choices, rows = _current_model(session_id)
    shown = {frozenset((r["chosen_id"], r["rejected_id"])) for r in rows}
    ids = bank.ids(domain)
    if len(ids) < 2:
        raise HTTPException(400, "not enough stimuli in this domain")
    try:
        a, b = next_pair(ids, bank.embeddings, model=model, already_shown=shown)
    except ValueError:
        # Every pair in this domain has been shown; allow repeats rather than dead-ending.
        a, b = next_pair(ids, bank.embeddings, model=model, already_shown=set())
    return PairOut(a=_stim_out(a), b=_stim_out(b))


@app.get("/sessions/{session_id}/grid", response_model=GridOut)
def get_grid(session_id: str, k: int = 9, domain: str | None = None) -> GridOut:
    _require_session(session_id)
    ids = bank.ids(domain)
    if k < 1:
        raise HTTPException(400, "k must be at least 1")
    if not ids:
        raise HTTPException(400, "no stimuli in this domain")
    chosen = diverse_sample(ids, bank.embeddings, k)
    return GridOut(items=[_stim_out(sid) for sid in chosen])


@app.get("/sessions/{session_id}/next-triad", response_model=TriadOut)
def get_next_triad(session_id: str, domain: str | None = None) -> TriadOut:
    _require_session(session_id)
    model, _choices, _rows = _current_model(session_id)
    ids = bank.ids(domain)
    if len(ids) < 3:
        raise HTTPException(400, "not enough stimuli in this domain for a triad")
    a, b, c = next_triad(ids, bank.embeddings, model=model)
    return TriadOut(a=_stim_out(a), b=_stim_out(b), c=_stim_out(c))


@app.post("/triads")
def post_triad(body: TriadChoiceIn) -> dict[str, bool]:
    """Record an odd-one-out answer as two pairwise rows: each kept item beat the odd one."""
    _require_session(body.session_id)
    if len(body.other_ids) != 2 or body.odd_id in body.other_ids:
        raise HTTPException(400, "a triad answer is one odd_id and two distinct other_ids")
    for sid in (body.odd_id, *body.other_ids):
        if sid not in bank.items:
            raise HTTPException(400, f"unknown stimulus {sid!r}")
    for kept in body.other_ids:
        db.add_choice(body.session_id, kept, body.odd_id, body.domain, kind="triad")
    return {"ok": True}


@app.post("/choices")
def post_choice(body: ChoiceIn) -> dict[str, bool]:
    _require_session(body.session_id)
    for sid in (body.chosen_id, body.rejected_id):
        if sid not in bank.items:
            raise HTTPException(400, f"unknown stimulus {sid!r}")
    db.add_choice(body.session_id, body.chosen_id, body.rejected_id, body.domain, body.kind)
    return {"ok": True}


@app.post("/choices/batch")
def post_choices_batch(body: BatchIn) -> dict[str, bool | int]:
    _require_session(body.session_id)
    for rec in body.records:
        for sid in (rec.chosen_id, rec.rejected_id):
            if sid not in bank.items:
                raise HTTPException(400, f"unknown stimulus {sid!r}")
    for rec in body.records:
        db.add_choice(
            body.session_id,
            rec.chosen_id,
            rec.rejected_id,
            rec.domain,
            rec.kind,
            rec.weight,
        )
    return {"ok": True, "n": len(body.records)}


@app.get("/sessions/{session_id}/progress", response_model=ProgressOut)
def progress(session_id: str) -> ProgressOut:
    """Stop-condition check the frontend polls between choices."""
    _require_session(session_id)
    rows = db.get_choices(session_id)
    choices = [(r["chosen_id"], r["rejected_id"]) for r in rows]
    acc = _holdout(choices)
    done, reason = should_stop(len(choices), acc)
    return ProgressOut(n_choices=len(choices), holdout_accuracy=acc, done=done, reason=reason)


@app.post("/sessions/{session_id}/fit", response_model=FitOut)
def fit(session_id: str) -> FitOut:
    person_id = _require_session(session_id)
    model, choices, rows = _current_model(session_id)
    acc = _holdout(choices)

    kernel_path: str | None = None
    if model is not None:
        by_domain: dict[str, list[Choice]] = defaultdict(list)
        for r in rows:
            by_domain[r["domain"]].append((r["chosen_id"], r["rejected_id"]))
        confidence = {
            d: cross_domain_accuracy(by_domain, bank.embeddings, d)
            for d in by_domain
            if d != "cross" and len(by_domain[d]) >= 3
        }
        kernel = build_kernel(
            person_id=person_id,
            model=model,
            choices=choices,
            choices_by_domain=dict(by_domain),
            ids_by_domain=bank.ids_by_domain(),
            emb=bank.embeddings,
            holdout_accuracy=acc,
            per_domain_confidence=confidence,
        )
        kernel_path = save_kernel(kernel, config.kernels_dir)

    return FitOut(n_choices=len(choices), holdout_accuracy=acc, kernel_path=kernel_path)


@app.get("/sessions/{session_id}/validate", response_model=ValidateOut)
def validate(session_id: str) -> ValidateOut:
    _require_session(session_id)
    rows = db.get_choices(session_id)
    choices = [(r["chosen_id"], r["rejected_id"]) for r in rows]
    by_domain: dict[str, list[Choice]] = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append((r["chosen_id"], r["rejected_id"]))

    def to_out(m) -> MetricOut:
        return MetricOut(accuracy=m.accuracy, n_test=m.n_test, low=m.low, high=m.high)

    holdout_m = (
        holdout_metric(choices, bank.embeddings) if len(choices) >= MIN_CHOICES_TO_FIT else None
    )
    exemplar_m = (
        holdout_report_exemplar(choices, bank.embeddings)
        if len(choices) >= MIN_CHOICES_TO_FIT
        else None
    )
    holdout = to_out(holdout_m) if holdout_m is not None else None
    holdout_exemplar = to_out(exemplar_m) if exemplar_m is not None else None
    best_model = "linear"
    if (
        holdout_m is not None
        and exemplar_m is not None
        and exemplar_m.accuracy > holdout_m.accuracy
    ):
        best_model = "exemplar"
    cross: dict[str, MetricOut] = {}
    for d in by_domain:
        if d != "cross" and len(by_domain[d]) >= 3:
            m = cross_domain_metric(by_domain, bank.embeddings, d)
            if m is not None:
                cross[d] = to_out(m)
    abl: dict[str, dict[str, MetricOut]] = {}
    for d in by_domain:
        if d != "cross" and len(by_domain[d]) >= MIN_CHOICES_TO_FIT:
            arms = ablation_metric(by_domain, bank.embeddings, d)
            if arms is not None:
                abl[d] = {arm: to_out(m) for arm, m in arms.items()}
    return ValidateOut(
        holdout=holdout,
        holdout_exemplar=holdout_exemplar,
        best_model=best_model,
        cross_domain=cross,
        ablation=abl,
        n_choices=len(choices),
        per_domain_counts={d: len(cs) for d, cs in by_domain.items()},
    )


@app.get("/sessions/{session_id}/reflect", response_model=ReflectOut)
def reflect(session_id: str) -> ReflectOut:
    _require_session(session_id)
    return _reflect(session_id)
