from __future__ import annotations

from pydantic import BaseModel


class StimulusOut(BaseModel):
    id: str
    domain: str
    path: str
    caption: str
    tags: list[str]
    # embeddings are intentionally omitted from API payloads


class CreateSessionIn(BaseModel):
    person_id: str = "anon"


class CreateSessionOut(BaseModel):
    session_id: str


class PairOut(BaseModel):
    a: StimulusOut
    b: StimulusOut


class GridOut(BaseModel):
    items: list[StimulusOut]


class TriadOut(BaseModel):
    a: StimulusOut
    b: StimulusOut
    c: StimulusOut


class ChoiceIn(BaseModel):
    session_id: str
    chosen_id: str
    rejected_id: str
    domain: str = "cross"
    kind: str = "pair"


class RecordIn(BaseModel):
    chosen_id: str
    rejected_id: str
    domain: str = "cross"
    kind: str = "pair"
    weight: float = 1.0


class BatchIn(BaseModel):
    session_id: str
    records: list[RecordIn]


class TriadChoiceIn(BaseModel):
    """Odd-one-out answer: `odd_id` felt least like the person; the two `other_ids` were kept.
    Decomposed server-side into two pairwise choice rows (kind="triad")."""

    session_id: str
    odd_id: str
    other_ids: list[str]
    domain: str = "cross"


class ProgressOut(BaseModel):
    """Stop-condition signal for the frontend: keep eliciting until `done`."""

    n_choices: int
    holdout_accuracy: float | None
    done: bool
    reason: str


class FitOut(BaseModel):
    n_choices: int
    holdout_accuracy: float | None
    kernel_path: str | None


class MetricOut(BaseModel):
    """An accuracy with its evidence: held-out sample size and Wilson 95% interval."""

    accuracy: float
    n_test: int
    low: float
    high: float


class ValidateOut(BaseModel):
    holdout: MetricOut | None
    holdout_exemplar: MetricOut | None
    best_model: str
    cross_domain: dict[str, MetricOut]
    ablation: dict[str, dict[str, MetricOut]]
    n_choices: int
    per_domain_counts: dict[str, int]


class ReflectOut(BaseModel):
    hero_ids: list[str]
    top_tags: list[str]
    summary: str


class MetaOut(BaseModel):
    provenance_banner: str
    embedding_backend: str
