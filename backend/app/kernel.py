"""Serialize the taste kernel to kernels/<person_id>.json.

Shape is defined in docs/DATA_MODEL.md. Phase 1 fills shared_vector, exemplars and provenance;
domain_offsets, axes and color_map arrive in Phases 2-3 and are stored empty until then.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.geometry.fit import Choice, Embeddings, TasteModel, rank


def build_kernel(
    person_id: str,
    model: TasteModel,
    choices: list[Choice],
    choices_by_domain: dict[str, list[Choice]],
    ids_by_domain: dict[str, list[str]],
    emb: Embeddings,
    holdout_accuracy: float | None,
    per_domain_confidence: dict[str, float],
    n_exemplars: int = 3,
) -> dict:
    exemplars = {
        domain: rank(ids, model, emb)[:n_exemplars]
        for domain, ids in ids_by_domain.items()
        if ids
    }
    return {
        "person_id": person_id,
        "shared_vector": model.weights.tolist(),
        "domain_offsets": {},  # Phase 2
        "axes": [],  # Phase 2
        "color_map": {},  # Phase 3
        "exemplars": exemplars,
        "provenance": {
            "n_choices": len(choices),
            "per_domain_n_choices": {d: len(cs) for d, cs in choices_by_domain.items()},
            "per_domain_confidence": per_domain_confidence,
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "holdout_accuracy": holdout_accuracy,
        },
    }


def save_kernel(kernel: dict, kernels_dir: str) -> str:
    out = Path(kernels_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{kernel['person_id']}.json"
    path.write_text(json.dumps(kernel, indent=2))
    return str(path)


def load_kernel(person_id: str, kernels_dir: str) -> dict | None:
    path = Path(kernels_dir) / f"{person_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def kernel_model(kernel: dict) -> TasteModel:
    return TasteModel(weights=np.asarray(kernel["shared_vector"], dtype=np.float64))
