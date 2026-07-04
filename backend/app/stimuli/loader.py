from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.embeddings.base import EmbeddingAdapter


@dataclass
class Stimulus:
    id: str
    domain: str
    path: str
    caption: str
    tags: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None


@dataclass
class StimulusBank:
    items: dict[str, Stimulus]

    @property
    def embeddings(self) -> dict[str, np.ndarray]:
        return {sid: s.embedding for sid, s in self.items.items() if s.embedding is not None}

    def ids(self, domain: str | None = None) -> list[str]:
        return [
            sid for sid, s in self.items.items() if domain is None or s.domain == domain
        ]

    def ids_by_domain(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for sid, s in self.items.items():
            out.setdefault(s.domain, []).append(sid)
        return out


def load_bank(path: str, adapter: EmbeddingAdapter) -> StimulusBank:
    raw = json.loads(Path(path).read_text())
    items: dict[str, Stimulus] = {}
    for r in raw:
        s = Stimulus(
            id=r["id"],
            domain=r["domain"],
            path=r.get("path", ""),
            caption=r.get("caption", ""),
            tags=r.get("tags", []),
        )
        text = " ".join([s.caption, *s.tags]).strip() or None
        s.embedding = adapter.embed(s.id, s.path or None, text)
        items[s.id] = s
    return StimulusBank(items=items)
