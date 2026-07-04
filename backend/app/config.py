from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_stimuli_path() -> str:
    env = os.getenv("STIMULI_PATH")
    if env:
        return env
    # Prefer the generated/curated bank when present; fall back to the tiny sample.
    if Path("data/stimuli.json").exists():
        return "data/stimuli.json"
    return "data/stimuli.sample.json"


@dataclass(frozen=True)
class Config:
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "stub")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "64"))
    openclip_model: str = os.getenv("OPENCLIP_MODEL", "ViT-B-32")
    openclip_pretrained: str = os.getenv("OPENCLIP_PRETRAINED", "laion2b_s34b_b79k")
    db_path: str = os.getenv("DB_PATH", "taste.sqlite3")
    stimuli_path: str = field(default_factory=_default_stimuli_path)
    kernels_dir: str = os.getenv("KERNELS_DIR", "kernels")


config = Config()
