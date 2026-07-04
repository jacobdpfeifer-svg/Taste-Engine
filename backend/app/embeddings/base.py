from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingAdapter(ABC):
    """Turns a stimulus (identified by id, optionally with an image path and/or text) into a
    vector.

    All embedding access in the system goes through this interface so the backend can be
    swapped (stub -> open_clip -> hosted API) without touching geometry or routes.

    `text` is the stimulus caption/tags. Adapters that can only embed images may ignore it;
    the stub uses it so that semantically similar stimuli land near each other even with no
    model downloads.
    """

    dim: int

    @abstractmethod
    def embed(
        self, stimulus_id: str, image_path: str | None = None, text: str | None = None
    ) -> np.ndarray:
        """Return an L2-normalized vector of shape (dim,)."""
        ...

    def embed_many(
        self, items: list[tuple[str, str | None, str | None]]
    ) -> dict[str, np.ndarray]:
        return {sid: self.embed(sid, path, text) for sid, path, text in items}


def get_adapter() -> EmbeddingAdapter:
    from app.config import config

    if config.embedding_backend == "openclip":
        from app.embeddings.openclip_adapter import OpenClipAdapter

        return OpenClipAdapter()
    from app.embeddings.stub_adapter import StubAdapter

    return StubAdapter(dim=config.embedding_dim)
