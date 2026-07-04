from __future__ import annotations

from pathlib import Path

import numpy as np

from app.embeddings.base import EmbeddingAdapter


class OpenClipAdapter(EmbeddingAdapter):
    """Real embeddings via open_clip (CLIP image encoder; text encoder as fallback when a
    stimulus has no image on disk).

    Requires the optional dependencies: `pip install open_clip_torch torch pillow`.
    Return contract is identical to StubAdapter: L2-normalized np.ndarray of shape (dim,).
    Embeddings are cached by stimulus_id for the lifetime of the adapter.
    """

    def __init__(self) -> None:
        try:
            import open_clip  # noqa: F401
            import torch  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on optional installs
            raise RuntimeError(
                "EMBEDDING_BACKEND=openclip requires optional deps: "
                "pip install open_clip_torch torch pillow"
            ) from e

        from app.config import config

        self._torch = torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.openclip_model, pretrained=config.openclip_pretrained
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(config.openclip_model)
        self._cache: dict[str, np.ndarray] = {}
        self.dim = int(model.visual.output_dim)

    def embed(
        self, stimulus_id: str, image_path: str | None = None, text: str | None = None
    ) -> np.ndarray:
        if stimulus_id in self._cache:
            return self._cache[stimulus_id]

        with self._torch.no_grad():
            feat = None
            if image_path and Path(image_path).is_file():
                try:
                    from PIL import Image

                    img = self._preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
                    feat = self._model.encode_image(img)
                except Exception:
                    # Unreadable format (e.g. placeholder SVGs) — fall through to text.
                    feat = None
            if feat is None and text:
                feat = self._model.encode_text(self._tokenizer([text]))
            if feat is None:
                raise ValueError(
                    f"stimulus {stimulus_id!r}: need a readable image or a caption to embed"
                )

        v = feat[0].cpu().numpy().astype(np.float64)
        n = np.linalg.norm(v)
        v = v / n if n else v
        self._cache[stimulus_id] = v
        return v
