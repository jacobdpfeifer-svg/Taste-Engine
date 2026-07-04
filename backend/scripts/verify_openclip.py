"""Smoke-test the OpenCLIP embedding backend end to end (run after installing the optional
deps: pip install open_clip_torch torch pillow).

Creates two tiny test images (a warm one and a cool one), embeds them plus two captions, and
checks the geometry is sane: image embeddings should match their own description better than
the opposite one. First run downloads the model weights (~600 MB) from HuggingFace.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `app` importable
os.environ["EMBEDDING_BACKEND"] = "openclip"

import numpy as np
from PIL import Image

from app.embeddings.openclip_adapter import OpenClipAdapter


def main() -> None:
    print("loading OpenCLIP model (first run downloads weights)...")
    adapter = OpenClipAdapter()
    print(f"model loaded, dim={adapter.dim}")

    tmp = Path(tempfile.mkdtemp())
    warm_path = tmp / "warm.png"
    cool_path = tmp / "cool.png"
    Image.new("RGB", (224, 224), (220, 120, 40)).save(warm_path)   # orange
    Image.new("RGB", (224, 224), (40, 90, 200)).save(cool_path)    # blue

    warm_img = adapter.embed("warm_img", str(warm_path))
    cool_img = adapter.embed("cool_img", str(cool_path))
    warm_txt = adapter.embed("warm_txt", None, "a plain warm orange colored image")
    cool_txt = adapter.embed("cool_txt", None, "a plain cool blue colored image")

    for v in (warm_img, cool_img, warm_txt, cool_txt):
        assert v.shape == (adapter.dim,)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    ww, wc = float(warm_img @ warm_txt), float(warm_img @ cool_txt)
    cc, cw = float(cool_img @ cool_txt), float(cool_img @ warm_txt)
    print(f"warm image ~ warm text: {ww:.3f}   warm image ~ cool text: {wc:.3f}")
    print(f"cool image ~ cool text: {cc:.3f}   cool image ~ warm text: {cw:.3f}")
    assert ww > wc, "warm image should match warm caption better"
    assert cc > cw, "cool image should match cool caption better"
    print("OK — OpenCLIP backend is working. Run the server with EMBEDDING_BACKEND=openclip")


if __name__ == "__main__":
    main()
