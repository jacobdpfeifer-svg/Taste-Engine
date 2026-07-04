"""Generate a placeholder stimulus bank: data/stimuli.json + SVG images in data/stimuli/.

The SVGs are not photos. Each one encodes exactly THREE style axes, rendered as saliently as
possible so a human choosing by look produces choices the tag-based stub embedding can learn:

  warm / cool          — unmistakable orange-red vs blue color family
  minimal / ornate     — 2-3 large shapes vs ~22 small ones
  geometric / organic  — axis-aligned rectangles/triangles vs rotated irregular blobs

Earlier versions encoded six axes with subtle randomized rendering; humans couldn't perceive
half of them, so their visual choices looked like noise to the stub. Fewer, louder axes make
the synthetic mode a fairer dry run. For the real Phase 1 gate use real photographs
(scripts/ingest_images.py) with EMBEDDING_BACKEND=openclip.

Run from backend/:  python scripts/generate_stimuli.py
Deterministic: same ids -> same images, every run.
"""
from __future__ import annotations

import colorsys
import json
import math
import random
from pathlib import Path

W, H = 480, 360
N_PER_DOMAIN = 44

NOUNS = {
    "interior": ["living room", "kitchen", "bedroom", "studio", "reading nook",
                 "hallway", "loft", "dining room", "bathroom", "den"],
    "exterior": ["facade", "cabin", "courtyard", "garden", "porch",
                 "rooftop", "entryway", "pool house", "townhouse", "terrace"],
    "apparel": ["jacket", "coat", "dress", "shirt", "trousers",
                "sweater", "boots", "bag", "scarf", "suit"],
    "object": ["chair", "lamp", "vase", "table", "clock",
               "kettle", "shelf", "mirror", "rug", "bowl"],
    "art": ["abstract print", "landscape painting", "line drawing", "collage",
            "poster", "still life", "mural study", "photograph", "etching", "relief"],
    "palette": ["neutral palette", "pastel palette", "jewel-tone palette", "earthy palette",
                "monochrome palette", "primary palette", "dusk palette", "clay palette",
                "sea palette", "forest palette"],
}
PREFIX = {"interior": "int", "exterior": "ext", "apparel": "app",
          "object": "obj", "art": "art", "palette": "pal"}

AXES = [("warm", "cool"), ("minimal", "ornate"), ("geometric", "organic")]


def hsv(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360, min(max(s, 0), 1), min(max(v, 0), 1))
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def pick_attrs(rng: random.Random) -> dict[str, str]:
    return {"temp": rng.choice(AXES[0]), "complexity": rng.choice(AXES[1]),
            "form": rng.choice(AXES[2])}


def _hue(temp: str, rng: random.Random) -> float:
    # Tight hue bands so warm vs cool is never ambiguous.
    return rng.uniform(18, 38) if temp == "warm" else rng.uniform(205, 230)


def _geometric_shape(cx: float, cy: float, size: float, fill: str, rng: random.Random) -> str:
    if rng.random() < 0.6:
        w, h = size, size * rng.uniform(0.6, 1.4)
        return f'<rect x="{cx - w / 2:.0f}" y="{cy - h / 2:.0f}" width="{w:.0f}" height="{h:.0f}" fill="{fill}"/>'
    pts = " ".join(
        f"{cx + size * 0.7 * math.cos(math.radians(ang)):.0f},"
        f"{cy + size * 0.7 * math.sin(math.radians(ang)):.0f}"
        for ang in (270, 30, 150))
    return f'<polygon points="{pts}" fill="{fill}"/>'


def _organic_shape(cx: float, cy: float, size: float, fill: str, rng: random.Random) -> str:
    # An irregular blob: two overlapping rotated ellipses read as clearly non-geometric.
    parts = []
    for _ in range(2):
        rx = size * rng.uniform(0.5, 0.8)
        ry = size * rng.uniform(0.3, 0.6)
        rot = rng.uniform(0, 180)
        dx, dy = rng.uniform(-size * 0.2, size * 0.2), rng.uniform(-size * 0.2, size * 0.2)
        parts.append(
            f'<ellipse cx="{cx + dx:.0f}" cy="{cy + dy:.0f}" rx="{rx:.0f}" ry="{ry:.0f}" '
            f'fill="{fill}" transform="rotate({rot:.0f} {cx + dx:.0f} {cy + dy:.0f})"/>')
    return "".join(parts)


def render_svg(a: dict[str, str], domain: str, rng: random.Random) -> str:
    hue = _hue(a["temp"], rng)
    bg = hsv(hue, 0.25, 0.92)
    body: list[str] = [f'<rect width="{W}" height="{H}" fill="{bg}"/>']

    if domain == "palette":
        # Swatch stripes in the same unmistakable family. minimal = 3 wide, ornate = 9 narrow;
        # geometric = hard vertical stripes, organic = overlapping round swatches.
        n = 3 if a["complexity"] == "minimal" else 9
        for i in range(n):
            h_i = _hue(a["temp"], rng)
            fill = hsv(h_i, rng.uniform(0.55, 0.8), rng.uniform(0.45, 0.85))
            if a["form"] == "geometric":
                body.append(f'<rect x="{i * W / n:.0f}" y="0" width="{W / n + 1:.0f}" height="{H}" fill="{fill}"/>')
            else:
                cx = (i + 0.5) * W / n
                body.append(_organic_shape(cx, H / 2, W / n * 1.1, fill, rng))
    else:
        if a["complexity"] == "minimal":
            n_shapes, size_lo, size_hi = rng.randint(2, 3), 70, 130
        else:
            n_shapes, size_lo, size_hi = rng.randint(20, 24), 18, 42
        for _ in range(n_shapes):
            fill = hsv(_hue(a["temp"], rng), rng.uniform(0.55, 0.85), rng.uniform(0.35, 0.75))
            cx, cy = rng.uniform(30, W - 30), rng.uniform(30, H - 30)
            size = rng.uniform(size_lo, size_hi)
            if a["form"] == "geometric":
                body.append(_geometric_shape(cx, cy, size, fill, rng))
            else:
                body.append(_organic_shape(cx, cy, size, fill, rng))

    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}">{"".join(body)}</svg>')


def main() -> None:
    out_dir = Path("data/stimuli")
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = []
    for domain, prefix in PREFIX.items():
        for i in range(N_PER_DOMAIN):
            sid = f"{prefix}_{i:03d}"
            rng = random.Random(sid)
            a = pick_attrs(rng)
            noun = rng.choice(NOUNS[domain])
            path = f"data/stimuli/{sid}.svg"
            (out_dir / f"{sid}.svg").write_text(render_svg(a, domain, rng))
            bank.append({
                "id": sid,
                "domain": domain,
                "path": path,
                "caption": f'{a["temp"]} {a["form"]} {a["complexity"]} {noun}',
                "tags": [a["temp"], a["complexity"], a["form"]],
            })
    Path("data/stimuli.json").write_text(json.dumps(bank, indent=2))
    print(f"wrote data/stimuli.json ({len(bank)} stimuli) and {len(bank)} SVGs to data/stimuli/")


if __name__ == "__main__":
    main()
