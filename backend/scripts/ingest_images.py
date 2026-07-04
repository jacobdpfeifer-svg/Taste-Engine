"""Build data/stimuli.json from folders of real photographs.

Usage (from backend/):
    1. Put images in per-domain folders:
         data/stimuli/interior/*.jpg   data/stimuli/apparel/*.png   ...
       Folder name = domain. Any of: interior, exterior, apparel, object, art,
       landscape, typography, palette (extensible — any folder name works).
    2. Optionally name files descriptively ("warm-oak-reading-nook.jpg"): the filename
       becomes the caption. Captions are only metadata/fallback display — with
       EMBEDDING_BACKEND=openclip the *image pixels* are what gets embedded.
    3. Run:  python scripts/ingest_images.py
    4. Point the backend at real embeddings:  EMBEDDING_BACKEND=openclip uvicorn app.main:app

Existing entries in data/stimuli.json for domains WITHOUT a photo folder are kept, so you can
replace the bank one domain at a time (photos take precedence over generated SVGs per domain).
"""
from __future__ import annotations

import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
STIMULI_DIR = Path("data/stimuli")
BANK_PATH = Path("data/stimuli.json")
MIN_PER_DOMAIN = 40  # Phase 1 target; warn below this


def caption_from_filename(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").strip()


def main() -> None:
    if not STIMULI_DIR.is_dir():
        raise SystemExit(f"{STIMULI_DIR} does not exist — create per-domain folders inside it")

    photo_domains: dict[str, list[Path]] = {}
    for sub in sorted(STIMULI_DIR.iterdir()):
        if not sub.is_dir():
            continue
        images = sorted(p for p in sub.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if images:
            photo_domains[sub.name] = images

    if not photo_domains:
        raise SystemExit(
            f"no photos found — put images in {STIMULI_DIR}/<domain>/ "
            "(e.g. data/stimuli/interior/warm-oak-nook.jpg)"
        )

    # Keep existing entries for domains that don't have photo folders yet.
    existing: list[dict] = []
    if BANK_PATH.exists():
        existing = [
            r for r in json.loads(BANK_PATH.read_text()) if r["domain"] not in photo_domains
        ]

    bank: list[dict] = list(existing)
    for domain, images in photo_domains.items():
        if len(images) < MIN_PER_DOMAIN:
            print(f"WARNING: {domain} has {len(images)} images; Phase 1 wants >= {MIN_PER_DOMAIN}")
        for i, img in enumerate(images):
            bank.append({
                "id": f"{domain[:3]}_{i:03d}",
                "domain": domain,
                "path": str(img).replace("\\", "/"),
                "caption": caption_from_filename(img),
                "tags": [],
            })
        print(f"{domain}: {len(images)} photos")

    ids = [r["id"] for r in bank]
    assert len(ids) == len(set(ids)), "duplicate stimulus ids — check domain prefixes"

    BANK_PATH.write_text(json.dumps(bank, indent=2))
    kept = f" (kept {len(existing)} existing entries in other domains)" if existing else ""
    print(f"wrote {BANK_PATH} with {len(bank)} stimuli{kept}")
    print("next: EMBEDDING_BACKEND=openclip uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
