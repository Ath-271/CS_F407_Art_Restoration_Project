"""Import a folder-per-style image dataset (e.g. the Kaggle Indian paintings set).

    python scripts/import_dataset.py "C:/Users/<you>/Downloads/archive"

- reads jpg/png/webp/gif/ppm/..., converts to PNG
- drops images whose short side < --min-side
- removes near-duplicates (perceptual dHash; Google-scrape sets have many)
- writes data/pool/<style>__<name>.png  and  data/pool/manifest.csv
- builds contact sheets (outputs/contact_sheets/) so you can pick the
  genuinely DAMAGED paintings; then run scripts/split_pool.py

The source folder is only read, never modified.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageOps

from restoration.utils import imwrite

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".ppm", ".bmp", ".tif", ".tiff", ".jpg_large"}


def dhash(img: np.ndarray, size: int = 8) -> int:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def contact_sheets(items, out_dir: Path, per_sheet=48, cols=8, thumb=200):
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    for s in range(0, len(items), per_sheet):
        chunk = items[s : s + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = np.full((rows * (thumb + 22), cols * thumb, 3), 255, np.uint8)
        for i, (idx, img) in enumerate(chunk):
            h, w = img.shape[:2]
            f = thumb / max(h, w)
            t = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
            r, c = divmod(i, cols)
            y0, x0 = r * (thumb + 22) + 20, c * thumb
            sheet[y0 : y0 + t.shape[0], x0 : x0 + t.shape[1]] = t
            cv2.putText(sheet, str(idx), (x0 + 3, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 0, 0), 2, cv2.LINE_AA)
        imwrite(out_dir / f"sheet_{s // per_sheet:02d}.png", sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=str(ROOT / "data/pool"))
    ap.add_argument("--min-side", type=int, default=400)
    ap.add_argument("--dup-bits", type=int, default=6, help="max Hamming distance to count as duplicate")
    ap.add_argument("--append", action="store_true", help="add to an existing pool, keeping existing ids")
    ap.add_argument("--prefix", default="", help="prefix for style names, e.g. hf_")
    ap.add_argument("--sheets", default=None, help="contact sheet folder (default outputs/contact_sheets[_<prefix>])")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    kept, hashes, skipped = [], [], {"small": 0, "dup": 0, "unreadable": 0}
    old_rows, old_hashes = [], []
    if a.append and (out / "manifest.csv").exists():
        old_rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
        for r in old_rows:
            old_hashes.append(dhash(np.array(Image.open(out / r["file"]).convert("RGB"))))
    for p in files:
        try:
            im = Image.open(p)
            im.seek(0)
            img = np.array(ImageOps.exif_transpose(im).convert("RGB"))
        except Exception:
            skipped["unreadable"] += 1
            continue
        if min(img.shape[:2]) < a.min_side:
            skipped["small"] += 1
            continue
        hsh = dhash(img)
        if any(bin(h ^ hsh).count("1") <= a.dup_bits for h in old_hashes):
            skipped["dup"] += 1
            continue
        # duplicates: keep the higher-resolution copy
        dup = next((i for i, h in enumerate(hashes) if bin(h ^ hsh).count("1") <= a.dup_bits), None)
        style = a.prefix + p.parent.name.replace(" painting", "").replace(" art drawing", "").replace(" ", "_")
        rec = {"style": style, "source": str(p.relative_to(src)), "h": img.shape[0], "w": img.shape[1], "img": img}
        if dup is not None:
            skipped["dup"] += 1
            if img.shape[0] * img.shape[1] > kept[dup]["h"] * kept[dup]["w"]:
                kept[dup], hashes[dup] = rec, hsh
            continue
        kept.append(rec)
        hashes.append(hsh)

    start = len(old_rows)
    with open(out / "manifest.csv", "a" if old_rows else "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        if not old_rows:
            wr.writerow(["id", "file", "style", "source", "width", "height"])
        for i, r in enumerate(kept, start=start):
            name = f"{i:04d}_{r['style']}.png"
            imwrite(out / name, r["img"])
            wr.writerow([i, name, r["style"], r["source"], r["w"], r["h"]])
    print(f"{len(files)} files -> kept {len(kept)} | skipped {skipped}")
    by = {}
    for r in kept:
        by[r["style"]] = by.get(r["style"], 0) + 1
    print("per style:", by)
    sheets = Path(a.sheets) if a.sheets else ROOT / ("outputs/contact_sheets" + (f"_{a.prefix.strip('_')}" if a.prefix else ""))
    contact_sheets([(i, r["img"]) for i, r in enumerate(kept, start=start)], sheets)
    print(f"ids {start}..{start + len(kept) - 1} | contact sheets in {sheets}")


if __name__ == "__main__":
    main()
