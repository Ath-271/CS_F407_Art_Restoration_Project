"""Prepare the HELD-OUT test set (Mandawa) for labelling an answer key.

    python scripts/make_test_tiles.py

* Groups photos that show the same panel (same photo number, plus visual
  near-duplicates) so a panel never sits in both halves.
* Splits groups into Test A / Test B (seeded, stratified by style series) and
  saves the split BEFORE any result is seen: data/test_holdout/split.csv
* Cuts ONE RANDOM tile per photo at the pipeline's working resolution. Tiles are
  random (not chosen by the model) and get NO suggested mask, so the answer key
  is not biased towards what the model predicts.

Writes into data/test_holdout/label/ (images/, masks/) -- never into
data/real_labeled, which training reads.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from restoration.utils import imread_rgb, imwrite, list_images, resize_max_side

# visually confirmed duplicates (same panel photographed twice)
SAME_PANEL = [("mandawa-19mar16", "mandawa-19mar20_21")]


def photo_group(stem: str) -> str:
    m = re.match(r"mandawa-19mar(\d+)", stem)
    return f"19mar{m.group(1)}" if m else stem


def series(stem: str) -> str:
    m = re.match(r"mandawa-19mar(\d+)", stem)
    if not m:
        return "mandawa0xx"
    n = int(m.group(1))
    return "ornament_16-23" if n <= 23 else "portraits_26-31" if n <= 31 else "blue_32-36" if n <= 36 else "friezes_37+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default=str(ROOT / "data/test_holdout"))
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--max-side", type=int, default=2048, help="must match restore/evaluate working size")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    td = Path(a.test_dir)
    paths = [p for p in list_images(td) if "label" not in p.parts and "masks" not in p.parts]
    if (td / "split.csv").exists():
        sys.exit("split.csv already exists - the split is fixed once made. Delete it only if NO result has been seen.")

    group = {p.stem: photo_group(p.stem) for p in paths}
    for x, y in SAME_PANEL:
        if x in group and y in group:
            g_old, g_new = group[y], group[x]
            group = {k: (g_new if v == g_old else v) for k, v in group.items()}

    rng = np.random.default_rng(a.seed)
    by_series, size = {}, {}
    for p in paths:
        by_series.setdefault(series(p.stem), set()).add(group[p.stem])
        size[group[p.stem]] = size.get(group[p.stem], 0) + 1
    # balanced by number of PHOTOS: within each series, biggest groups first (random
    # tie-break), each to the half with fewer photos overall-in-series so far
    half, count = {}, {"A": 0, "B": 0}
    for s in sorted(by_series):
        gs = sorted(by_series[s])
        rng.shuffle(gs)
        gs.sort(key=lambda g: -size[g])
        local = {"A": 0, "B": 0}
        for g in gs:
            pick = min("AB", key=lambda h: (local[h], count[h], rng.random()))
            half[g] = pick
            local[pick] += size[g]
            count[pick] += size[g]

    out = td / "label"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(paths):
        img, _ = resize_max_side(imread_rgb(p), a.max_side)
        h, w = img.shape[:2]
        t = min(a.tile, h, w)
        y, x = int(rng.integers(0, h - t + 1)), int(rng.integers(0, w - t + 1))
        name = f"{p.stem}_y{y}_x{x}"
        imwrite(out / "images" / f"{name}.png", img[y : y + t, x : x + t])
        rows.append({"image": p.relative_to(td).as_posix(), "tile": name, "split": half[group[p.stem]],
                     "group": group[p.stem], "series": series(p.stem), "y": y, "x": x, "t": t, "work_h": h, "work_w": w})
    with open(td / "split.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for s in "AB":
        r = [x for x in rows if x["split"] == s]
        print(f"Test {s}: {len(r)} photos, {len({x['group'] for x in r})} panels/groups, series "
              f"{sorted({x['series'] for x in r})}")
    print(f"Tiles in {out / 'images'}; split saved to {td / 'split.csv'}")


if __name__ == "__main__":
    main()
