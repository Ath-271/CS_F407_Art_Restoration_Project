"""Merge returned label zips/folders into data/real_labeled/masks.

    python scripts/merge_labels.py returned/*.zip
    python scripts/merge_labels.py returned/          # folder of zips / unzipped kits

* Normal tiles: copied into data/real_labeled/masks (existing masks are kept
  unless --force).
* Calibration tiles (calib_*): pairwise IoU between annotators is reported
  (outputs/annotator_agreement.csv) and a CONSENSUS mask is written:
  damage where >= half of the annotators marked it; type = most common label.
"""
import argparse
import csv
import itertools
import re
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from restoration.utils import imread_any, imwrite


def collect(inputs, tmp: Path):
    """Yield (annotator, mask_path) for every returned mask."""
    srcs = []
    for s in inputs:
        p = Path(s)
        srcs += sorted(p.glob("*.zip")) + [d for d in p.iterdir() if d.is_dir()] if p.is_dir() else [p]
    for s in srcs:
        who = re.sub(r"_labels$", "", s.stem)
        if s.suffix == ".zip":
            d = tmp / who
            zipfile.ZipFile(s).extractall(d)
        else:
            d = s
        masks = list(d.rglob("*.png"))
        for m in masks:
            if "premasks" in m.parts or "images" in m.parts:
                continue
            yield who, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--root", default=str(ROOT / "data/real_labeled"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--agreement-out", default=str(ROOT / "outputs/annotator_agreement.csv"))
    a = ap.parse_args()
    root = Path(a.root)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    calib = defaultdict(dict)
    n_new = n_skip = n_bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for who, mp in collect(a.inputs, Path(tmp)):
            name = mp.stem
            is_cal = name.startswith("calib_")
            tile = name[len("calib_"):] if is_cal else name
            img_p = root / "images" / f"{tile}.png"
            if not img_p.exists():
                print(f"  ? {who}: {name} has no matching tile image - skipped"); n_bad += 1; continue
            m = imread_any(mp, cv2.IMREAD_GRAYSCALE)
            h, w = imread_any(img_p, cv2.IMREAD_GRAYSCALE).shape
            if m.shape != (h, w):
                print(f"  ! {who}: {name} wrong size {m.shape} vs {(h, w)} - skipped"); n_bad += 1; continue
            if is_cal:
                calib[tile][who] = m
                continue
            dst = root / "masks" / f"{tile}.png"
            if dst.exists() and not a.force:
                n_skip += 1; continue
            imwrite(dst, m, rgb=False); n_new += 1

    print(f"Merged {n_new} masks ({n_skip} already present, {n_bad} problems).")
    if not calib:
        return
    rows, ious = [], []
    for tile, by in sorted(calib.items()):
        names = sorted(by)
        for x, y in itertools.combinations(names, 2):
            A, B = by[x] > 0, by[y] > 0
            union = (A | B).sum()
            iou = float((A & B).sum() / union) if union else 1.0
            ious.append(iou)
            rows.append({"tile": tile, "a": x, "b": y, "iou_any_damage": round(iou, 3),
                         "a_damage_%": round(100 * A.mean(), 2), "b_damage_%": round(100 * B.mean(), 2)})
        st = np.stack([by[n] for n in names])
        votes = (st > 0).sum(0)
        cons = np.zeros(st.shape[1:], np.uint8)
        dmg = votes >= max(1, int(np.ceil(len(names) / 2)))
        typed = np.where(st == 255, 0, st)  # 'unknown' doesn't vote for a type
        counts = np.stack([(typed == c).sum(0) for c in (1, 2, 3)])
        best = counts.argmax(0) + 1
        cons[dmg] = np.where(counts.max(0)[dmg] > 0, best[dmg], 255)
        imwrite(root / "masks" / f"{tile}.png", cons, rgb=False)
    out = Path(a.agreement_out)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"Calibration: {len(calib)} tiles, {len(ious)} annotator pairs, mean IoU = {np.mean(ious):.3f} "
          f"(min {np.min(ious):.3f}). Details: {out}. Consensus masks written.")
    if np.mean(ious) < 0.4:
        print("  Low agreement - compare the calibration masks together and align on the guide before continuing.")


if __name__ == "__main__":
    main()
