"""Compare segmentation checkpoints on the development data (never on Test B).

    python scripts/compare_models.py checkpoints/seg_best.pt checkpoints/round3/seg_best.pt

Each checkpoint is used exactly as it would be at test time (its own frozen
threshold, TTA, no dilation):
  * Mandawa dev  - the model sees the WHOLE Test A dev photo; the labelled tiles
                   (data/mandawa_dev) are cropped out afterwards, as in evaluate_test.py
  * old real val - the 512-px validation tiles of data/real_labeled (split_paths, 0.3)
Writes outputs/model_comparison.csv.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from restoration.dataset import load_real_pairs, split_paths
from restoration.utils import imread_any, imread_rgb, resize_max_side


def scores(c):
    tp, fp, fn = c
    return dict(precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn),
                f1=2 * tp / max(1, 2 * tp + fp + fn), iou=tp / max(1, tp + fp + fn))


def counts(pr, gt):
    return np.array([(pr & gt).sum(), (pr & ~gt).sum(), (~pr & gt).sum()], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "outputs/model_comparison.csv"))
    a = ap.parse_args()
    from restoration.pipeline import RestorationPipeline

    roles = list(csv.DictReader(open(ROOT / "data/mandawa_A_label/roles.csv", encoding="utf-8")))
    dev_tiles = {p.stem for p in (ROOT / "data/mandawa_dev/masks").glob("*.png")}
    kind = {r["tile"]: r["kind"] for r in roles}  # answer_key = labelled from blank; others started from old-model suggestion
    photos = sorted({r["image"] for r in roles if r["role"] == "dev"})
    _, val = split_paths(load_real_pairs(ROOT / "data/real_labeled"), 0.3, seed=0)
    rows = []
    for ck in a.ckpts:
        pipe = RestorationPipeline(ck, None, str(ROOT / "weights/big-lama.pt"), str(ROOT / "weights/RealESRGAN_x4plus.pth"))
        thr = pipe.seg_ck["threshold"]
        c_dev, c_val, c_blank, c_sugg = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        for ph in photos:
            work, _ = resize_max_side(imread_rgb(ROOT / "data/test_holdout" / ph), 2048)
            mask = pipe.detect(work, dilate=0)[0] > 0
            stem = Path(ph).stem
            for t in sorted(dev_tiles):
                m = re.fullmatch(re.escape(stem) + r"_y(\d+)_x(\d+)", t)
                if not m:
                    continue
                y, x = int(m.group(1)), int(m.group(2))
                gt = imread_any(ROOT / "data/mandawa_dev/masks" / f"{t}.png", cv2.IMREAD_GRAYSCALE) > 0
                c = counts(mask[y : y + gt.shape[0], x : x + gt.shape[1]], gt)
                c_dev += c
                if kind.get(t) == "answer_key":
                    c_blank += c
                else:
                    c_sugg += c
        for ip, mp in val:
            pr = pipe.detect(imread_rgb(ip), dilate=0)[0] > 0
            c_val += counts(pr, imread_any(mp, cv2.IMREAD_GRAYSCALE) > 0)
        for name, c in (("mandawa_dev", c_dev), ("mandawa_dev_blank_start", c_blank),
                        ("mandawa_dev_from_suggestion", c_sugg), ("old_real_val", c_val)):
            rows.append({"checkpoint": ck, "threshold": thr, "set": name, **{k: round(v, 3) for k, v in scores(c).items()}})
            print(rows[-1])
        del pipe
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
