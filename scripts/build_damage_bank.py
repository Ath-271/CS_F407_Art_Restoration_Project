"""Extract REAL damage cut-outs from labelled tiles -> data/damage_bank/*.png (RGBA).

These are pasted onto clean paintings during training ("copy-paste
augmentation"), giving thousands of realistic damage examples from a few labels.

    python scripts/build_damage_bank.py
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from restoration.dataset import load_real_pairs, split_paths
from restoration.synth_damage import CLASSES
from restoration.utils import imread_any, imread_rgb, imwrite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=str(ROOT / "data/real_labeled"))
    ap.add_argument("--out", default=str(ROOT / "data/damage_bank"))
    ap.add_argument("--min-area", type=int, default=25)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--real-val-frac", type=float, default=0.3,
                    help="must match train_seg.py: validation paintings are left OUT of the bank (no leakage)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():  # rebuild from scratch so stale cut-outs (e.g. from val paintings) never linger
        for f in out.rglob("*.png"):
            f.unlink()
    pairs = load_real_pairs(a.real)
    pairs, val = split_paths(pairs, a.real_val_frac, seed=0)
    print(f"bank from {len(pairs)} training tiles ({len(val)} validation tiles excluded)")
    n = 0
    for ip, mp in pairs:
        img, m = imread_rgb(ip), imread_any(mp, cv2.IMREAD_GRAYSCALE)
        h, w = m.shape
        # "unknown type" (255) components: thin/elongated -> crack, compact -> loss
        if (m == 255).any():
            k, lab, stats, _ = cv2.connectedComponentsWithStats((m == 255).astype(np.uint8), connectivity=8)
            m = m.copy()
            for j in range(1, k):
                bw, bh, area = stats[j, 2], stats[j, 3], stats[j, 4]
                thin = area / max(1, bw * bh) < 0.35 or max(bw, bh) / max(1, min(bw, bh)) > 4
                m[lab == j] = 1 if thin else 2
        for ci, cname in enumerate(CLASSES, start=1):
            cm = (m == ci).astype(np.uint8)
            k, lab, stats, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)
            for j in range(1, k):
                x, y, bw, bh, area = stats[j]
                if area < a.min_area or bw * bh > 0.6 * h * w:
                    continue
                x0, y0 = max(0, x - a.pad), max(0, y - a.pad)
                x1, y1 = min(w, x + bw + a.pad), min(h, y + bh + a.pad)
                alpha = (lab[y0:y1, x0:x1] == j).astype(np.float32)
                alpha = np.maximum(alpha, cv2.GaussianBlur(alpha, (0, 0), 1.0) * 0.8)
                rgba = np.dstack([img[y0:y1, x0:x1], (alpha * 255).astype(np.uint8)])
                imwrite(Path(a.out) / f"{cname}_{ip.stem}_{j}.png", rgba)
                n += 1
    print(f"Wrote {n} damage cut-outs to {a.out}")


if __name__ == "__main__":
    main()
