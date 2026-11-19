"""Cut tiles from REAL damaged paintings for hand-labelling.

    python scripts/make_label_tiles.py --per-image 6

Tiles are taken at the SAME working resolution the pipeline uses (--max-side),
so labels match the scale the model sees at inference time.
Tiles with the most suspected damage are chosen (from the current model if a
checkpoint exists, else the classical detector), plus random ones so you also
label negatives / damage the model currently misses.

Writes:
    data/real_labeled/images/<name>.png     tile
    data/real_labeled/premasks/<name>.png   suggested mask (NOT used for training
                                            until you check & save it in label_tool)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from restoration.preprocess import classical_crack_mask
from restoration.utils import imread_rgb, imwrite, list_images, resize_max_side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/damaged"))
    ap.add_argument("--out", default=str(ROOT / "data/real_labeled"))
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/seg_best.pt"))
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--max-side", type=int, default=2048)
    ap.add_argument("--per-image", type=int, default=6)
    ap.add_argument("--random-frac", type=float, default=0.34)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    model = ck = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Path(a.ckpt).exists():
        from restoration.segmentation import load_seg_model, predict_probs

        model, ck = load_seg_model(a.ckpt, device)
        print("Using model predictions for pre-masks.")
    else:
        print("No checkpoint -> using classical crack detector for pre-masks (expect false positives on linework).")

    paths = list_images(a.src)
    if not paths:
        sys.exit(f"No images in {a.src}. Put real damaged paintings there first.")
    total = 0
    for p in paths:
        img, _ = resize_max_side(imread_rgb(p), a.max_side)
        if model is not None:
            probs, _ = predict_probs(model, img, device, ck.get("use_clahe", True), tta=False)
            pre = np.zeros(img.shape[:2], np.uint8)
            cls = probs[..., 1:].argmax(-1) + 1
            pre[probs[..., 0] > ck.get("threshold", 0.5)] = cls[probs[..., 0] > ck.get("threshold", 0.5)]
        else:
            pre = np.where(classical_crack_mask(img) > 0, 255, 0).astype(np.uint8)
        h, w = img.shape[:2]
        t = min(a.tile, h, w)
        cands = [(y, x) for y in range(0, h - t + 1, t // 2) for x in range(0, w - t + 1, t // 2)]
        scores = np.array([(pre[y : y + t, x : x + t] > 0).mean() for y, x in cands])
        n_rand = int(round(a.per_image * a.random_frac))
        chosen = list(np.argsort(-scores)[: a.per_image - n_rand])
        rest = [i for i in range(len(cands)) if i not in chosen]
        if rest and n_rand:
            chosen += list(rng.choice(rest, size=min(n_rand, len(rest)), replace=False))
        for i in chosen:
            y, x = cands[i]
            name = f"{p.stem}_y{y}_x{x}"
            if (out / "masks" / f"{name}.png").exists():
                continue  # never overwrite finished labels
            imwrite(out / "images" / f"{name}.png", img[y : y + t, x : x + t])
            imwrite(out / "premasks" / f"{name}.png", pre[y : y + t, x : x + t], rgb=False)
            total += 1
    print(f"Wrote {total} tiles to {out / 'images'}. Now run: python scripts/label_tool.py")


if __name__ == "__main__":
    main()
