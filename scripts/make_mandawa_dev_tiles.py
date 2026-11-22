"""Extra labelling tiles from the Test A photos (after Test A was scored once).

    python scripts/make_mandawa_dev_tiles.py

Protocol: Test A has been scored with the frozen model, so A is now a DEVELOPMENT
set. Test B stays sealed: this script reads only split == "A" photos and refuses
to touch anything else (Test B and data/personal_holdout are never opened).

1. Splits the Test A panels (photo groups) into
     train  (~2/3) - new tiles AND the existing answer-key tile go to training
     dev    (~1/3) - used only to compare the old and new model
   balanced by style series; saved once to data/mandawa_A_label/roles.csv.
2. Cuts 2 new 512-px tiles per A photo at the working size (2048), each overlapping
   any earlier tile of that photo by at most 25 %:
     uncertain - where the model's damage probability is most often 0.2-0.6
     random    - anywhere else
3. Writes the model's prediction as a suggested mask (premasks/), so annotators
   correct it instead of drawing from scratch.

Output: data/mandawa_A_label/{images,premasks,masks}, roles.csv
Then:   python scripts/make_label_kits.py --root data/mandawa_A_label --kit-name mandawaA ...
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from restoration.utils import imread_rgb, imwrite, resize_max_side


def overlap(a, b, t):
    dy, dx = max(0, t - abs(a[0] - b[0])), max(0, t - abs(a[1] - b[1]))
    return dy * dx / (t * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default=str(ROOT / "data/test_holdout"))
    ap.add_argument("--out", default=str(ROOT / "data/mandawa_A_label"))
    ap.add_argument("--seg-ckpt", default=str(ROOT / "checkpoints/seg_best.pt"))
    ap.add_argument("--dev-frac", type=float, default=0.35)
    ap.add_argument("--max-overlap", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    td, out = Path(a.test_dir), Path(a.out)
    if (out / "roles.csv").exists():
        sys.exit(f"{out / 'roles.csv'} exists - tiles and the train/dev split are made once.")
    rows = [r for r in csv.DictReader(open(td / "split.csv", encoding="utf-8")) if r["split"] == "A"]
    assert rows and all(r["split"] == "A" for r in rows)
    rng = np.random.default_rng(a.seed)

    # ---- train / dev split of A by panel group, balanced within each series
    size, by_series = {}, {}
    for r in rows:
        size[r["group"]] = size.get(r["group"], 0) + 1
        by_series.setdefault(r["series"], set()).add(r["group"])
    role = {}
    n_dev_total = 0
    for s in sorted(by_series):
        gs = sorted(by_series[s])
        rng.shuffle(gs)
        photos = sum(size[g] for g in gs)
        target, got = max(1, round(photos * a.dev_frac)) if len(gs) > 1 else 0, 0
        for g in sorted(gs, key=lambda g: size[g]):
            if got + size[g] <= target or (got == 0 and target > 0):
                role[g], got = "dev", got + size[g]
            else:
                role[g] = "train"
        n_dev_total += got

    from restoration.pipeline import RestorationPipeline

    pipe = RestorationPipeline(a.seg_ckpt, None, str(ROOT / "weights/big-lama.pt"), str(ROOT / "weights/RealESRGAN_x4plus.pth"))
    for sub in ("images", "premasks", "masks"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    roles, n = [], 0
    for r in rows:
        img, _ = resize_max_side(imread_rgb(td / r["image"]), 2048)
        h, w = img.shape[:2]
        assert (h, w) == (int(r["work_h"]), int(r["work_w"])), r["image"]
        t = int(r["t"])
        mask, prob, _, thr = pipe.detect(img, dilate=0)
        from restoration.segmentation import predict_probs

        probs, _ = predict_probs(pipe.seg, img, pipe.device, use_clahe=pipe.seg_ck.get("use_clahe", True), tta=False)
        pre = np.zeros((h, w), np.uint8)
        cls = probs[..., 1:].argmax(-1) + 1
        pre[mask > 0] = cls[mask > 0]
        unsure = ((prob >= 0.2) & (prob <= 0.6)).astype(np.float32)

        taken = [(int(r["y"]), int(r["x"]))]
        cands = [(y, x) for y in range(0, h - t + 1, 32) for x in range(0, w - t + 1, 32)]
        if (h - t, w - t) not in cands:
            cands.append((h - t, w - t))
        ok = lambda c: all(overlap(c, q, t) <= a.max_overlap for q in taken)
        picks = []
        free = [c for c in cands if ok(c)]
        if free:
            sc = np.array([unsure[y : y + t, x : x + t].mean() for y, x in free])
            c = free[int(sc.argmax())]
            picks.append((c, "uncertain"))
            taken.append(c)
        free = [c for c in cands if ok(c)]
        if free:
            c = free[int(rng.integers(len(free)))]
            picks.append((c, "random"))
            taken.append(c)
        stem = Path(r["image"]).stem
        g_role = role[r["group"]]
        roles.append({"tile": r["tile"], "image": r["image"], "group": r["group"], "series": r["series"],
                      "role": g_role, "kind": "answer_key"})
        for (y, x), kind in picks:
            name = f"{stem}_y{y}_x{x}"
            imwrite(out / "images" / f"{name}.png", img[y : y + t, x : x + t])
            imwrite(out / "premasks" / f"{name}.png", pre[y : y + t, x : x + t], rgb=False)
            roles.append({"tile": name, "image": r["image"], "group": r["group"], "series": r["series"],
                          "role": g_role, "kind": kind})
            n += 1
        print(f"{stem:28s} {g_role:5s} {len(picks)} new tiles")
    with open(out / "roles.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, list(roles[0]))
        wr.writeheader()
        wr.writerows(roles)
    for k in ("train", "dev"):
        ph = {x["image"] for x in roles if x["role"] == k}
        print(f"{k}: {len(ph)} photos, {sum(x['role'] == k and x['kind'] != 'answer_key' for x in roles)} new tiles")
    print(f"{n} new tiles in {out / 'images'}; roles saved to {out / 'roles.csv'}")


if __name__ == "__main__":
    main()
