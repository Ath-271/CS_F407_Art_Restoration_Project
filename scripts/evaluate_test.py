"""FINAL evaluation on the held-out test set (e.g. the professor's Mandawa paintings).

    python scripts/evaluate_test.py                      # data/test_holdout
    python scripts/evaluate_test.py --no-restore         # metrics only

Rules (to keep the test honest):
  * data/test_holdout/ is never read by any training / labelling script.
  * The checkpoint and its threshold are FROZEN (chosen on the validation set);
    this script never tunes anything.
  * Run it once you're done developing. If you change the model after looking
    at these results, say so in your report.

Ground truth for segmentation metrics (either):
  * labelled TILES: data/test_holdout/split.csv + label/masks/<tile>.png
    (made by scripts/make_test_tiles.py). The model sees the WHOLE photo; only the
    labelled tile region is scored, after prediction.
  * full masks: data/test_holdout/masks/<same name>.png  (0 = background, >0 = damage)

Two-stage protocol (split.csv): --split A for the current model; if the model is
changed afterwards, the changed model is scored ONCE on --split B. Every run is
appended to outputs/test_runs_log.csv (time, split, checkpoint hash) so the
number of looks at each half is documented.

Outputs: outputs/test_results_<split>/<name>/ (restorations) + metrics.csv + summary.json
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from restoration.utils import imread_any, imread_rgb, list_images, resize_max_side

TRAIN_DIRS = ["data/clean", "data/damaged", "data/real_labeled/images", "data/damage_bank", "data/pool"]


def file_hashes(paths):
    return {hashlib.md5(Path(p).read_bytes()).hexdigest(): p for p in paths}


def pixel_hash(img):
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (16, 16), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).tobytes()


def leak_check(test_paths):
    """Fail loudly if any test image (exact file or same picture re-encoded) is in training data."""
    train = [p for d in TRAIN_DIRS for p in list_images(ROOT / d) if "masks" not in p.parts]
    th = file_hashes(train)
    leaks = [(t, th[h]) for t, h in ((t, hashlib.md5(t.read_bytes()).hexdigest()) for t in test_paths) if h in th]
    tp = {pixel_hash(resize_max_side(imread_rgb(p), 256)[0]): p for p in test_paths}
    for p in train:
        try:
            h = pixel_hash(resize_max_side(imread_rgb(p), 256)[0])
        except Exception:
            continue
        if h in tp:
            leaks.append((tp[h], p))
    return leaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(ROOT / "data/test_holdout"))
    ap.add_argument("--out", default=None, help="default outputs/test_results_<split>")
    ap.add_argument("--split", default=None, choices=["A", "B", "all"], help="required when split.csv exists")
    ap.add_argument("--force", action="store_true", help="allow re-running a split that already has results")
    ap.add_argument("--seg-ckpt", default=str(ROOT / "checkpoints/seg_best.pt"))
    ap.add_argument("--color-ckpt", default=str(ROOT / "checkpoints/color_best.pt"))
    ap.add_argument("--max-side", type=int, default=2048)
    ap.add_argument("--sr-scale", type=float, default=2.0)
    ap.add_argument("--no-restore", action="store_true")
    ap.add_argument("--skip-leak-check", action="store_true")
    a = ap.parse_args()

    test_dir = Path(a.test)
    paths = [p for p in list_images(test_dir) if "masks" not in p.parts and "label" not in p.parts]
    if not paths:
        sys.exit(f"No test images in {test_dir}")
    tiles = {}
    split_csv = test_dir / "split.csv"
    if split_csv.exists():
        if a.split is None:
            sys.exit("split.csv exists: choose --split A (current model), B (changed model, once) or all.")
        rows_split = list(csv.DictReader(open(split_csv, encoding="utf-8")))
        keep = {r["image"] for r in rows_split if a.split == "all" or r["split"] == a.split}
        tiles = {r["image"]: r for r in rows_split}
        paths = [p for p in paths if p.relative_to(test_dir).as_posix() in keep]
    tag = a.split or "all"
    out = Path(a.out) if a.out else ROOT / f"outputs/test_results_{tag}"
    if (out / "summary.json").exists() and not a.force:
        sys.exit(f"{out} already has results. Each test half should be scored once; use --force only if you "
                 "know why (and mention it in the report).")
    if not a.skip_leak_check:
        leaks = leak_check(paths)
        if leaks:
            for t, tr in leaks:
                print(f"LEAK: test image {t.name} also in training data: {tr}")
            sys.exit("Test images found in training data - remove them and retrain before evaluating.")
        print(f"Leak check passed: none of the {len(paths)} test images appear in training data.")

    from restoration.pipeline import RestorationPipeline

    pipe = RestorationPipeline(a.seg_ckpt, a.color_ckpt, ROOT / "weights/big-lama.pt", ROOT / "weights/RealESRGAN_x4plus.pth")
    thr = pipe.seg_ck["threshold"] if pipe.seg_ck else None
    print(f"Frozen checkpoint: {a.seg_ckpt} (threshold {thr})")
    out.mkdir(parents=True, exist_ok=True)
    rows, tot = [], np.zeros(3)
    for p in paths:
        img = imread_rgb(p)
        work, _ = resize_max_side(img, a.max_side)
        row = {"image": p.name}
        rel = p.relative_to(test_dir).as_posix()
        gt, pr = None, None
        tr = tiles.get(rel)
        tile_mask = test_dir / "label" / "masks" / f"{tr['tile']}.png" if tr else None
        if tile_mask is not None and tile_mask.exists():
            if work.shape[:2] != (int(tr["work_h"]), int(tr["work_w"])):
                sys.exit(f"{p.name}: working size {work.shape[:2]} != size used for tiles - check --max-side")
            y, x, t = int(tr["y"]), int(tr["x"]), int(tr["t"])
            mask, *_ = pipe.detect(work, dilate=0)  # model sees the whole photo, no inpainting margin
            pr = mask[y : y + t, x : x + t] > 0
            gt = imread_any(tile_mask, cv2.IMREAD_GRAYSCALE) > 0
            row.update(split=tr["split"], tile=tr["tile"])
        elif (test_dir / "masks" / f"{p.stem}.png").exists():
            gt = imread_any(test_dir / "masks" / f"{p.stem}.png", cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            mask, *_ = pipe.detect(work, dilate=0)  # raw detection, no inpainting margin
            pr = mask > 0
        if gt is not None:
            c = np.array([(pr & gt).sum(), (pr & ~gt).sum(), (~pr & gt).sum()], float)
            tot += c
            row.update(precision=c[0] / max(1, c[0] + c[1]), recall=c[0] / max(1, c[0] + c[2]),
                       f1=2 * c[0] / max(1, 2 * c[0] + c[1] + c[2]), iou=c[0] / max(1, c.sum()))
        if not a.no_restore:
            rep = pipe.run(img, out / p.stem, work_max_side=a.max_side, sr_scale=a.sr_scale)
            row.update(damaged_fraction=rep["damaged_fraction"], mean_conf_infill=rep["mean_confidence_in_infill"])
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "image", k))
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    ck_md5 = hashlib.md5(Path(a.seg_ckpt).read_bytes()).hexdigest()[:12]
    summary = {"split": tag, "n_images": len(rows), "n_scored": sum("f1" in r for r in rows),
               "checkpoint": a.seg_ckpt, "checkpoint_md5": ck_md5, "threshold": thr}
    if tot.sum():
        summary.update(pooled_precision=tot[0] / max(1, tot[0] + tot[1]), pooled_recall=tot[0] / max(1, tot[0] + tot[2]),
                       pooled_f1=2 * tot[0] / max(1, 2 * tot[0] + tot[1] + tot[2]), pooled_iou=tot[0] / max(1, tot.sum()))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    import datetime
    log = ROOT / "outputs/test_runs_log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "split", "checkpoint", "checkpoint_md5", "n_scored", "pooled_f1", "pooled_iou"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), tag, a.seg_ckpt, ck_md5, summary["n_scored"],
                    round(summary.get("pooled_f1", float("nan")), 4), round(summary.get("pooled_iou", float("nan")), 4)])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
