"""Figures for the report (no Mandawa data is used anywhere here).

    python scripts/make_report_figures.py

Writes outputs/report_figures/:
  fig1_seg_training_curves.png   real vs synthetic F1 per epoch (round 1 vs round 2)
  fig2_color_training_curve.png  de-fading PSNR per epoch vs the faded-input baseline
  fig3_synthetic_damage.png      one clean tile + each synthetic damage type
  fig4_round1_vs_round2.png      validation tiles: ground truth vs both models
  fig5_annotator_agreement.png   fresco calibration tiles labelled by all 4 annotators
  fig6_pipeline_stages.png       input -> mask -> inpainted -> colour -> final -> confidence
  numbers.json                   the headline numbers quoted in the captions
"""
import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from restoration.utils import imread_any, imread_rgb, list_images, resize_max_side

COL = {1: (220, 30, 30), 2: (30, 90, 255), 3: (255, 210, 0), 255: (230, 0, 230)}  # RGB
NAMES = {1: "crack", 2: "loss", 3: "stain", 255: "unknown type"}


def overlay(img, m, alpha=0.6):
    v = img.copy()
    for c, rgb in COL.items():
        sel = m == c
        v[sel] = (v[sel] * (1 - alpha) + np.array(rgb) * alpha).astype(np.uint8)
    return v


def overlay_binary(img, m, rgb=(220, 30, 30), alpha=0.6):
    v = img.copy()
    v[m] = (v[m] * (1 - alpha) + np.array(rgb) * alpha).astype(np.uint8)
    return v


def read_log(p):
    rows = list(csv.DictReader(open(p)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def fig_seg_curves(out, nums):
    r2 = read_log(ROOT / "checkpoints/seg_log.csv")
    r1p = ROOT / "checkpoints/round1_synthetic_only/seg_log.csv"
    fig, ax = plt.subplots(layout="constrained", figsize=(7.5, 4.2))
    ax.plot(r2["epoch"], r2["synth_f1"], color="#7a7a7a", lw=2, label="Round 2 - synthetic validation")
    ax.plot(r2["epoch"], r2["real_f1"], color="#c0392b", lw=2.4, label="Round 2 - REAL validation tiles")
    best = int(np.nanargmax(r2["real_f1"]))
    ax.scatter([r2["epoch"][best]], [r2["real_f1"][best]], color="#c0392b", zorder=5, s=45)
    ax.annotate(f"selected: epoch {int(r2['epoch'][best])}, F1 {r2['real_f1'][best]:.2f}",
                (r2["epoch"][best], r2["real_f1"][best]), xytext=(8, -18), textcoords="offset points", fontsize=9)
    if r1p.exists():
        r1 = read_log(r1p)
        ax.plot(r1["epoch"], r1["synth_f1"], color="#7a7a7a", lw=1.2, ls="--", label="Round 1 (synthetic only) - synthetic val")
    if "round1_real_f1" in nums:
        ax.axhline(nums["round1_real_f1"], color="#c0392b", ls=":", lw=1.4, label=f"Round 1 on real tiles: F1 {nums['round1_real_f1']:.2f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("F1 (any damage)"); ax.set_ylim(0, 1)
    ax.set_title("Damage detection: synthetic vs real validation")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.savefig(out / "fig1_seg_training_curves.png", dpi=200); plt.close(fig)
    nums.update(round2_best_epoch=int(r2["epoch"][best]), round2_real_f1=float(r2["real_f1"][best]),
                round2_real_iou=float(r2["real_iou"][best]), round2_threshold=float(r2["real_thr"][best]),
                round2_epochs_run=int(len(r2["epoch"])))


def fig_color_curve(out, nums):
    c = read_log(ROOT / "checkpoints/color_log.csv")
    fig, ax = plt.subplots(layout="constrained", figsize=(7.5, 4))
    ax.plot(c["epoch"], c["val_psnr"], color="#2471a3", lw=2.2, label="de-faded output (Pix2Pix)")
    ax.plot(c["epoch"], c["baseline_psnr"], color="#7a7a7a", ls="--", lw=1.6, label="faded input (no model)")
    b = int(np.argmin(c["val_l1"]))  # the checkpoint saved as color_best.pt (selected by val L1)
    ax.scatter([c["epoch"][b]], [c["val_psnr"][b]], color="#2471a3", zorder=5)
    ax.annotate(f"saved: epoch {int(c['epoch'][b])}, {c['val_psnr'][b]:.2f} dB (+{c['val_psnr'][b] - c['baseline_psnr'][b]:.1f})",
                (c["epoch"][b], c["val_psnr"][b]), xytext=(-110, 10), textcoords="offset points", fontsize=9)
    ax.set_xlabel("epoch"); ax.set_ylabel("PSNR on validation (dB)")
    ax.set_title("Colour de-fading: synthetic-fade validation")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.savefig(out / "fig2_color_training_curve.png", dpi=200); plt.close(fig)
    nums.update(color_best_psnr=float(c["val_psnr"][b]), color_baseline_psnr=float(c["baseline_psnr"][b]),
                color_best_epoch=int(c["epoch"][b]))


def fig_synthetic(out, seed=3):
    from restoration.dataset import random_crop
    from restoration.synth_damage import SynthConfig, synthesize

    rng = np.random.default_rng(seed)
    clean_paths = list_images(ROOT / "data/clean")
    # a colourful, detailed clean tile
    best, best_score = None, -1
    for p in rng.choice(clean_paths, 25, replace=False):
        t = random_crop(resize_max_side(imread_rgb(p), 1500)[0], 384, rng, (0.9, 1.1))
        score = cv2.Laplacian(cv2.cvtColor(t, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var() * (t.std() > 35)
        if score > best_score:
            best, best_score = t, score
    base = dict(p_clean=0.0, p_crack=0.0, p_loss=0.0, p_stain=0.0, p_bank=0.0, p_plaster=0.0, p_scan_noise=0.0)
    variants = [("cracks / craquelure", dict(p_crack=1.0)), ("paint loss / flaking", dict(p_loss=1.0)),
                ("stains", dict(p_stain=1.0)), ("plaster loss (fresco)", dict(p_plaster=1.0)),
                ("combined + scan noise", dict(p_crack=1.0, p_loss=1.0, p_stain=1.0, p_scan_noise=1.0))]
    fig, axes = plt.subplots(2, 6, layout="constrained", figsize=(16, 5.8))
    axes[0, 0].imshow(best); axes[0, 0].set_title("clean tile", fontsize=10)
    axes[1, 0].axis("off")
    for i, (name, kw) in enumerate(variants, start=1):
        cfg = SynthConfig(**{**base, **kw})
        img, lab = synthesize(best.copy(), np.random.default_rng(seed + i), cfg)
        m = np.zeros(lab.shape[:2], np.uint8)
        for ci in range(3):
            m[lab[..., ci] > 0.5] = ci + 1
        axes[0, i].imshow(img); axes[0, i].set_title(name, fontsize=10)
        axes[1, i].imshow(overlay(img, m)); axes[1, i].set_title("label", fontsize=9)
    for a in axes.flat:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Synthetic damage generator (labels: red crack, blue loss, yellow stain)", fontsize=12)
    fig.savefig(out / "fig3_synthetic_damage.png", dpi=160); plt.close(fig)


def fig_round_compare(out, nums, device):
    import torch
    from torch.utils.data import DataLoader

    from restoration.dataset import RealValDataset, load_real_pairs, painting_key, split_paths
    from restoration.segmentation import load_seg_model, predict_probs

    sys.path.insert(0, str(ROOT / "scripts"))
    from train_seg import evaluate

    _, val = split_paths(load_real_pairs(ROOT / "data/real_labeled"), 0.3, seed=0)
    m1, ck1 = load_seg_model(ROOT / "checkpoints/round1_synthetic_only/seg_best.pt", device)
    m2, ck2 = load_seg_model(ROOT / "checkpoints/seg_best.pt", device)
    dl = DataLoader(RealValDataset(val, 384, ck1.get("use_clahe", True)), batch_size=8)
    f1_1, iou_1, thr_1 = evaluate(m1, dl, device)
    nums.update(round1_real_f1=float(f1_1), round1_real_iou=float(iou_1), round1_best_thr_on_val=float(thr_1),
                n_val_tiles=len(val), n_val_paintings=len({painting_key(p) for p in val}))
    # pick 5 tiles with clear damage, mixing paper and fresco
    scored = []
    for ip, mp in val:
        gm = imread_any(mp, cv2.IMREAD_GRAYSCALE)
        scored.append(((gm > 0).mean(), ip, mp))
    paper = sorted([s for s in scored if "fresco" not in s[1].stem and 0.01 < s[0] < 0.4], key=lambda s: -s[0])[:2]
    fresco = sorted([s for s in scored if "fresco" in s[1].stem and 0.03 < s[0] < 0.5], key=lambda s: -s[0])[:3]
    pick = paper + fresco
    fig, axes = plt.subplots(len(pick), 4, layout="constrained", figsize=(12, 3 * len(pick)))
    for r, (_, ip, mp) in enumerate(pick):
        img, gm = imread_rgb(ip), imread_any(mp, cv2.IMREAD_GRAYSCALE)
        preds = []
        for m, ck in ((m1, ck1), (m2, ck2)):
            pr, _ = predict_probs(m, img, device, ck.get("use_clahe", True), tta=False)
            preds.append(pr[..., 0] > ck.get("threshold", 0.5))
        panels = [(img, "input (unseen painting)"), (overlay(img, gm), "human label"),
                  (overlay_binary(img, preds[0]), "round 1: synthetic only"),
                  (overlay_binary(img, preds[1]), "round 2: + real labels")]
        for c, (v, t) in enumerate(panels):
            axes[r, c].imshow(v); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(t, fontsize=11)
        axes[r, 0].set_ylabel(painting_key(ip), fontsize=8)
    fig.suptitle(f"Validation paintings never used for training  |  real-tile F1: round 1 {f1_1:.2f} -> round 2 "
                 f"{nums.get('round2_real_f1', float('nan')):.2f}", fontsize=12)
    fig.savefig(out / "fig4_round1_vs_round2.png", dpi=150); plt.close(fig)


def fig_agreement(out, nums):
    rets = ROOT / "outputs/label_returns"
    rows = list(csv.DictReader(open(ROOT / "outputs/annotator_agreement.csv")))
    ious = np.array([float(r["iou_any_damage"]) for r in rows])
    nums.update(agreement_mean_iou=float(ious.mean()), agreement_median_iou=float(np.median(ious)),
                agreement_pairs=len(ious), agreement_tiles=len({r["tile"] for r in rows}))
    people = ["Atharva", "Rudranil", "Soumya", "Vidwath"]
    tiles = sorted({r["tile"] for r in rows if r["tile"].startswith("1")})  # batch-2 fresco calibration tiles
    fig, axes = plt.subplots(len(tiles), 6, layout="constrained", figsize=(17, 3 * len(tiles)))
    for i, t in enumerate(tiles):
        img = imread_rgb(ROOT / "data/real_labeled/images" / f"{t}.png")
        cons = imread_any(ROOT / "data/real_labeled/masks" / f"{t}.png", cv2.IMREAD_GRAYSCALE)
        axes[i, 0].imshow(img)
        for j, p in enumerate(people):
            m = cv2.imdecode(np.frombuffer(zipfile.ZipFile(rets / f"{p}_batch2_labels.zip").read(f"calib_{t}.png"), np.uint8), 0)
            axes[i, j + 1].imshow(overlay(img, m))
            if i == 0:
                axes[i, j + 1].set_title(f"annotator {j + 1}", fontsize=11)
        axes[i, 5].imshow(overlay(img, cons))
        tile_iou = np.mean([float(r["iou_any_damage"]) for r in rows if r["tile"] == t])
        axes[i, 0].set_ylabel(f"{t}\nmean pairwise IoU {tile_iou:.2f}", fontsize=8)
        if i == 0:
            axes[i, 0].set_title("tile", fontsize=11); axes[i, 5].set_title("majority-vote consensus", fontsize=11)
    for a in axes.flat:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Inter-annotator agreement on calibration tiles (all {len(ious)} pairs over "
                 f"{nums['agreement_tiles']} tiles: mean IoU {ious.mean():.2f})", fontsize=12)
    fig.savefig(out / "fig5_annotator_agreement.png", dpi=140); plt.close(fig)


def fig_pipeline(out, name):
    d = ROOT / "outputs/pipeline_check_v2" / name
    rep = json.load(open(d / "report.json"))
    files = [("00_input.png", "1. input"), ("01_mask_overlay.png", f"2. detected damage ({100 * rep['damaged_fraction']:.1f}%)"),
             ("02_inpainted.png", "3. inpainted (LaMa)"), ("03_color.png", "4. de-faded (adaptive)"),
             ("04_final.png", "5. final, 2x super-resolved"), ("confidence_map.png", "6. confidence (red = guess)")]
    fig, axes = plt.subplots(2, 3, layout="constrained", figsize=(16, 7.5))
    for a, (f, t) in zip(axes.flat, files):
        a.imshow(imread_rgb(d / f)); a.set_title(t, fontsize=11); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Full pipeline on a validation fresco ({name}); total {rep['total_s']:.0f} s on an RTX 4060 laptop",
                 fontsize=12)
    fig.savefig(out / "fig6_pipeline_stages.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/report_figures"))
    ap.add_argument("--pipeline-example", default="1155_fresco_bundi")
    a = ap.parse_args()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nums = {}
    fig_color_curve(out, nums)
    fig_round_compare(out, nums, device)  # also measures round-1 real F1 used in fig 1
    fig_seg_curves(out, nums)
    fig_round_compare(out, nums, device)  # redraw title with round-2 number
    fig_synthetic(out)
    fig_agreement(out, nums)
    fig_pipeline(out, a.pipeline_example)
    (out / "numbers.json").write_text(json.dumps(nums, indent=2))
    print(json.dumps(nums, indent=2))
    print(f"Figures in {out}")


if __name__ == "__main__":
    main()
