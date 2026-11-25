"""Figure 7: held-out Mandawa test results (numbers only - no Mandawa image is shown).

    python scripts/make_test_figure.py

Reads outputs/test_results_{A,B}/metrics.csv + summary.json, outputs/model_comparison.csv
and data/test_holdout/label/masks (to pool A+B). Writes
docs/figures/fig7_test_results.png and adds the test numbers to docs/figures/numbers.json.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# reference palette (validated: blue / orange pass CVD and contrast checks)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    out = ROOT / "docs/figures"
    summ = {s: json.loads((ROOT / f"outputs/test_results_{s}/summary.json").read_text()) for s in "AB"}
    tiles = {s: [r for r in csv.DictReader(open(ROOT / f"outputs/test_results_{s}/metrics.csv")) if r.get("f1")]
             for s in "AB"}
    comp = {(r["checkpoint"], r["set"]): r for r in csv.DictReader(open(ROOT / "outputs/model_comparison.csv"))}
    old, new = "checkpoints/seg_best.pt", "checkpoints/round3/seg_best.pt"

    # exact A+B counts from each split's pooled precision/recall and answer-key size
    # (per-tile precision cannot be used: it loses the false positives of tiles with precision 0)
    import cv2

    tot = np.zeros(3)
    for s in "AB":
        g = sum((cv2.imread(str(ROOT / "data/test_holdout/label/masks" / f"{r['tile']}.png"), 0) > 0).sum() for r in tiles[s])
        tp = summ[s]["pooled_recall"] * g
        tot += [tp, tp / summ[s]["pooled_precision"] - tp, g - tp]
    f1_ab = 2 * tot[0] / (2 * tot[0] + tot[1] + tot[2])

    val = float(comp[(old, "old_real_val")]["f1"])
    bars = [("Validation\n(36 tiles, mostly fresco)", val),
            ("Test A\n(31 photos)", summ["A"]["pooled_f1"]),
            ("Test B\n(32 photos)", summ["B"]["pooled_f1"]),
            ("Test A + B\n(63 photos)", f1_ab)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1.25, 0.9, 1.25]})
    fig.patch.set_facecolor("white")

    # (a) pooled F1 per evaluation set
    ax = axes[0]
    style(ax)
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], width=0.55, color=BLUE, edgecolor="white", linewidth=2)
    for i, (_, v) in enumerate(bars):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")
    ax.set_xticks(x, [b[0] for b in bars], color=INK2, fontsize=9)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Damage-detection F1 (pooled pixels)", color=INK2)
    ax.set_title("(a) Same frozen model: validation vs Mandawa test", loc="left", color=INK, fontsize=11)

    # (b) per-tile F1 distribution
    ax = axes[1]
    style(ax)
    rng = np.random.default_rng(0)
    for i, s in enumerate("AB"):
        f = np.array([float(r["f1"]) for r in tiles[s]])
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(f)), f, s=34, color=BLUE, alpha=0.75,
                   edgecolor="white", linewidth=0.8, zorder=3)
        med = float(np.median(f))
        ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=2, zorder=4)
        ax.text(i + 0.31, med, f"median {med:.2f}", va="center", color=INK, fontsize=9)
        ax.text(i, -0.07, f"{(f == 0).sum()} tiles at 0", ha="center", color=INK2, fontsize=8.5)
    ax.set_xticks([0, 1], ["Test A", "Test B"], color=INK2)
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(-0.1, 0.8)
    ax.set_yticks(np.arange(0, 0.81, 0.1))
    ax.set_ylabel("F1 per labelled tile", color=INK2)
    ax.set_title("(b) Per-tile F1 (one random tile per photo)", loc="left", color=INK, fontsize=11)

    # (c) round-3 fine-tuning vs original model
    ax = axes[2]
    style(ax)
    sets = [("mandawa_dev_blank_start", "Mandawa dev,\nlabelled from blank\n(8 tiles, fair)"),
            ("mandawa_dev_from_suggestion", "Mandawa dev,\nlabelled from old\nmodel's suggestion (15)"),
            ("old_real_val", "Old validation\n(36 tiles)")]
    x = np.arange(len(sets))
    w = 0.36
    for j, (ck, lab, col) in enumerate([(old, "Original model (used for Test B)", BLUE),
                                        (new, "Fine-tuned on 64 Mandawa tiles", ORANGE)]):
        vals = [float(comp[(ck, k)]["f1"]) for k, _ in sets]
        ax.bar(x + (j - 0.5) * (w + 0.02), vals, width=w, color=col, edgecolor="white", linewidth=2, label=lab)
        for i, v in enumerate(vals):
            ax.text(x[i] + (j - 0.5) * (w + 0.02), v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    color=INK, fontsize=9.5)
    ax.set_xticks(x, [s[1] for s in sets], color=INK2, fontsize=8.5)
    ax.set_ylim(0, 0.9)
    ax.set_yticks(np.arange(0, 0.81, 0.1))
    ax.set_ylabel("F1 (whole photo, frozen threshold)", color=INK2)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK)
    ax.set_title("(c) Fine-tuning attempt after Test A", loc="left", color=INK, fontsize=11)

    fig.suptitle("Held-out Mandawa wall paintings: automatic damage detection", color=INK, fontsize=13,
                 x=0.01, ha="left")
    fig.savefig(out / "fig7_test_results.png", dpi=200)
    plt.close(fig)

    nums_p = out / "numbers.json"
    nums = json.loads(nums_p.read_text())
    nums.update({
        "test_A_f1": summ["A"]["pooled_f1"], "test_A_precision": summ["A"]["pooled_precision"],
        "test_A_recall": summ["A"]["pooled_recall"], "test_B_f1": summ["B"]["pooled_f1"],
        "test_B_precision": summ["B"]["pooled_precision"], "test_B_recall": summ["B"]["pooled_recall"],
        "test_AB_f1": f1_ab, "test_AB_precision": tot[0] / (tot[0] + tot[1]), "test_AB_recall": tot[0] / (tot[0] + tot[2]),
        "test_checkpoint_md5": summ["B"]["checkpoint_md5"],
        "test_median_tile_f1_A": float(np.median([float(r["f1"]) for r in tiles["A"]])),
        "test_median_tile_f1_B": float(np.median([float(r["f1"]) for r in tiles["B"]])),
        "val_f1_whole_tile_same_scoring": val,
        "round3_dev_blank_f1_old": float(comp[(old, "mandawa_dev_blank_start")]["f1"]),
        "round3_dev_blank_f1_new": float(comp[(new, "mandawa_dev_blank_start")]["f1"]),
    })
    nums_p.write_text(json.dumps(nums, indent=2))
    print(f"Saved {out / 'fig7_test_results.png'}; A+B F1 {f1_ab:.3f}")


if __name__ == "__main__":
    main()
