"""Generates notebooks/validation.ipynb (system validation notebook deliverable)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# System Validation Notebook\n"
       "Digital Restoration of Indian Folk Paintings. Covers data augmentation and patch-conversion statistics, "
       "synthetic damage samples, segmentation and colour loss curves, and a comparison on **real** labelled tiles "
       "(classical detector vs U-Net, with per-class metrics)."),
    code("import sys, json\nfrom pathlib import Path\n"
         "ROOT = next(d for d in [Path.cwd(), *Path.cwd().parents] if (d / 'restoration').is_dir())  # project root\nsys.path.insert(0, str(ROOT))\n"
         "import numpy as np, pandas as pd, matplotlib.pyplot as plt, cv2, torch\n"
         "from restoration.utils import list_images, imread_rgb, imread_any, resize_max_side\n"
         "from restoration.dataset import load_real_pairs, random_crop\n"
         "from restoration.synth_damage import SynthConfig, synthesize, synthetic_fade, load_damage_bank\n"
         "from restoration.preprocess import classical_crack_mask, clahe_lab\n\n"
         "# ---- configuration ----\n"
         "CKPT_DIR  = ROOT / 'checkpoints'\n"
         "CLEAN_DIR = ROOT / 'data/clean'\n"
         "REAL_DIR  = ROOT / 'data/real_labeled'\n"
         "BANK_DIR  = ROOT / 'data/damage_bank'\n"
         "TILE, OVERLAP, MAX_SIDE = 384, 96, 3000"),
    md("## 1. Dataset & patch-conversion rates\n"
       "How many overlapping training patches each painting yields at the chosen tile size "
       "(the proposal's *sub-tile extraction* step), plus random-scale cropping which makes the effective set unlimited."),
    code("def n_patches(h, w, tile=TILE, overlap=OVERLAP):\n"
         "    s = tile - overlap\n"
         "    return max(1, (h - overlap) // s) * max(1, (w - overlap) // s)\n"
         "rows = []\n"
         "for p in list_images(CLEAN_DIR):\n"
         "    im = imread_rgb(p); h, w = resize_max_side(im, MAX_SIDE)[0].shape[:2]\n"
         "    rows.append({'image': p.name, 'size': f'{w}x{h}', 'patches_256': n_patches(h, w, 256, 64), f'patches_{TILE}': n_patches(h, w)})\n"
         "stats = pd.DataFrame(rows)\n"
         "real = load_real_pairs(REAL_DIR); bank = load_damage_bank(BANK_DIR)\n"
         "print(f'clean paintings: {len(stats)} -> {stats.get(f\"patches_{TILE}\", pd.Series(dtype=int)).sum()} patches @{TILE}px '\n"
         "      f'(x{stats.get(f\"patches_{TILE}\", pd.Series([0])).mean():.0f} per painting); real labelled tiles: {len(real)}; damage-bank cut-outs: {len(bank)}')\n"
         "stats"),
    code("if real:\n"
         "    fr = []\n"
         "    for ip, mp in real:\n"
         "        m = imread_any(mp, cv2.IMREAD_GRAYSCALE)\n"
         "        fr.append({'tile': ip.name, 'damage_%': 100*(m>0).mean(), 'crack_%': 100*(m==1).mean(), 'loss_%': 100*(m==2).mean(), 'stain_%': 100*(m==3).mean(), 'unknown_%': 100*(m==255).mean()})\n"
         "    fr = pd.DataFrame(fr); display(fr.describe().round(2))\n"
         "    print('Class imbalance: damage covers on average %.1f%% of pixels -> why Dice+Focal loss is used.' % fr['damage_%'].mean())"),
    md("## 2. Synthetic degradation samples\nColumns: clean | synthetic damage | labels (red crack, blue loss, yellow stain) | synthetic fading."),
    code("rng = np.random.default_rng(0); paths = list_images(CLEAN_DIR)\n"
         "cfg = SynthConfig(damage_bank=bank, p_clean=0.0)\n"
         "fig, ax = plt.subplots(4, 4, figsize=(14, 14))\n"
         "for r in range(4):\n"
         "    c = random_crop(imread_rgb(paths[rng.integers(len(paths))]), TILE, rng)\n"
         "    d, lab = synthesize(c, rng, cfg); ov = d.copy()\n"
         "    for k, col in enumerate([(255,0,0),(0,90,255),(255,220,0)]):\n"
         "        ov[lab[...,k]>0] = col\n"
         "    for j, im in enumerate([c, d, ov, synthetic_fade(c, rng)]):\n"
         "        ax[r, j].imshow(im); ax[r, j].axis('off')\n"
         "plt.tight_layout()"),
    md("## 3. Segmentation training curves\n**Real F1** is what matters; a large gap between synthetic and real F1 indicates a synthetic-to-real domain gap."),
    code("log = CKPT_DIR / 'seg_log.csv'\n"
         "if log.exists():\n"
         "    df = pd.read_csv(log)\n"
         "    fig, ax = plt.subplots(1, 2, figsize=(13, 4))\n"
         "    ax[0].plot(df.epoch, df.train_loss, 'o-'); ax[0].set_title('train loss (Dice + Focal)'); ax[0].set_xlabel('epoch')\n"
         "    ax[1].plot(df.epoch, df.synth_f1, 'o-', label='synthetic val F1')\n"
         "    if df.real_f1.notna().any(): ax[1].plot(df.epoch, df.real_f1, 's-', label='REAL val F1')\n"
         "    ax[1].plot(df.epoch, df.real_iou, '^--', label='REAL val IoU'); ax[1].legend(); ax[1].set_title('validation'); ax[1].set_xlabel('epoch')\n"
         "    display(df.tail())\n"
         "else:\n    print('No seg_log.csv yet - run scripts/train_seg.py')"),
    md("## 4. Real-tile evaluation: classical baseline vs U-Net (per class)"),
    code("from restoration.dataset import split_paths\n"
         "from restoration.segmentation import load_seg_model, predict_probs, CHANNELS\n"
         "def prf(pred, gt):\n"
         "    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()\n"
         "    return tp, fp, fn\n"
         "def summarize(c):\n"
         "    tp, fp, fn = c\n"
         "    return {'precision': tp/max(1,tp+fp), 'recall': tp/max(1,tp+fn), 'F1': 2*tp/max(1,2*tp+fp+fn), 'IoU': tp/max(1,tp+fp+fn)}\n"
         "ck_path = CKPT_DIR / 'seg_best.pt'\n"
         "_, val_pairs = split_paths(real, 0.3, seed=0)   # same split as train_seg.py\n"
         "if val_pairs and ck_path.exists():\n"
         "    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
         "    model, ck = load_seg_model(ck_path, dev); thr = ck['threshold']\n"
         "    acc = {k: np.zeros(3) for k in ['classical (any)', 'unet (any)', 'unet crack', 'unet loss', 'unet stain']}\n"
         "    examples = []\n"
         "    for ip, mp in val_pairs:\n"
         "        img, m = imread_rgb(ip), imread_any(mp, cv2.IMREAD_GRAYSCALE)\n"
         "        gt = m > 0\n"
         "        acc['classical (any)'] += prf(classical_crack_mask(img) > 0, gt)\n"
         "        probs, _ = predict_probs(model, img, dev, ck.get('use_clahe', True))\n"
         "        acc['unet (any)'] += prf(probs[..., 0] > thr, gt)\n"
         "        for ci, name in enumerate(['crack', 'loss', 'stain'], 1):\n"
         "            if not (m == 255).any():  # per-class only on tiles whose damage types were labelled\n"
         "                acc[f'unet {name}'] += prf(probs[..., ci] > thr, m == ci)\n"
         "        examples.append((img, gt, probs[..., 0] > thr))\n"
         "    acc = {k: v for k, v in acc.items() if v[0] + v[2] > 0}  # drop classes with no labelled pixels\n"
         "    display(pd.DataFrame({k: summarize(v) for k, v in acc.items()}).T.round(3))\n"
         "    n = min(4, len(examples)); fig, ax = plt.subplots(n, 3, figsize=(12, 4*n), squeeze=False)\n"
         "    for i, (img, gt, pr) in enumerate(examples[:n]):\n"
         "        for j, (im, t) in enumerate([(img, 'real tile'), (gt, 'ground truth'), (pr, 'U-Net prediction')]):\n"
         "            ax[i, j].imshow(im, cmap='gray'); ax[i, j].set_title(t); ax[i, j].axis('off')\n"
         "    plt.tight_layout()\n"
         "else:\n    print('Need real labelled tiles (data/real_labeled) and checkpoints/seg_best.pt')"),
    md("## 5. Colour (de-fading) training curves\nPSNR of the model output vs the faded input; the model should sit clearly above the baseline."),
    code("log = CKPT_DIR / 'color_log.csv'\n"
         "if log.exists():\n"
         "    df = pd.read_csv(log)\n"
         "    fig, ax = plt.subplots(1, 2, figsize=(13, 4))\n"
         "    ax[0].plot(df.epoch, df.g_l1, 'o-', label='train L1'); ax[0].plot(df.epoch, df.val_l1, 's-', label='val L1'); ax[0].legend(); ax[0].set_title('L1 loss')\n"
         "    ax[1].plot(df.epoch, df.val_psnr, 'o-', label='de-faded'); ax[1].plot(df.epoch, df.baseline_psnr, '--', label='faded input (baseline)'); ax[1].legend(); ax[1].set_title('PSNR (dB)')\n"
         "else:\n    print('No color_log.csv yet - run scripts/train_color.py')"),
    md("## 6. Restoration reports\nSummary of `report.json` for every painting restored with `scripts/restore.py`."),
    code("reps = []\n"
         "for p in sorted((ROOT / 'outputs/restored').glob('*/report.json')):\n"
         "    r = json.loads(p.read_text()); r['painting'] = p.parent.name; reps.append(r)\n"
         "if reps:\n"
         "    display(pd.DataFrame(reps)[['painting', 'final_size', 'damaged_fraction', 'color_method', 'mean_confidence_in_infill', 'speculative_fraction_of_infill(conf<0.5)', 'total_s']])\n"
         "else:\n    print('No restored paintings yet.')"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "ai-restoration", "display_name": "AI Restoration (py3.11)", "language": "python"}})
out = ROOT / "notebooks/validation.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"Wrote {out}")
