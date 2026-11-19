"""Milestone 1: train the damage segmentation U-Net.

    python scripts/train_seg.py                       # defaults
    python scripts/train_seg.py --epochs 60 --batch 6 --tile 384

Data:
    data/clean/                 clean / well-preserved paintings (or clean crops)
    data/real_labeled/images    real damaged tiles      } created with
    data/real_labeled/masks     their masks (0/1/2/3/255) } make_label_tiles + label_tool
    data/damage_bank/           optional real damage cut-outs (build_damage_bank.py)

Model selection uses F1 on REAL labelled validation tiles whenever they exist --
synthetic validation scores are misleading (that is how the old model looked
good but failed on real paintings).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from restoration.dataset import RealValDataset, SegDataset, load_real_pairs, split_paths
from restoration.segmentation import SegLoss, build_seg_model, confusion_at_thresholds, f1_iou, save_seg_ckpt
from restoration.synth_damage import SynthConfig, load_damage_bank
from restoration.utils import imwrite, list_images, overlay_mask

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(ROOT / "data/clean"))
    ap.add_argument("--real", default=str(ROOT / "data/real_labeled"))
    ap.add_argument("--bank", default=str(ROOT / "data/damage_bank"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints"))
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=400, help="training steps per epoch")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--p-real", type=float, default=0.3, help="fraction of batches drawn from real labelled tiles")
    ap.add_argument("--real-val-frac", type=float, default=0.3)
    ap.add_argument("--synth-val", type=int, default=160)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--resume", default=None)
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    conf = torch.zeros(len(THRESHOLDS), 3, device=device)
    for b in loader:
        x, t = b["image"].to(device), b["target"].to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x).float())
        conf += confusion_at_thresholds(p[:, 0], t[:, 0], THRESHOLDS)
    f1, iou = f1_iou(conf)
    i = int(f1.argmax())
    return float(f1[i]), float(iou[i]), THRESHOLDS[i]


@torch.no_grad()
def save_vis(model, ds, device, path, n=4):
    if len(ds) == 0:
        return
    model.eval()
    rows = []
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    for i in np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int):
        b = ds[i]
        p = torch.sigmoid(model(b["image"][None].to(device)))[0, 0].cpu().numpy()
        img = ((b["image"].numpy().transpose(1, 2, 0) * std + mean).clip(0, 1) * 255).astype(np.uint8)
        gt = overlay_mask(img, b["target"][0].numpy() > 0.5, (0, 255, 0))
        pr = overlay_mask(img, p > 0.5, (255, 0, 0))
        heat = np.repeat((p * 255).astype(np.uint8)[..., None], 3, -1)
        rows.append(np.concatenate([img, gt, pr, heat], 1))
    imwrite(path, np.concatenate(rows, 0))


def main():
    a = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vis_dir = ROOT / "outputs/train_vis"

    clean = list_images(a.clean)
    real = load_real_pairs(a.real)
    real_tr, real_val = split_paths(real, a.real_val_frac, seed=0) if real else ([], [])
    bank = load_damage_bank(a.bank)
    print(f"clean images: {len(clean)} | real labelled: {len(real)} (train {len(real_tr)}, val {len(real_val)}) | bank items: {len(bank)}")
    if not real_val:
        print("WARNING: no real labelled validation tiles. Model selection will use SYNTHETIC validation,\n"
              "         which does NOT tell you whether it works on real paintings. Label ~20 real tiles!")

    use_clahe = not a.no_clahe
    cfg = SynthConfig(damage_bank=bank)
    train_ds = SegDataset(clean, real_tr, a.tile, a.steps * a.batch, a.p_real, cfg, use_clahe)
    synth_val_ds = SegDataset(clean, [], a.tile, a.synth_val, 0.0, SynthConfig(damage_bank=bank), use_clahe, seed=12345) if clean else None
    real_val_ds = RealValDataset(real_val, a.tile, use_clahe) if real_val else None

    dl_kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, drop_last=True, **dl_kw)
    synth_dl = DataLoader(synth_val_ds, batch_size=a.batch, **dl_kw) if synth_val_ds else None
    real_dl = DataLoader(real_val_ds, batch_size=a.batch, num_workers=0) if real_val_ds else None

    model = build_seg_model(a.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    crit = SegLoss()
    start, best = 0, -1.0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        start, best = ck.get("epoch", 0) + 1, ck.get("score", -1.0)

    log = out / "seg_log.csv"
    new_log = not log.exists() or not a.resume
    with open(log, "w" if new_log else "a", newline="") as fh:
        w = csv.writer(fh)
        if new_log:
            w.writerow(["epoch", "train_loss", "synth_f1", "synth_iou", "synth_thr", "real_f1", "real_iou", "real_thr", "lr", "time_s"])
        for ep in range(start, a.epochs):
            model.train()
            t0, tot, n = time.time(), 0.0, 0
            for i, b in enumerate(train_dl):
                x, t, wt = b["image"].to(device, non_blocking=True), b["target"].to(device), b["weight"].to(device)
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    logits = model(x)
                loss = crit(logits.float(), t, wt)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
                tot, n = tot + loss.item(), n + 1
                if i % 50 == 0:
                    print(f"  ep {ep} step {i}/{len(train_dl)} loss {loss.item():.4f}", flush=True)

            s_f1, s_iou, s_thr = evaluate(model, synth_dl, device) if synth_dl else (float("nan"),) * 3
            r_f1, r_iou, r_thr = evaluate(model, real_dl, device) if real_dl else (float("nan"),) * 3
            score, thr = (r_f1, r_thr) if real_dl else (s_f1, s_thr)
            lr = opt.param_groups[0]["lr"]
            print(f"epoch {ep}: loss {tot / max(n, 1):.4f} | synth F1 {s_f1:.3f} IoU {s_iou:.3f} @{s_thr} | "
                  f"REAL F1 {r_f1:.3f} IoU {r_iou:.3f} @{r_thr} | {time.time() - t0:.0f}s", flush=True)
            w.writerow([ep, tot / max(n, 1), s_f1, s_iou, s_thr, r_f1, r_iou, r_thr, lr, time.time() - t0])
            fh.flush()

            extra = {"epoch": ep, "score": score, "real_f1": r_f1, "synth_f1": s_f1}
            save_seg_ckpt(out / "seg_last.pt", model, a.encoder, thr, use_clahe, extra)
            if score > best:
                best = score
                save_seg_ckpt(out / "seg_best.pt", model, a.encoder, thr, use_clahe, extra)
                print(f"  -> new best ({score:.3f}), threshold {thr}")
            save_vis(model, real_val_ds if real_val_ds else synth_val_ds, device, vis_dir / f"epoch_{ep:03d}.png")
    print(f"Done. Best score {best:.3f}. Checkpoint: {out / 'seg_best.pt'}")


if __name__ == "__main__":
    main()
