"""Milestone 3: train the Pix2Pix de-fading model on (synthetic_fade(clean), clean).

    python scripts/train_color.py --epochs 30

Generator: residual U-Net (pretrained encoder). Discriminator: 70x70 PatchGAN.
Loss: 100*L1 + 20*chroma L1 + LSGAN adversarial.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from restoration.color import PatchDiscriminator, ResidualGenerator, lab_chroma_loss, save_color_ckpt
from restoration.dataset import FadeDataset
from restoration.utils import imwrite, list_images


def to_img(t):
    return ((t.detach().float().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(ROOT / "data/clean"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints"))
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda-l1", type=float, default=100.0)
    ap.add_argument("--lambda-chroma", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean = list_images(a.clean)
    print(f"clean images: {len(clean)}")
    ds = FadeDataset(clean, a.tile, a.steps * a.batch)
    val = FadeDataset(clean, a.tile, 64, p_identity=0.0, seed=999)  # only truly faded samples
    dl = DataLoader(ds, a.batch, shuffle=True, drop_last=True, num_workers=a.workers, persistent_workers=a.workers > 0, pin_memory=True)
    vdl = DataLoader(val, a.batch, num_workers=0)

    G, D = ResidualGenerator(a.encoder).to(device), PatchDiscriminator().to(device)
    oG = torch.optim.Adam(G.parameters(), a.lr, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), a.lr, betas=(0.5, 0.999))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best = 1e9
    with open(out / "color_log.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["epoch", "g_l1", "g_adv", "d_loss", "val_l1", "val_psnr", "baseline_psnr", "time_s"])
        for ep in range(a.epochs):
            G.train(), D.train()
            t0, s = time.time(), np.zeros(3)
            for i, b in enumerate(dl):
                x, y = b["faded"].to(device), b["clean"].to(device)
                fake = G(x)
                # D step (LSGAN)
                d_real, d_fake = D(x, y), D(x, fake.detach())
                lD = 0.5 * (F.mse_loss(d_real, torch.ones_like(d_real)) + F.mse_loss(d_fake, torch.zeros_like(d_fake)))
                oD.zero_grad(set_to_none=True)
                lD.backward()
                oD.step()
                # G step
                d_fake = D(x, fake)
                adv = F.mse_loss(d_fake, torch.ones_like(d_fake))
                l1 = F.l1_loss(fake, y)
                lG = adv + a.lambda_l1 * l1 + a.lambda_chroma * lab_chroma_loss(fake, y)
                oG.zero_grad(set_to_none=True)
                lG.backward()
                oG.step()
                s += [l1.item(), adv.item(), lD.item()]
                if i % 50 == 0:
                    print(f"  ep {ep} step {i}/{len(dl)} L1 {l1.item():.4f} adv {adv.item():.3f} D {lD.item():.3f}", flush=True)
            # validation
            G.eval()
            vl1, psnr, base = [], [], []
            with torch.no_grad():
                for b in vdl:
                    x, y = b["faded"].to(device), b["clean"].to(device)
                    f = G(x)
                    vl1.append(F.l1_loss(f, y).item())
                    mse = ((f - y) / 2).pow(2).mean((1, 2, 3))
                    psnr += (10 * torch.log10(1 / mse.clamp_min(1e-10))).tolist()
                    mse0 = ((x - y) / 2).pow(2).mean((1, 2, 3))
                    base += (10 * torch.log10(1 / mse0.clamp_min(1e-10))).tolist()
                grid = np.concatenate([np.concatenate([to_img(x[k]), to_img(f[k]), to_img(y[k])], 1) for k in range(min(4, len(x)))], 0)
                imwrite(ROOT / f"outputs/color_vis/epoch_{ep:03d}.png", grid)
            s /= max(1, len(dl))
            v = float(np.mean(vl1))
            print(f"epoch {ep}: val L1 {v:.4f} | PSNR {np.mean(psnr):.2f} dB (faded input: {np.mean(base):.2f} dB) | {time.time() - t0:.0f}s")
            wr.writerow([ep, *s, v, np.mean(psnr), np.mean(base), time.time() - t0])
            fh.flush()
            save_color_ckpt(out / "color_last.pt", G, {"epoch": ep})
            if v < best:
                best = v
                save_color_ckpt(out / "color_best.pt", G, {"epoch": ep, "val_l1": v})
                torch.save(D.state_dict(), out / "color_disc_best.pt")
                print("  -> new best")


if __name__ == "__main__":
    main()
