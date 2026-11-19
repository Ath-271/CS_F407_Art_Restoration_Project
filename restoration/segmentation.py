"""Milestone 1: damage segmentation network, losses, metrics and tiled inference.

Model: U-Net with an ImageNet-pretrained encoder (segmentation_models_pytorch).
A pretrained encoder is far better at telling painted linework from cracks than
a U-Net trained from scratch on a few dozen images.

Outputs 4 sigmoid channels: [any_damage, crack, loss, stain].
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import seg_preprocess
from .utils import process_tiled

CHANNELS = ("any", "crack", "loss", "stain")


def build_seg_model(encoder: str = "resnet34", pretrained: bool = True) -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=len(CHANNELS),
        decoder_attention_type="scse",
    )


# ----------------------------------------------------------------------------- losses
def focal_loss(logits, target, weight, alpha=0.75, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    a = alpha * target + (1 - alpha) * (1 - target)
    loss = a * (1 - pt) ** gamma * ce * weight
    return loss.sum() / weight.sum().clamp_min(1.0)


def soft_dice_loss(logits, target, weight, eps=1.0):
    """Dice per (sample, channel); empty-target channels contribute via FP penalty."""
    p = torch.sigmoid(logits) * weight
    t = target * weight
    inter = (p * t).sum((2, 3))
    denom = p.sum((2, 3)) + t.sum((2, 3))
    dice = 1 - (2 * inter + eps) / (denom + eps)
    valid = weight.sum((2, 3)) > 0
    return dice[valid].mean() if valid.any() else logits.sum() * 0


class SegLoss(nn.Module):
    """Dice + Focal: robust to the heavy imbalance (damage is often <5% of pixels),
    which is what makes plain BCE collapse to "predict nothing" on real images."""

    def __init__(self, any_weight: float = 2.0):
        super().__init__()
        self.cw = torch.tensor([any_weight, 1.0, 1.0, 1.0]).view(1, -1, 1, 1)

    def forward(self, logits, target, weight):
        w = weight * self.cw.to(weight.device)
        return focal_loss(logits, target, w) * 10 + soft_dice_loss(logits, target, weight)


# ----------------------------------------------------------------------------- metrics
@torch.no_grad()
def confusion_at_thresholds(prob_any: torch.Tensor, target_any: torch.Tensor, thresholds):
    """Return tensor (len(thr), 3) of [TP, FP, FN] summed over batch."""
    out = []
    t = target_any > 0.5
    for th in thresholds:
        p = prob_any > th
        out.append(torch.stack([(p & t).sum(), (p & ~t).sum(), (~p & t).sum()]).float())
    return torch.stack(out)


def f1_iou(conf: torch.Tensor):
    tp, fp, fn = conf[..., 0], conf[..., 1], conf[..., 2]
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    iou = tp / (tp + fp + fn).clamp_min(1)
    return f1, iou


# ----------------------------------------------------------------------------- checkpoint
def save_seg_ckpt(path, model, encoder, threshold, use_clahe, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "encoder": encoder, "threshold": float(threshold), "use_clahe": use_clahe, **(extra or {})},
        path,
    )


def load_seg_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_seg_model(ck["encoder"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def predict_probs(
    model: nn.Module,
    img_rgb: np.ndarray,
    device,
    use_clahe: bool = True,
    tile: int = 512,
    overlap: int = 96,
    tta: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Tiled, blended prediction over a full painting.

    Returns (probs (H,W,4) float32, tta_std (H,W) float32 of the 'any' channel).
    TTA (8 dihedral transforms) improves recall on thin cracks and gives a
    per-pixel disagreement map used later for uncertainty.
    """
    # Preprocess the WHOLE image once (CLAHE needs global context; doing it per
    # tile would differ from how we'd see a full painting).
    x_full = seg_preprocess(img_rgb, use_clahe)
    use_amp = device.type == "cuda"

    def run(tile_x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(tile_x.transpose(2, 0, 1).copy())[None].to(device)
        variants = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
        outs = []
        for k, f in variants:
            v = torch.rot90(t, k, (2, 3))
            if f:
                v = torch.flip(v, (3,))
            with torch.autocast(device.type, enabled=use_amp):
                p = torch.sigmoid(model(v).float())
            if f:
                p = torch.flip(p, (3,))
            outs.append(torch.rot90(p, -k, (2, 3)))
        st = torch.stack(outs)  # (V,1,4,h,w)
        mean = st.mean(0)[0]
        std = st[:, 0, 0].std(0, unbiased=False) if len(outs) > 1 else torch.zeros_like(mean[0])
        return torch.cat([mean, std[None]], 0).permute(1, 2, 0).cpu().numpy()

    res = process_tiled(x_full, run, tile=tile, overlap=overlap, out_channels=5)
    return res[..., :4], res[..., 4]


def postprocess_mask(prob_any: np.ndarray, threshold: float, min_area: int = 12, dilate: int = 3) -> np.ndarray:
    """Threshold, remove specks, dilate slightly (inpainting needs a margin around defects)."""
    m = (prob_any > threshold).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    m = keep[lab]
    if dilate > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
    return (m * 255).astype(np.uint8)
