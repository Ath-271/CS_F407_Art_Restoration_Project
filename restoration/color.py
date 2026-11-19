"""Milestone 3: digital colour reconstruction / de-fading.

Two methods:
  * Pix2Pix (supervised): U-Net generator (pretrained encoder, residual output)
    + 70x70 PatchGAN discriminator, trained on (synthetic_fade(clean), clean).
    Residual output = the network predicts a colour CORRECTION, so linework and
    texture pass through untouched (no "artificial bleeding").
  * Reference transfer (classical, unsupervised): match LAB chroma statistics
    of the faded painting to a reference -- e.g. a protected margin that was
    under the frame, or a well-preserved painting from the same school.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from .utils import process_tiled


# ============================================================================ networks
class ResidualGenerator(nn.Module):
    def __init__(self, encoder: str = "resnet34", pretrained: bool = True):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=3)
        self.encoder = encoder
        # zero-init the output conv: the generator starts as an exact identity and
        # only learns corrections (stable start, never worse than the input early on)
        head = self.net.segmentation_head[0]
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)

    def forward(self, x):  # x in [-1, 1]
        return (x + torch.tanh(self.net(x))).clamp(-1, 1)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN on concat(input, output). Its per-patch score map is also
    usable as a spatial 'realism' signal."""

    def __init__(self, in_ch: int = 6, nf: int = 64):
        super().__init__()
        sn = nn.utils.spectral_norm

        def block(i, o, s, norm=True):
            layers = [sn(nn.Conv2d(i, o, 4, s, 1))]
            if norm:
                layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        self.net = nn.Sequential(
            *block(in_ch, nf, 2, False),
            *block(nf, nf * 2, 2),
            *block(nf * 2, nf * 4, 2),
            *block(nf * 4, nf * 8, 1),
            nn.Conv2d(nf * 8, 1, 4, 1, 1),
        )

    def forward(self, a, b):
        return self.net(torch.cat([a, b], 1))


def lab_chroma_loss(fake, real):
    """L1 on a/b chroma in an approximate (differentiable) opponent space."""
    def ab(x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        return torch.stack([r - g, 0.5 * (r + g) - b], 1)

    return (ab(fake) - ab(real)).abs().mean()


def save_color_ckpt(path, G, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": G.state_dict(), "encoder": G.encoder, **(extra or {})}, path)


def load_color_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    G = ResidualGenerator(ck["encoder"], pretrained=False)
    G.load_state_dict(ck["state_dict"])
    return G.to(device).eval()


@torch.no_grad()
def defade_pix2pix(G, img: np.ndarray, device, tile: int = 512, overlap: int = 64, strength: float = 1.0,
                   adaptive: bool = True, return_info: bool = False):
    use_amp = device.type == "cuda"

    def run(t):
        x = torch.from_numpy(t.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1)[None].to(device)
        with torch.autocast(device.type, enabled=use_amp):
            y = G(x).float()
        return (y[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5

    out = process_tiled(img, run, tile=tile, overlap=overlap, out_channels=3)
    out = img.astype(np.float32) * (1 - strength) + out * strength
    out = out.clip(0, 255).astype(np.uint8)
    info = {"fade_score": None}
    if adaptive:
        out, info = limit_color_change(img, out)
    return (out, info) if return_info else out


# Colour statistics of the clean training set (300 random paintings, 512 px) vs the
# same paintings after synthetic_fade. Used to judge how faded a new input is.
CLEAN_REF = {"chroma_p90": 51.6, "l_range": 71.4}
FADED_REF = {"chroma_p90": 31.0, "l_range": 52.9}


def fade_score(img: np.ndarray) -> float:
    """0 = as vivid/contrasty as a typical clean painting, 1 = as dull as a typical
    synthetically faded one. Based on 90th-pct LAB chroma and the 5-95% L range."""
    small = img if max(img.shape[:2]) <= 512 else cv2.resize(img, None, fx=512 / max(img.shape[:2]), fy=512 / max(img.shape[:2]), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.percentile(np.hypot(lab[..., 1] - 128, lab[..., 2] - 128), 90)
    L = lab[..., 0] * 100 / 255
    l_range = np.percentile(L, 95) - np.percentile(L, 5)
    s = [(CLEAN_REF[k] - v) / (CLEAN_REF[k] - FADED_REF[k]) for k, v in (("chroma_p90", chroma), ("l_range", l_range))]
    return float(np.clip(np.mean(s), 0, 1))


def limit_color_change(src: np.ndarray, out: np.ndarray, base_gain: float = 1.05, extra_gain: float = 0.35,
                       base_cast: float = 1.0, extra_cast: float = 2.0):
    """Keep de-fading plausible: a well-preserved painting gets (almost) no change,
    a faded one may gain up to base_gain+extra_gain x chroma. A global colour cast
    (mean a/b shift = re-tinting the whole image) is capped at base_cast+extra_cast*fade
    LAB units. Lightness/contrast changes from the model are kept."""
    f = fade_score(src)
    max_gain, max_cast = base_gain + extra_gain * f, base_cast + extra_cast * f
    lab_s = cv2.cvtColor(src.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    lab_o = cv2.cvtColor(out.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    d = lab_o[..., 1:] - lab_s[..., 1:]
    cast = d.reshape(-1, 2).mean(0)
    n = float(np.hypot(*cast))
    if n > max_cast:  # remove the excess part of the global tint
        d -= cast * (1 - max_cast / n)
    c_in = np.hypot(lab_s[..., 1], lab_s[..., 2]).mean()
    c_out = np.hypot(lab_s[..., 1] + d[..., 0], lab_s[..., 2] + d[..., 1]).mean()
    k = 1.0
    if c_out > c_in * max_gain and c_out > c_in:
        k = float(np.clip((c_in * max_gain - c_in) / (c_out - c_in), 0, 1))
        d *= k
    lab_o[..., 1:] = lab_s[..., 1:] + d
    res = (cv2.cvtColor(lab_o, cv2.COLOR_LAB2RGB) * 255).clip(0, 255).astype(np.uint8)
    return res, {"fade_score": round(f, 3), "chroma_gain_cap": round(max_gain, 3), "cast_removed": round(max(0.0, n - max_cast), 2),
                 "chroma_scale": round(k, 3)}


# ============================================================================ classical
def reference_color_transfer(img: np.ndarray, ref: np.ndarray, strength: float = 1.0, keep_luminance: bool = True) -> np.ndarray:
    """Reinhard-style LAB statistics transfer (chroma only by default)."""
    src = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = src.copy()
    chans = (1, 2) if keep_luminance else (0, 1, 2)
    for c in chans:
        ms, ss = src[..., c].mean(), src[..., c].std() + 1e-6
        mr, sr = r[..., c].mean(), r[..., c].std() + 1e-6
        out[..., c] = (src[..., c] - ms) * (sr / ss) + mr
    out = src * (1 - strength) + out * strength
    return cv2.cvtColor(out.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def auto_saturation_boost(img: np.ndarray, target_chroma: float = 38.0, max_gain: float = 1.8) -> np.ndarray:
    """Very simple fallback if no model/reference is given: scale LAB chroma so the
    95th-percentile chroma reaches a typical folk-painting level."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    chroma = np.sqrt(a**2 + b**2)
    g = np.clip(target_chroma / (np.percentile(chroma, 95) + 1e-6), 1.0, max_gain)
    lab[..., 1], lab[..., 2] = a * g + 128, b * g + 128
    return cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
