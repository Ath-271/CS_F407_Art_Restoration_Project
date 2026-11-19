"""Milestone 4a: deep super-resolution with Real-ESRGAN (RRDBNet).

The RRDBNet architecture is re-implemented here (identical parameter names to
BasicSR) so we can load the official RealESRGAN_x4plus.pth weights without the
`basicsr` package (which is broken on recent torchvision).

Memory: inference runs on overlapping 256px tiles in fp16, blended with a
cosine window, so an 8 GB GPU can upscale arbitrarily large canvases without
seams and without errors at tile borders propagating.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import process_tiled


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SuperResolver:
    SCALE = 4

    def __init__(self, weights: str | Path = "weights/RealESRGAN_x4plus.pth", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"{weights} not found - run: python scripts/download_weights.py")
        sd = torch.load(str(weights), map_location="cpu", weights_only=True)
        sd = sd.get("params_ema", sd.get("params", sd))
        self.model = RRDBNet()
        self.model.load_state_dict(sd, strict=True)
        self.model.eval().to(self.device)
        self.half = self.device.type == "cuda"
        if self.half:
            self.model.half()

    @torch.no_grad()
    def upscale(self, img: np.ndarray, outscale: float = 4.0, tile: int = 256, overlap: int = 24) -> np.ndarray:
        def run(t):
            x = torch.from_numpy(t.transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(self.device)
            if self.half:
                x = x.half()
            y = self.model(x).float().clamp(0, 1)
            return y[0].permute(1, 2, 0).cpu().numpy() * 255.0

        out = process_tiled(img, run, tile=tile, overlap=overlap, out_channels=3, scale=self.SCALE)
        out = out.clip(0, 255).astype(np.uint8)
        if outscale != self.SCALE:
            h, w = img.shape[:2]
            out = cv2.resize(out, (round(w * outscale), round(h * outscale)), interpolation=cv2.INTER_AREA)
        return out
