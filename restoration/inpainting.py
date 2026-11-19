"""Milestone 2: contextual generative inpainting with LaMa (Fast Fourier Convolutions).

Uses the pretrained Big-LaMa TorchScript model (weights/big-lama.pt, fetched by
scripts/download_weights.py). FFCs have an image-wide receptive field, which is
what lets repetitive Madhubani borders / Warli patterns continue across a hole.

Large paintings: LaMa runs on overlapping windows centred on damage only, and
results are composited back ONLY inside the mask -- undamaged pixels are never
altered (important for authenticity).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .utils import process_tiled


class LamaInpainter:
    def __init__(self, weights: str | Path = "weights/big-lama.pt", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"{weights} not found - run: python scripts/download_weights.py")
        self.model = torch.jit.load(str(weights), map_location=self.device).eval()

    @torch.no_grad()
    def _run(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """img uint8 RGB (h,w,3), mask uint8 {0,255} -> float RGB 0..255."""
        h, w = img.shape[:2]
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        im = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
        mk = np.pad(mask, ((0, ph), (0, pw)), mode="symmetric")
        t_img = torch.from_numpy(im).permute(2, 0, 1)[None].float().div(255).to(self.device)
        t_mask = torch.from_numpy((mk > 127).astype(np.float32))[None, None].to(self.device)
        out = self.model(t_img, t_mask)[0].permute(1, 2, 0).float().cpu().numpy()
        return (out[:h, :w] * 255).clip(0, 255)

    def inpaint(self, img: np.ndarray, mask: np.ndarray, window: int = 1024, overlap: int = 256) -> np.ndarray:
        """Inpaint an arbitrarily large image. Returns uint8 RGB."""
        if not mask.any():
            return img.copy()
        h, w = img.shape[:2]
        if max(h, w) <= window:
            res = self._run(img, mask)
        else:
            both = np.concatenate([img, mask[..., None]], -1)
            res = process_tiled(
                both,
                lambda t: self._run(t[..., :3], t[..., 3]),
                tile=window,
                overlap=overlap,
                out_channels=3,
                skip=lambda y, x, th, tw: not mask[y : y + th, x : x + tw].any(),
                default=img.astype(np.float32),
            )
        # composite only inside (feathered) mask
        a = cv2.GaussianBlur((mask > 127).astype(np.float32), (0, 0), 1.0)
        a = np.maximum(a, (mask > 127).astype(np.float32))[..., None]
        return (img * (1 - a) + res * a).clip(0, 255).astype(np.uint8)


class OpenCVInpainter:
    """Fallback (no deep model): Telea fast marching. Fine for thin cracks only."""

    def inpaint(self, img: np.ndarray, mask: np.ndarray, **_) -> np.ndarray:
        return cv2.inpaint(img, (mask > 127).astype(np.uint8), 5, cv2.INPAINT_TELEA)


def get_inpainter(weights="weights/big-lama.pt", device=None):
    try:
        return LamaInpainter(weights, device)
    except Exception as e:  # pragma: no cover
        print(f"[inpainting] LaMa unavailable ({e}); falling back to OpenCV Telea.")
        return OpenCVInpainter()
