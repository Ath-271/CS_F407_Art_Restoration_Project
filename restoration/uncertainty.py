"""Milestone 4b: spatial confidence map for conservators.

Per-pixel confidence in [0, 1] (1 = original / verified, 0 = speculative) from
three independent signals:

  1. Generative disagreement - LaMa is re-run on perturbed inputs (horizontal
     flip, mask dilated by +4 / +8 px). Where the reconstructions disagree, the
     model is guessing.  (Monte-Carlo style test-time ensemble.)
  2. Distance from known pixels - deep inside a large lacuna there is no local
     evidence; the infill is compositional guesswork.
  3. Detection ambiguity - segmentation probability near the threshold and
     disagreement across segmentation TTA.

Outside the damage mask pixels are original; confidence stays high except
where the detector itself was unsure.
"""
from __future__ import annotations

import cv2
import numpy as np


def inpaint_disagreement(inpainter, img: np.ndarray, mask: np.ndarray, base: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    outs = [base.astype(np.float32)]
    # flipped
    f = inpainter.inpaint(img[:, ::-1].copy(), mask[:, ::-1].copy())[:, ::-1]
    outs.append(f.astype(np.float32))
    for d in (4, 8):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
        outs.append(inpainter.inpaint(img, cv2.dilate(mask, k)).astype(np.float32))
    labs = [cv2.cvtColor(o.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) for o in outs]
    std = np.stack(labs).std(0).mean(-1)  # mean over L,a,b
    return cv2.GaussianBlur(std, (0, 0), 1.5)


def confidence_map(
    mask: np.ndarray,
    prob_any: np.ndarray | None = None,
    seg_tta_std: np.ndarray | None = None,
    threshold: float = 0.5,
    inpaint_std: np.ndarray | None = None,
    std_scale: float = 12.0,
    dist_scale: float = 12.0,
) -> np.ndarray:
    m = mask > 127
    h, w = mask.shape
    # detection ambiguity
    amb = np.zeros((h, w), np.float32)
    if prob_any is not None:
        amb = np.exp(-(((prob_any - threshold) / 0.15) ** 2)).astype(np.float32)
    if seg_tta_std is not None:
        amb = np.maximum(amb, np.clip(seg_tta_std / 0.25, 0, 1))
    # distance into the hole
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    dist_n = 1 - np.exp(-dist / dist_scale)
    var_n = np.zeros((h, w), np.float32) if inpaint_std is None else 1 - np.exp(-inpaint_std / std_scale)

    unc_in = np.clip(0.45 * var_n + 0.4 * dist_n + 0.15 * amb, 0, 1)
    unc_out = 0.4 * amb
    unc = np.where(m, np.maximum(unc_in, 0.15), unc_out)  # any infill is at least slightly uncertain
    return (1 - unc).astype(np.float32)


def confidence_heatmap(img: np.ndarray, conf: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Overlay: green = confident/original ... red = speculative infill."""
    c8 = (np.clip(conf, 0, 1) * 255).astype(np.uint8)
    # map: 0 -> red, 255 -> green via HSV hue 0..60 (OpenCV hue units)
    hsv = np.stack([(c8.astype(np.float32) * 60 / 255).astype(np.uint8), np.full_like(c8, 255), np.full_like(c8, 255)], -1)
    col = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
    a = (alpha * (1 - conf) + 0.1)[..., None]  # show mostly the uncertain parts
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB).astype(np.float32)
    return (gray * (1 - a) + col * a).clip(0, 255).astype(np.uint8)
