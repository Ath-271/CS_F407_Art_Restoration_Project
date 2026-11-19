"""Milestone 1 preprocessing: CLAHE in LAB space + a classical crack detector.

IMPORTANT: `seg_preprocess` is used by BOTH training and inference. Any mismatch
between the two (a common cause of "works on synthetic, fails on real") is
avoided by routing everything through this one function.
"""
from __future__ import annotations

import cv2
import numpy as np


def clahe_lab(img_rgb: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on the L channel only.

    Working in LAB boosts faint crack networks without shifting hues.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def seg_preprocess(img_rgb: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    """uint8 RGB -> float32 (H,W,3) normalised with ImageNet stats (pretrained encoder)."""
    x = clahe_lab(img_rgb) if use_clahe else img_rgb
    x = x.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return (x - mean) / std


def classical_crack_mask(
    img_rgb: np.ndarray,
    kernel_sizes=(7, 11, 15),
    percentile: float = 97.0,
    min_area: int = 30,
    min_elongation: float = 3.0,
) -> np.ndarray:
    """Non-learning baseline: morphological black-hat on CLAHE luminance.

    Highlights thin dark structures (cracks). It WILL also fire on painted
    linework -- use it only as a sanity check / starting point for hand labels.
    Returns uint8 mask {0,255}.
    """
    L = cv2.cvtColor(clahe_lab(img_rgb, clip=3.0), cv2.COLOR_RGB2LAB)[..., 0]
    resp = np.zeros(L.shape, np.float32)
    for k in kernel_sizes:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        resp = np.maximum(resp, cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, se).astype(np.float32))
    thr = np.percentile(resp, percentile)
    m = (resp > max(thr, 10)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        elong = max(w, h) / max(1.0, min(w, h))
        fill = area / max(1, w * h)
        if area >= min_area and (elong >= min_elongation or fill < 0.3):
            keep[i] = True
    return (keep[lab] * 255).astype(np.uint8)
