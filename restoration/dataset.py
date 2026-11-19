"""Datasets for segmentation (Milestone 1) and de-fading (Milestone 3).

Segmentation samples come from two sources:
  1. Clean paintings (data/clean) + on-the-fly synthetic damage  -> unlimited pairs
  2. Hand-labelled REAL damaged tiles (data/real_labeled)         -> closes domain gap

Real mask PNG format (single channel):
    0 = background, 1 = crack, 2 = loss, 3 = stain, 255 = damage (type unknown)
Painting with 255 is fine -- the model's "any damage" head is supervised by
it, the per-class heads are ignored for those pixels.
"""
from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import seg_preprocess
from .synth_damage import SynthConfig, synthesize, synthetic_fade
from .utils import imread_any, imread_rgb, list_images, resize_max_side


def random_crop(img: np.ndarray, size: int, rng, scale_range=(0.5, 1.5), mask: np.ndarray | None = None):
    """Random crop with random rescale (teaches scale invariance -> robust to scan DPI)."""
    s = rng.uniform(*scale_range)
    src = int(round(size / s))
    h, w = img.shape[:2]
    if h < src or w < src:  # upscale small images
        f = src / min(h, w) + 1e-3
        img = cv2.resize(img, (int(w * f) + 1, int(h * f) + 1), interpolation=cv2.INTER_CUBIC)
        if mask is not None:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        h, w = img.shape[:2]
    y, x = int(rng.integers(0, h - src + 1)), int(rng.integers(0, w - src + 1))
    crop = cv2.resize(img[y : y + src, x : x + src], (size, size), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    if mask is None:
        return crop
    mcrop = cv2.resize(mask[y : y + src, x : x + src], (size, size), interpolation=cv2.INTER_NEAREST)
    return crop, mcrop


def geo_aug(arrs: list[np.ndarray], rng) -> list[np.ndarray]:
    k = int(rng.integers(4))
    flip = rng.random() < 0.5
    out = []
    for a in arrs:
        a = np.rot90(a, k)
        if flip:
            a = a[:, ::-1]
        out.append(np.ascontiguousarray(a))
    return out


def color_aug(img: np.ndarray, rng) -> np.ndarray:
    """Mild hue/sat jitter on CLEAN images before damage (more palette variety)."""
    if rng.random() < 0.5:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
    hsv[..., 1] *= rng.uniform(0.8, 1.2)
    return cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def real_mask_to_targets(m: np.ndarray):
    """single-channel real mask -> (target (4,H,W), weight (4,H,W))
    channels: [any, crack, loss, stain]"""
    any_ = (m > 0).astype(np.float32)
    tgt = np.stack([any_, (m == 1), (m == 2), (m == 3)]).astype(np.float32)
    w = np.ones_like(tgt)
    unknown = m == 255
    w[1:, unknown] = 0.0  # type unknown -> don't supervise class heads there
    return tgt, w


def load_real_pairs(root: str | Path):
    root = Path(root)
    pairs = []
    for ip in list_images(root / "images"):
        mp = root / "masks" / (ip.stem + ".png")
        if mp.exists():
            pairs.append((ip, mp))
    return pairs


class SegDataset(Dataset):
    def __init__(
        self,
        clean_paths: list,
        real_pairs: list,
        tile: int = 384,
        length: int = 2000,
        p_real: float = 0.3,
        synth_cfg: SynthConfig | None = None,
        use_clahe: bool = True,
        seed: int | None = None,
        cache: bool = True,
        max_side: int = 3000,
    ):
        self.max_side = max_side
        self.clean_paths, self.real_pairs = list(clean_paths), list(real_pairs)
        if not self.clean_paths and not self.real_pairs:
            raise ValueError("No training data: add images to data/clean and/or data/real_labeled")
        self.tile, self.length = tile, length
        self.p_real = p_real if self.real_pairs else 0.0
        if not self.clean_paths:
            self.p_real = 1.0
        self.cfg = synth_cfg or SynthConfig()
        self.use_clahe, self.seed = use_clahe, seed
        self._cache = {} if cache else None

    def __len__(self):
        return self.length

    def _load(self, p, kind):
        """Images are capped at `max_side` px so caching dozens of large scans in
        every DataLoader worker doesn't exhaust RAM. Masks get the identical resize."""
        if self._cache is not None and p in self._cache:
            return self._cache[p]
        if kind == "rgb":
            x, _ = resize_max_side(imread_rgb(p), self.max_side)
        else:
            x, _ = resize_max_side(imread_any(p, cv2.IMREAD_GRAYSCALE), self.max_side, interp=cv2.INTER_NEAREST)
        if self._cache is not None:
            self._cache[p] = x
        return x

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.seed is None else self.seed + i)
        if rng.random() < self.p_real:
            ip, mp = self.real_pairs[rng.integers(len(self.real_pairs))]
            img, m = random_crop(self._load(ip, "rgb"), self.tile, rng, (0.7, 1.3), self._load(mp, "gray"))
            img, m = geo_aug([img, m], rng)
            tgt, w = real_mask_to_targets(m)
            is_real = 1.0
        else:
            clean = random_crop(self._load(self.clean_paths[rng.integers(len(self.clean_paths))], "rgb"), self.tile, rng)
            clean = color_aug(geo_aug([clean], rng)[0], rng)
            img, lab = synthesize(clean, rng, self.cfg)
            lab = lab.transpose(2, 0, 1).astype(np.float32)
            tgt = np.concatenate([lab.max(0, keepdims=True), lab], 0)
            w = np.ones_like(tgt)
            is_real = 0.0
        x = seg_preprocess(img, self.use_clahe).transpose(2, 0, 1)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x)),
            "target": torch.from_numpy(tgt),
            "weight": torch.from_numpy(w),
            "is_real": torch.tensor(is_real),
            "raw": torch.from_numpy(np.ascontiguousarray(img)),
        }


class RealValDataset(Dataset):
    """Deterministic tiles from real labelled images for validation."""

    def __init__(self, real_pairs: list, tile: int = 384, use_clahe: bool = True):
        self.items = []
        for ip, mp in real_pairs:
            img, m = imread_rgb(ip), imread_any(mp, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            if h < tile or w < tile:
                f = tile / min(h, w) + 1e-3
                img = cv2.resize(img, (int(w * f) + 1, int(h * f) + 1), interpolation=cv2.INTER_CUBIC)
                m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                h, w = img.shape[:2]
            for y in range(0, h - tile + 1, tile):
                for x in range(0, w - tile + 1, tile):
                    self.items.append((img[y : y + tile, x : x + tile], m[y : y + tile, x : x + tile]))
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, m = self.items[i]
        tgt, w = real_mask_to_targets(m)
        x = seg_preprocess(img, self.use_clahe).transpose(2, 0, 1)
        return {"image": torch.from_numpy(np.ascontiguousarray(x)), "target": torch.from_numpy(tgt), "weight": torch.from_numpy(w)}


class FadeDataset(Dataset):
    """(faded, original) pairs for supervised de-fading (Pix2Pix)."""

    def __init__(self, clean_paths: list, tile: int = 256, length: int = 2000, p_identity: float = 0.1, seed: int | None = None):
        if not clean_paths:
            raise ValueError("No clean images in data/clean")
        self.paths, self.tile, self.length, self.p_identity, self.seed = list(clean_paths), tile, length, p_identity, seed
        self._cache = {}

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.seed is None else self.seed + i)
        p = self.paths[rng.integers(len(self.paths))]
        if p not in self._cache:
            self._cache[p] = resize_max_side(imread_rgb(p), 3000)[0]
        clean = random_crop(self._cache[p], self.tile, rng, (0.5, 1.2))
        clean = geo_aug([clean], rng)[0]
        # p_identity: un-faded input should come out unchanged (no over-saturation)
        faded = clean.copy() if rng.random() < self.p_identity else synthetic_fade(clean, rng)
        to_t = lambda a: torch.from_numpy(a.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0)
        return {"faded": to_t(faded), "clean": to_t(clean)}


def painting_key(item) -> str:
    """Source painting of a tile: '0105_kalighat_y512_x0' -> '0105_kalighat'."""
    p = Path(item[0] if isinstance(item, (tuple, list)) else item)
    return re.sub(r"_y\d+_x\d+$", "", p.stem)


def split_paths(paths: list, val_frac: float, seed: int = 0):
    """Train/val split by SOURCE PAINTING (tiles of one painting never straddle the
    split, which would inflate validation scores). Assignment is a stable hash of
    the painting name, so adding new labels never reshuffles existing ones."""
    paths = list(paths)
    if val_frac <= 0 or len(paths) < 2:
        return paths, []

    def in_val(key):
        h = int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return h < val_frac

    val = [p for p in paths if in_val(painting_key(p))]
    if not val:  # guarantee at least one validation painting
        first = sorted({painting_key(p) for p in paths})[0]
        val = [p for p in paths if painting_key(p) == first]
    val_set = {id(p) for p in val}
    return [p for p in paths if id(p) not in val_set], val
