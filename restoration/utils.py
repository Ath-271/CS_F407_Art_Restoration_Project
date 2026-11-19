"""Shared helpers: image I/O, tiling with overlap + smooth blending, noise fields."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


# ----------------------------------------------------------------------------- I/O
def list_images(folder: str | os.PathLike) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def imread_rgb(path: str | os.PathLike) -> np.ndarray:
    """Read image as uint8 RGB. Works with non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imread_any(path: str | os.PathLike, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img


def imwrite(path: str | os.PathLike, img: np.ndarray, rgb: bool = True) -> None:
    """Write image (RGB by default). Works with non-ASCII Windows paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = img
    if rgb and img.ndim == 3 and img.shape[2] == 3:
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif rgb and img.ndim == 3 and img.shape[2] == 4:
        out = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(path.suffix or ".png", out)
    if not ok:
        raise IOError(f"Could not encode image: {path}")
    buf.tofile(str(path))


def resize_max_side(img: np.ndarray, max_side: int | None, interp=cv2.INTER_AREA) -> tuple[np.ndarray, float]:
    """Downscale so the longest side <= max_side. Returns (image, scale)."""
    if not max_side:
        return img, 1.0
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img, 1.0
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp), s


# ----------------------------------------------------------------------------- noise
def smooth_noise(h: int, w: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency noise in [0, 1]; `scale` = approx feature size in pixels."""
    sh, sw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    small = rng.random((sh, sw)).astype(np.float32)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    big -= big.min()
    big /= big.max() + 1e-6
    return big


def fractal_noise(h: int, w: int, rng: np.random.Generator, base_scale: float = 64, octaves: int = 4) -> np.ndarray:
    out = np.zeros((h, w), np.float32)
    amp, total, scale = 1.0, 0.0, base_scale
    for _ in range(octaves):
        out += amp * smooth_noise(h, w, max(scale, 2), rng)
        total += amp
        amp *= 0.5
        scale /= 2
    out /= total
    out -= out.min()
    out /= out.max() + 1e-6
    return out


# ----------------------------------------------------------------------------- tiling
def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def blend_window(tile: int, overlap: int) -> np.ndarray:
    """2-D weight window: 1 in the centre, cosine ramp to ~0 over `overlap` px at edges."""
    ramp = np.ones(tile, np.float32)
    if overlap > 0:
        r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap + 2)[1:-1])
        ramp[:overlap] = r
        ramp[-overlap:] = r[::-1]
    return np.outer(ramp, ramp).astype(np.float32) + 1e-4


def process_tiled(
    img: np.ndarray,
    fn: Callable[[np.ndarray], np.ndarray],
    tile: int,
    overlap: int,
    out_channels: int,
    scale: int = 1,
    skip: Callable[[int, int, int, int], bool] | None = None,
    default: np.ndarray | None = None,
) -> np.ndarray:
    """Run `fn` on overlapping tiles of `img` (H,W,C) and blend outputs.

    fn(tile_img) -> (tile*scale, tile*scale, out_channels) float32.
    skip(y, x, th, tw) -> True to skip a tile (its region keeps `default`).
    Avoids seams by cosine-weighted blending, which also stops errors at tile
    borders from cascading (each pixel is a weighted vote of several tiles).
    """
    h, w = img.shape[:2]
    pad_h, pad_w = max(0, tile - h), max(0, tile - w)
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)) + ((0, 0),) * (img.ndim - 2), mode="reflect")
    H, W = img.shape[:2]
    stride = max(1, tile - overlap)
    acc = np.zeros((H * scale, W * scale, out_channels), np.float32)
    wsum = np.zeros((H * scale, W * scale, 1), np.float32)
    win = blend_window(tile * scale, overlap * scale)[..., None]
    for y in _tile_starts(H, tile, stride):
        for x in _tile_starts(W, tile, stride):
            if skip is not None and skip(y, x, tile, tile):
                continue
            out = fn(img[y : y + tile, x : x + tile])
            if out.ndim == 2:
                out = out[..., None]
            ys, xs = y * scale, x * scale
            acc[ys : ys + tile * scale, xs : xs + tile * scale] += out * win
            wsum[ys : ys + tile * scale, xs : xs + tile * scale] += win
    filled = wsum[..., 0] > 0
    res = np.zeros_like(acc)
    res[filled] = acc[filled] / wsum[filled]
    if default is not None:
        d = default
        if pad_h or pad_w:
            d = np.pad(d, ((0, pad_h * scale), (0, pad_w * scale)) + ((0, 0),) * (d.ndim - 2), mode="reflect")
        d = d.reshape(res.shape).astype(np.float32)
        res[~filled] = d[~filled]
    return res[: h * scale, : w * scale]


def overlay_mask(img: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.5) -> np.ndarray:
    out = img.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = out[m] * (1 - alpha) + np.array(color, np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)


def get_device(prefer: str | None = None):
    import torch

    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chunked(it: Iterable, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
