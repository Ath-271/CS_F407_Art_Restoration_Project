"""Realistic synthetic degradation for training without manual pixel labels.

Why the old approach failed on real paintings: smooth Bezier strokes of uniform
width and flat colour are trivially separable from real damage, so the network
learned "detect perfect synthetic strokes" instead of "detect damage".

This module instead produces:
  * craquelure  - polygonal crack networks (Voronoi cells, warped), variable
                  width, partially present, dark OR light (lifted paint) cracks,
                  with lifted-edge highlights; plus branching random-walk cracks
  * losses      - jagged flakes exposing paper / cloth / gesso ground with
                  fibre texture and shadowed rims, clustered along cracks
  * stains      - water stains with tide-lines, foxing, mold speckle
  * real damage - cut-outs from real paintings (data/damage_bank), pasted in
  * scan noise  - uneven light, blur, sensor noise, JPEG, resampling, applied
                  AFTER the damage so edges look scanned, not rendered

Labels: (H, W, 3) uint8 binary channels = [crack, loss, stain].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .utils import fractal_noise, imread_any, list_images, smooth_noise

CLASSES = ("crack", "loss", "stain")

GROUND_COLORS = np.array(
    [
        [222, 208, 178],  # handmade paper
        [235, 228, 210],  # off-white cloth
        [200, 180, 140],  # aged paper
        [245, 242, 235],  # gesso / chalk ground
        [170, 140, 100],  # board / dark cloth
        [190, 165, 125],  # cow-dung/clay wash (Madhubani walls, Warli)
    ],
    np.float32,
)


@dataclass
class SynthConfig:
    p_clean: float = 0.25  # fraction of samples with NO damage (teaches linework != crack)
    p_crack: float = 0.75
    p_loss: float = 0.45
    p_stain: float = 0.35
    p_bank: float = 0.5  # used only if the bank has items
    p_plaster: float = 0.2  # fresco-type damage (plaster loss, salt bloom, streaks)
    p_scan_noise: float = 0.9
    damage_bank: list = field(default_factory=list)  # list of (class_idx, rgba ndarray)


# ============================================================================ helpers
def _warp(mask: np.ndarray, rng, amp: float, scale: float) -> np.ndarray:
    h, w = mask.shape[:2]
    dx = (smooth_noise(h, w, scale, rng) - 0.5) * 2 * amp
    dy = (smooth_noise(h, w, scale, rng) - 0.5) * 2 * amp
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(mask, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _ground_texture(h: int, w: int, rng) -> np.ndarray:
    base = GROUND_COLORS[rng.integers(len(GROUND_COLORS))] + rng.normal(0, 8, 3)
    tex = base[None, None, :] * (0.92 + 0.16 * fractal_noise(h, w, rng, base_scale=24)[..., None])
    # fibres: motion-blurred noise
    fib = rng.normal(0, 1, (h, w)).astype(np.float32)
    k = int(rng.integers(5, 15))
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2, k / 2), float(rng.uniform(0, 180)), 1.0)
    kern = cv2.warpAffine(kern, M, (k, k))
    fib = cv2.filter2D(fib, -1, kern / (kern.sum() + 1e-6))
    tex += fib[..., None] * rng.uniform(4, 12)
    tex += rng.normal(0, 3, (h, w, 1))
    return tex.clip(0, 255)


# ============================================================================ cracks
def craquelure(h: int, w: int, rng) -> np.ndarray:
    """Polygonal crack network, returns soft strength map in [0,1]."""
    cell = rng.uniform(18, 90)
    n = max(4, int(h * w / cell**2))
    seeds = np.ones((h, w), np.uint8)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    seeds[ys, xs] = 0
    _, labels = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
    labels = labels.astype(np.float32)
    labels = _warp(labels, rng, amp=rng.uniform(1.5, 5), scale=rng.uniform(8, 25))
    labels = np.round(labels).astype(np.int32)
    edge = np.zeros((h, w), np.uint8)
    edge[:, 1:] |= (labels[:, 1:] != labels[:, :-1]).astype(np.uint8)
    edge[1:, :] |= (labels[1:, :] != labels[:-1, :]).astype(np.uint8)
    # region-limited, with broken segments
    region = fractal_noise(h, w, rng, base_scale=rng.uniform(60, 200)) > rng.uniform(0.2, 0.6)
    gaps = smooth_noise(h, w, rng.uniform(6, 20), rng) > rng.uniform(0.15, 0.45)
    edge = edge * region * gaps
    # variable width
    thick = cv2.dilate(edge, np.ones((2, 2), np.uint8))
    thicker = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    sel = smooth_noise(h, w, 30, rng)
    out = np.where(sel < 0.4, edge, np.where(sel < 0.8, thick, thicker)).astype(np.float32)
    return out


def walk_cracks(h: int, w: int, rng) -> np.ndarray:
    """Branching random-walk cracks (long structural cracks / folds)."""
    ss = 2  # supersample for anti-aliasing
    canvas = np.zeros((h * ss, w * ss), np.uint8)
    stack = []
    for _ in range(int(rng.integers(1, 4))):
        stack.append(
            (rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0, 2 * np.pi), rng.uniform(60, 1.5 * max(h, w)), rng.uniform(1.0, 3.5))
        )
    while stack:
        x, y, ang, length, th = stack.pop()
        drift = rng.normal(0, 0.02)
        step = 2.0
        for _ in range(int(length / step)):
            ang += drift + rng.normal(0, 0.18)
            nx, ny = x + step * np.cos(ang), y + step * np.sin(ang)
            t = max(1, int(round(th * ss * rng.uniform(0.7, 1.3))))
            cv2.line(canvas, (int(x * ss), int(y * ss)), (int(nx * ss), int(ny * ss)), 255, t, cv2.LINE_AA)
            x, y = nx, ny
            th = max(0.6, th * rng.uniform(0.995, 1.0))
            if rng.random() < 0.012 and len(stack) < 30:
                stack.append((x, y, ang + rng.choice([-1, 1]) * rng.uniform(0.4, 1.3), length * rng.uniform(0.2, 0.5), th * 0.7))
            if not (-20 < x < w + 20 and -20 < y < h + 20):
                break
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def render_cracks(img: np.ndarray, s: np.ndarray, rng) -> np.ndarray:
    """Composite a crack strength map onto float RGB image."""
    s = cv2.GaussianBlur(s, (0, 0), rng.uniform(0.4, 1.0))
    s = np.clip(s * rng.uniform(1.0, 1.6), 0, 1)[..., None]
    if rng.random() < 0.7:  # dark crack (shadow / dirt in crack)
        dark = img * rng.uniform(0.12, 0.5) + rng.normal(0, 6, 3) + np.array([10, 5, 0])
        target = dark.clip(0, 255)
        if rng.random() < 0.6:  # lifted edges catch light
            rim = cv2.dilate(s[..., 0], np.ones((3, 3), np.uint8)) - s[..., 0]
            rim = np.clip(rim, 0, 1)[..., None] * rng.uniform(0.08, 0.25)
            img = img * (1 + rim)
    else:  # light crack: paint gone, ground visible
        target = _ground_texture(*img.shape[:2], rng)
    a = rng.uniform(0.65, 1.0)
    return img * (1 - a * s) + target * a * s


# ============================================================================ losses
def blob_mask(h: int, w: int, cx: float, cy: float, r: float, rng, jag: float = 0.6) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, np.pi)
    ex = rng.uniform(0.5, 1.5)
    dx, dy = xx - cx, yy - cy
    rx = dx * np.cos(ang) + dy * np.sin(ang)
    ry = (-dx * np.sin(ang) + dy * np.cos(ang)) * ex
    d = np.sqrt(rx**2 + ry**2) / max(r, 1)
    n = fractal_noise(h, w, rng, base_scale=max(4, r / 2), octaves=4)
    return (d + jag * (n - 0.5)) < 1.0


def render_losses(img: np.ndarray, rng, crack_hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    m = np.zeros((h, w), bool)
    # large flakes
    for _ in range(int(rng.integers(1, 5))):
        r = rng.uniform(4, 0.18 * min(h, w))
        m |= blob_mask(h, w, rng.uniform(0, w), rng.uniform(0, h), r, rng, jag=rng.uniform(0.4, 1.0))
    # small flakes along cracks (very common in real paintings)
    if crack_hint is not None and crack_hint.any() and rng.random() < 0.7:
        ys, xs = np.nonzero(crack_hint > 0.5)
        for i in rng.choice(len(ys), size=min(len(ys), int(rng.integers(3, 25))), replace=False):
            r = rng.uniform(2, 9)
            y0, y1 = max(0, int(ys[i] - 3 * r)), min(h, int(ys[i] + 3 * r))
            x0, x1 = max(0, int(xs[i] - 3 * r)), min(w, int(xs[i] + 3 * r))
            if y1 - y0 < 3 or x1 - x0 < 3:
                continue
            m[y0:y1, x0:x1] |= blob_mask(y1 - y0, x1 - x0, xs[i] - x0, ys[i] - y0, r, rng)
    # micro-flaking: dense clusters of tiny flecks where paint popped off (common on old Kalighat/Kangra)
    if rng.random() < 0.5:
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.08, 0.3) * min(h, w)
        specks = np.zeros((h, w), np.uint8)
        for _ in range(int(rng.integers(40, 300))):
            x, y = int(cx + rng.normal(0, r)), int(cy + rng.normal(0, r))
            ax_, ay_ = int(rng.integers(1, 4)), int(rng.integers(1, 3))
            cv2.ellipse(specks, (x, y), (ax_, ay_), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        m |= specks.astype(bool)
    if not m.any():
        return img, m
    ground = _ground_texture(h, w, rng)
    mf = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.6)[..., None]
    out = img * (1 - mf) + ground * mf
    if rng.random() < 0.65:  # shadowed / dirty rim around loss
        rim = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32) - m
        rim = cv2.GaussianBlur(rim, (0, 0), 0.8)[..., None]
        out = out * (1 - rim * rng.uniform(0.2, 0.55))
    return out, m


def render_abrasion(img: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Paint rubbed thin: ground shows through gradually (soft, patchy). Labelled
    as loss only where it's clearly visible (> ~45% of paint gone)."""
    h, w = img.shape[:2]
    r = rng.uniform(0.12, 0.4) * min(h, w)
    region = cv2.GaussianBlur(blob_mask(h, w, rng.uniform(0, w), rng.uniform(0, h), r, rng, 0.8).astype(np.float32), (0, 0), r * 0.15 + 1)
    grain = fractal_noise(h, w, rng, base_scale=rng.uniform(6, 20), octaves=3)
    a = np.clip(region * (0.4 + 0.9 * grain) * rng.uniform(0.6, 1.0), 0, 1)
    ground = _ground_texture(h, w, rng)
    out = img * (1 - a[..., None]) + ground * a[..., None]
    return out, a > 0.45


# ============================================================================ stains
def render_stains(img: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    label = np.zeros((h, w), bool)
    out = img
    kinds = rng.choice(["water", "foxing", "mold"], size=int(rng.integers(1, 3)))
    for kind in kinds:
        if kind == "water":
            r = rng.uniform(0.1, 0.45) * min(h, w)
            m = blob_mask(h, w, rng.uniform(0, w), rng.uniform(0, h), r, rng, jag=0.5).astype(np.float32)
            soft = cv2.GaussianBlur(m, (0, 0), r * 0.08 + 1)
            ring = np.clip(soft * (1 - soft) * 4, 0, 1) ** 2  # tide-line
            col = np.array([150, 110, 55], np.float32) + rng.normal(0, 15, 3)
            a = rng.uniform(0.15, 0.45)
            alpha = (soft * a + ring * a * 1.2).clip(0, 0.85)[..., None]
            out = out * (1 - alpha) + (out * col / 255.0) * alpha
            label |= alpha[..., 0] > 0.12
        else:
            n = int(rng.integers(15, 120))
            r = rng.uniform(0.08, 0.3) * min(h, w)
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            spots = np.zeros((h, w), np.float32)
            for _ in range(n):
                x, y = cx + rng.normal(0, r), cy + rng.normal(0, r)
                cv2.circle(spots, (int(x), int(y)), int(rng.integers(1, 5)), float(rng.uniform(0.5, 1)), -1, cv2.LINE_AA)
            spots = cv2.GaussianBlur(spots, (0, 0), rng.uniform(0.5, 1.5))
            if kind == "foxing":
                col = np.array([165, 95, 45], np.float32) + rng.normal(0, 15, 3)
            else:
                col = np.array([70, 80, 60], np.float32) + rng.normal(0, 15, 3)
            a = spots[..., None] * rng.uniform(0.5, 0.9)
            out = out * (1 - a) + col * a
            label |= spots > 0.25
    return out, label


# ============================================================================ wall / fresco damage
def _substrate(h, w, rng):
    """Brick, lime plaster or mud/stone exposed under a fresco."""
    kind = rng.choice(["brick", "lime", "mud"])
    if kind == "brick":
        base = np.array([150, 75, 50], np.float32) + rng.normal(0, 12, 3)
        tex = base * (0.8 + 0.4 * fractal_noise(h, w, rng, base_scale=10)[..., None])
        bh = int(rng.integers(12, 22))  # mortar lines
        mortar = np.zeros((h, w), np.float32)
        for y in range(int(rng.integers(0, bh)), h, bh):
            mortar[y : y + 2] = 1
            off = int(rng.integers(0, bh * 2))
            for x in range(off, w, bh * 3):
                mortar[y : y + bh, x : x + 2] = 1
        tex = tex * (1 - mortar[..., None]) + np.array([190, 180, 165]) * mortar[..., None]
    elif kind == "lime":
        tex = (np.array([215, 210, 198]) + rng.normal(0, 6, 3)) * (0.85 + 0.25 * fractal_noise(h, w, rng, 16)[..., None])
    else:
        tex = (np.array([160, 135, 105]) + rng.normal(0, 10, 3)) * (0.8 + 0.35 * fractal_noise(h, w, rng, 12)[..., None])
    return (tex + rng.normal(0, 4, (h, w, 1))).clip(0, 255)


def render_plaster(img: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fresco damage. Returns (img, loss_mask, stain_mask)."""
    h, w = img.shape[:2]
    loss = np.zeros((h, w), bool)
    stain = np.zeros((h, w), bool)
    out = img
    if rng.random() < 0.7:  # plaster loss with stepped edge (intonaco layer + deeper layer)
        r = rng.uniform(0.1, 0.35) * min(h, w)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        outer = blob_mask(h, w, cx, cy, r, rng, jag=rng.uniform(0.5, 1.1))
        inner = blob_mask(h, w, cx, cy, r * rng.uniform(0.4, 0.8), rng, jag=0.9) & outer
        layer1 = (np.array([225, 218, 205]) + rng.normal(0, 8, 3)) * (0.9 + 0.15 * fractal_noise(h, w, rng, 8)[..., None])
        sub = _substrate(h, w, rng)
        m1 = cv2.GaussianBlur(outer.astype(np.float32), (0, 0), 0.7)[..., None]
        m2 = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), 0.7)[..., None]
        out = out * (1 - m1) + layer1 * m1
        out = out * (1 - m2) + sub * m2
        edge = cv2.morphologyEx(outer.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
        out = out * (1 - 0.35 * cv2.GaussianBlur(edge, (0, 0), 0.8)[..., None])  # shadow at broken edge
        loss |= outer
    if rng.random() < 0.45:  # salt efflorescence: white haze
        r = rng.uniform(0.15, 0.45) * min(h, w)
        soft = cv2.GaussianBlur(blob_mask(h, w, rng.uniform(0, w), rng.uniform(0, h), r, rng, 0.9).astype(np.float32), (0, 0), r * 0.2 + 1)
        a = (soft * (0.4 + 0.6 * fractal_noise(h, w, rng, 6)) * rng.uniform(0.4, 0.8))[..., None]
        before = out
        out = out * (1 - a) + np.array([245, 243, 238]) * a
        stain |= np.abs(out - before).max(-1) > 32
    if rng.random() < 0.4:  # water streaks / soot running down
        streak = np.zeros((h, w), np.float32)
        for _ in range(int(rng.integers(3, 12))):
            x, y0 = int(rng.uniform(0, w)), int(rng.uniform(0, h * 0.5))
            cv2.line(streak, (x, y0), (x + int(rng.normal(0, 6)), int(y0 + rng.uniform(0.3, 1.0) * h)), 1.0, int(rng.integers(2, 10)))
        streak = cv2.GaussianBlur(streak, (0, 0), rng.uniform(1.5, 4)) * fractal_noise(h, w, rng, 20)
        col = np.array([60, 50, 40]) if rng.random() < 0.5 else np.array([130, 105, 70])
        a = (np.clip(streak, 0, 1) * rng.uniform(0.3, 0.7))[..., None]
        before = out
        out = out * (1 - a) + col * a
        stain |= np.abs(out - before).max(-1) > 32
    return out, loss, stain


# ============================================================================ damage bank
def load_damage_bank(folder: str | Path) -> list:
    """Files named <class>_*.png with alpha channel = damage mask (crack/loss/stain)."""
    bank = []
    for p in list_images(folder):
        cls = p.stem.split("_")[0].lower()
        if cls not in CLASSES:
            continue
        im = imread_any(p)
        if im.ndim != 3 or im.shape[2] != 4:
            continue
        rgba = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
        bank.append((CLASSES.index(cls), rgba))
    return bank


def paste_bank(img: np.ndarray, label: np.ndarray, bank: list, rng) -> np.ndarray:
    h, w = img.shape[:2]
    for _ in range(int(rng.integers(1, 4))):
        cls, rgba = bank[rng.integers(len(bank))]
        s = rng.uniform(0.6, 1.4)
        piece = cv2.resize(rgba, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        k = int(rng.integers(4))
        piece = np.rot90(piece, k)
        if rng.random() < 0.5:
            piece = piece[:, ::-1]
        ph, pw = piece.shape[:2]
        if ph >= h or pw >= w:
            piece = piece[: h - 1, : w - 1]
            ph, pw = piece.shape[:2]
        y, x = int(rng.integers(0, h - ph + 1)), int(rng.integers(0, w - pw + 1))
        a = piece[..., 3:4].astype(np.float32) / 255.0
        region = img[y : y + ph, x : x + pw]
        img[y : y + ph, x : x + pw] = region * (1 - a) + piece[..., :3].astype(np.float32) * a
        label[y : y + ph, x : x + pw, cls] |= (a[..., 0] > 0.5).astype(np.uint8)
    return img


# ============================================================================ scan degradations
def scan_degrade(img: np.ndarray, rng) -> np.ndarray:
    """Unlabelled degradations making tiles look like real photos/scans."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    if rng.random() < 0.6:  # uneven illumination
        out *= (1 + rng.uniform(0.05, 0.25) * (smooth_noise(h, w, max(h, w) / 2, rng) - 0.5) * 2)[..., None]
    if rng.random() < 0.5:  # brightness / contrast / gamma
        out = out * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
        out = 255 * (np.clip(out, 0, 255) / 255) ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.3:  # mild global yellowing
        out = out * np.array([1.0, rng.uniform(0.95, 1.0), rng.uniform(0.82, 0.97)])
    out = out.clip(0, 255)
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.1))
    if rng.random() < 0.3:  # resampling (photo at lower res)
        s = rng.uniform(0.5, 0.9)
        small = cv2.resize(out, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.6:
        out = out + rng.normal(0, rng.uniform(1, 6), out.shape)
    out = out.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 96))])
        out = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return out


# ============================================================================ main entry
def synthesize(clean: np.ndarray, rng: np.random.Generator, cfg: SynthConfig | None = None):
    """clean uint8 RGB tile -> (damaged uint8 RGB, label uint8 (H,W,3) in {0,1})."""
    cfg = cfg or SynthConfig()
    h, w = clean.shape[:2]
    img = clean.astype(np.float32)
    label = np.zeros((h, w, 3), np.uint8)

    if rng.random() >= cfg.p_clean:
        use_bank = bool(cfg.damage_bank) and rng.random() < cfg.p_bank
        do = {
            "crack": rng.random() < cfg.p_crack,
            "loss": rng.random() < cfg.p_loss,
            "stain": rng.random() < cfg.p_stain,
        }
        if not any(do.values()) and not use_bank:
            do[rng.choice(list(do))] = True

        crack_s = np.zeros((h, w), np.float32)
        if do["stain"]:  # stains first: they sit under later cracks
            before = img
            img, st = render_stains(img, rng)
            # label only VISIBLE staining (faint invisible spots would teach false positives)
            visible = np.abs(img - before).max(-1) > 14
            visible = cv2.morphologyEx(visible.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
            label[..., 2] |= st & visible
        if do["crack"]:
            if rng.random() < 0.7:
                crack_s = np.maximum(crack_s, craquelure(h, w, rng))
            if rng.random() < 0.5 or not crack_s.any():
                crack_s = np.maximum(crack_s, walk_cracks(h, w, rng))
            img = render_cracks(img, crack_s, rng)
            label[..., 0] |= crack_s > 0.35
        if do["loss"] and rng.random() < 0.35:
            img, am = render_abrasion(img, rng)
            label[..., 1] |= am
        if do["loss"]:
            img, lm = render_losses(img, rng, crack_s if crack_s.any() else None)
            label[..., 1] |= lm
            label[..., 0] &= ~lm  # a pixel that's a loss is not also a crack
        if rng.random() < cfg.p_plaster:
            img, pl, ps = render_plaster(img, rng)
            label[..., 1] |= pl
            label[..., 2] |= ps & ~pl
            label[..., 0] &= ~pl
        if use_bank:
            img = paste_bank(img, label, cfg.damage_bank, rng)

    img = img.clip(0, 255).astype(np.uint8)
    if rng.random() < cfg.p_scan_noise:
        img = scan_degrade(img, rng)
    return img, label


# ============================================================================ fading (Milestone 3)
def synthetic_fade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate pigment fading / oxidation / yellowing of varnish.

    - global + spatially varying chroma loss
    - organic reds/pinks/blues fade faster than earth pigments (hue-selective)
    - yellowing (b* shift), wash-out or darkening, contrast loss
    """
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    hue = np.arctan2(b, a)
    base = rng.uniform(0.3, 0.85)
    spatial = 1 + rng.uniform(0.0, 0.35) * (smooth_noise(h, w, max(h, w) / 1.5, rng) - 0.5) * 2
    red_sel = np.clip(np.cos(hue - rng.uniform(-0.3, 0.5)), 0, 1)  # around red/magenta
    blue_sel = np.clip(np.cos(hue - (-1.8)), 0, 1)
    extra = rng.uniform(0, 0.35) * red_sel + rng.uniform(0, 0.2) * blue_sel
    f = np.clip(base * spatial - extra, 0.1, 1.0)
    a, b = a * f, b * f
    b = b + rng.uniform(2, 14)  # yellowing
    a = a + rng.uniform(-1, 4)
    mean = L.mean()
    L = mean + (L - mean) * rng.uniform(0.65, 0.95)
    if rng.random() < 0.6:
        L = L + rng.uniform(0, 12)  # wash-out
    else:
        L = L * rng.uniform(0.85, 0.98)  # soot/darkening
    lab = np.stack([np.clip(L, 0, 100), a, b], -1)
    out = (cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).clip(0, 1) * 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = scan_degrade(out, rng)
    return out
