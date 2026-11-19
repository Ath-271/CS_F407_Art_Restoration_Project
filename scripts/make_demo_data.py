"""Generate procedural folk-art-STYLE images for smoke-testing the code only.

These are NOT a substitute for real paintings - they only check that every
script runs end-to-end. Output goes to data/demo/ (never mixed with your data).

    python scripts/make_demo_data.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from restoration.synth_damage import SynthConfig, synthesize, synthetic_fade
from restoration.utils import fractal_noise, imwrite


def warli(rng, h=1024, w=1280):
    base = np.array([120, 60, 35]) + rng.normal(0, 10, 3)
    img = (base * (0.9 + 0.2 * fractal_noise(h, w, rng, 80)[..., None])).clip(0, 255).astype(np.uint8)
    white = (240, 235, 225)
    for _ in range(int(rng.integers(25, 45))):  # stick figures: two triangles + head + limbs
        x, y, s = int(rng.integers(40, w - 40)), int(rng.integers(60, h - 60)), int(rng.integers(12, 26))
        cv2.fillPoly(img, [np.array([[x - s, y - 2 * s], [x + s, y - 2 * s], [x, y - s // 2]])], white, cv2.LINE_AA)
        cv2.fillPoly(img, [np.array([[x, y - s // 2], [x - s, y + s], [x + s, y + s]])], white, cv2.LINE_AA)
        cv2.circle(img, (x, y - 2 * s - s // 2), s // 2, white, -1, cv2.LINE_AA)
        for sgn in (-1, 1):
            cv2.line(img, (x, y - int(1.6 * s)), (x + sgn * int(1.4 * s), y - s + int(rng.integers(-s, s))), white, 2, cv2.LINE_AA)
            cv2.line(img, (x + sgn * s // 2, y + s), (x + sgn * s, y + 2 * s), white, 2, cv2.LINE_AA)
    for r in range(60, 300, 40):  # concentric circle (tarpa dance motif)
        cv2.circle(img, (w // 2, h // 2), r, white, 2, cv2.LINE_AA)
    return img


def madhubani(rng, h=1024, w=1280):
    paper = np.array([230, 215, 180]) + rng.normal(0, 6, 3)
    img = (paper * (0.95 + 0.1 * fractal_noise(h, w, rng, 60)[..., None])).clip(0, 255).astype(np.uint8)
    pal = [(200, 30, 40), (240, 170, 20), (20, 120, 60), (30, 60, 160), (220, 90, 140)]
    # border with repeating pattern
    b = 60
    cv2.rectangle(img, (b, b), (w - b, h - b), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.rectangle(img, (b // 3, b // 3), (w - b // 3, h - b // 3), (0, 0, 0), 3, cv2.LINE_AA)
    for x in range(b // 3, w - b // 3, 24):
        cv2.line(img, (x, b // 3), (x + 12, b), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.line(img, (x, h - b // 3), (x + 12, h - b), (0, 0, 0), 2, cv2.LINE_AA)
    for y in range(b // 3, h - b // 3, 24):
        cv2.line(img, (b // 3, y), (b, y + 12), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.line(img, (w - b // 3, y), (w - b, y + 12), (0, 0, 0), 2, cv2.LINE_AA)
    for _ in range(int(rng.integers(6, 11))):  # fish / leaf motifs with hatching
        cx, cy = int(rng.integers(b + 120, w - b - 120)), int(rng.integers(b + 80, h - b - 80))
        ax, ay, ang = int(rng.integers(50, 120)), int(rng.integers(25, 60)), float(rng.uniform(0, 180))
        col = pal[rng.integers(len(pal))]
        cv2.ellipse(img, (cx, cy), (ax, ay), ang, 0, 360, col, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (ax, ay), ang, 0, 360, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (ax - 12, ay - 10), ang, 0, 360, (0, 0, 0), 1, cv2.LINE_AA)
        for k in range(-ax + 15, ax - 15, 9):  # hatching
            t = np.deg2rad(ang)
            p1 = (int(cx + k * np.cos(t) - (ay - 14) * np.sin(t) * 0.7), int(cy + k * np.sin(t) + (ay - 14) * np.cos(t) * 0.7))
            p2 = (int(cx + k * np.cos(t) + (ay - 14) * np.sin(t) * 0.7), int(cy + k * np.sin(t) - (ay - 14) * np.cos(t) * 0.7))
            cv2.line(img, p1, p2, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(img, (int(cx + 0.6 * ax * np.cos(np.deg2rad(ang))), int(cy + 0.6 * ax * np.sin(np.deg2rad(ang)))), 7, (0, 0, 0), 2)
    return img


def main():
    rng = np.random.default_rng(7)
    out = ROOT / "data/demo"
    for i in range(6):
        img = warli(rng) if i % 2 else madhubani(rng)
        imwrite(out / "clean" / f"demo_{i}.png", img)
    # "damaged" test paintings: different seed, heavier damage + fading
    cfg = SynthConfig(p_clean=0.0, p_crack=1.0, p_loss=1.0, p_stain=0.6)
    for i in range(2):
        img = madhubani(rng) if i == 0 else warli(rng)
        imwrite(out / "damaged_reference_clean" / f"test_{i}.png", img)
        dmg, lab = synthesize(img, np.random.default_rng(100 + i), cfg)
        dmg = synthetic_fade(dmg, np.random.default_rng(200 + i))
        imwrite(out / "damaged" / f"test_{i}.png", dmg)
        imwrite(out / "damaged_gt" / f"test_{i}.png", (lab.max(-1) * 255).astype(np.uint8), rgb=False)
    print(f"Demo data written to {out}")


if __name__ == "__main__":
    main()
