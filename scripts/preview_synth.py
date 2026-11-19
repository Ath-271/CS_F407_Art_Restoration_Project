"""Visual check of synthetic damage: clean | damaged | label overlay.

    python scripts/preview_synth.py --n 8

ALWAYS look at this next to your real damaged paintings. If the synthetic
cracks look obviously different from real ones (width, colour, density,
pattern), the model will not transfer -- tweak synth_damage.py or add real
cut-outs to data/damage_bank.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from restoration.dataset import random_crop
from restoration.synth_damage import SynthConfig, load_damage_bank, synthesize, synthetic_fade
from restoration.utils import imread_rgb, imwrite, list_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(ROOT / "data/clean"))
    ap.add_argument("--bank", default=str(ROOT / "data/damage_bank"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs/synth_preview.png"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    paths = list_images(a.clean)
    if not paths:
        sys.exit(f"No images in {a.clean}")
    cfg = SynthConfig(damage_bank=load_damage_bank(a.bank), p_clean=0.0)
    rows = []
    for _ in range(a.n):
        clean = random_crop(imread_rgb(paths[rng.integers(len(paths))]), a.tile, rng)
        dmg, lab = synthesize(clean, rng, cfg)
        ov = dmg.copy()
        for c, col in enumerate([(255, 0, 0), (0, 90, 255), (255, 220, 0)]):
            m = lab[..., c] > 0
            ov[m] = (ov[m] * 0.4 + np.array(col) * 0.6).astype(np.uint8)
        rows.append(np.concatenate([clean, dmg, ov, synthetic_fade(clean, rng)], 1))
    imwrite(a.out, np.concatenate(rows, 0))
    print(f"Saved {a.out}  (columns: clean | damaged | labels red=crack blue=loss yellow=stain | faded)")


if __name__ == "__main__":
    main()
