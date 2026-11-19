"""Integrated restoration CLI (all four milestones).

    python scripts/restore.py path/to/painting.jpg
    python scripts/restore.py data/damaged --out outputs/restored          # whole folder
    python scripts/restore.py painting.jpg --color-ref margin_crop.png      # reference de-fading
    python scripts/restore.py painting.jpg --sr-scale 4 --threshold 0.4

Outputs per painting (outputs/restored/<name>/):
    00_input.png  01_mask.png  01_mask_overlay.png  01_damage_prob.png
    02_inpainted.png  03_color.png  04_final.png
    confidence.npy (float16, same size as final)  confidence_map.png  report.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from restoration.pipeline import RestorationPipeline
from restoration.utils import imread_rgb, list_images


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="image file or folder")
    ap.add_argument("--out", default=str(ROOT / "outputs/restored"))
    ap.add_argument("--seg-ckpt", default=str(ROOT / "checkpoints/seg_best.pt"))
    ap.add_argument("--color-ckpt", default=str(ROOT / "checkpoints/color_best.pt"))
    ap.add_argument("--lama", default=str(ROOT / "weights/big-lama.pt"))
    ap.add_argument("--sr", default=str(ROOT / "weights/RealESRGAN_x4plus.pth"))
    ap.add_argument("--max-side", type=int, default=2048, help="working resolution for detection + inpainting")
    ap.add_argument("--threshold", type=float, default=None, help="override tuned threshold")
    ap.add_argument("--classes", default="crack,loss,stain", help="which damage types to inpaint")
    ap.add_argument("--color-ref", default=None, help="reference image (protected margin / well-preserved painting)")
    ap.add_argument("--color-strength", type=float, default=1.0)
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--no-adaptive-color", action="store_true",
                    help="apply the de-fading model unrestricted (default limits change by how faded the input is)")
    ap.add_argument("--auto-boost", action="store_true", help="simple saturation boost if no colour model/ref")
    ap.add_argument("--sr-scale", type=float, default=2.0, help="0 or 1 disables super-resolution")
    ap.add_argument("--no-uncertainty", action="store_true", help="skip extra LaMa runs (faster)")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--dilate", type=int, default=4, help="grow mask by N px before inpainting (removes crack halos)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    src = Path(a.input)
    paths = [src] if src.is_file() else list_images(src)
    if not paths:
        sys.exit(f"No images found at {src}")
    pipe = RestorationPipeline(a.seg_ckpt, None if a.no_color else a.color_ckpt, a.lama, a.sr, a.device)
    ref = imread_rgb(a.color_ref) if a.color_ref else None
    for p in paths:
        print(f"Restoring {p.name} ...")
        rep = pipe.run(
            imread_rgb(p),
            Path(a.out) / p.stem,
            work_max_side=a.max_side,
            threshold=a.threshold,
            classes=tuple(c.strip() for c in a.classes.split(",")),
            color_ref=ref,
            color_strength=a.color_strength,
            color_adaptive=not a.no_adaptive_color,
            auto_boost=a.auto_boost,
            sr_scale=a.sr_scale,
            uncertainty=not a.no_uncertainty,
            tta=not a.no_tta,
            dilate=a.dilate,
        )
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
