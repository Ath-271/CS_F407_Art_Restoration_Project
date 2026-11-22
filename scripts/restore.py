"""Integrated restoration CLI (all four milestones).

    python scripts/restore.py path/to/painting.jpg
    python scripts/restore.py data/damaged --out outputs/restored          # whole folder
    python scripts/restore.py painting.jpg --color-ref margin_crop.png      # reference de-fading
    python scripts/restore.py painting.jpg --sr-scale 4 --threshold 0.4
    python scripts/restore.py painting.jpg --review-mask                    # correct the mask by hand first

Outputs per painting (outputs/restored/<name>/):
    00_input.png  01_mask.png  01_mask_overlay.png (coloured by type: red crack, blue loss,
    yellow stain)  01_mask_types.png (1/2/3 per pixel)  01_damage_prob.png
    02_inpainted.png  03_color.png  04_final.png
    confidence.npy (float16, same size as final)  confidence_map.png  report.json
    with --review-mask also: 01_mask_types_model.png (before correction) and review/
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from restoration.pipeline import RestorationPipeline
from restoration.utils import imread_any, imread_rgb, imwrite, list_images


REVIEW_HELP = """
Review the damage mask (same keys as the label tool):
  left-drag paint  | right-drag erase | 1 crack (red) 2 loss (blue) 3 stain (yellow) 4 unknown
  wheel / + -  zoom | middle-drag or arrows  pan | [ ]  brush size | h  hide overlay
  r  back to the model's mask | c  clear | s  ACCEPT and restore | q  quit = keep the model's mask
"""


def make_mask_editor():
    """Mask editor for --review-mask, built on scripts/label_tool.py. The model's and the
    corrected mask are both kept in <out>/review/ for the record."""
    from label_tool import Labeler

    def edit(work, types, out_dir):
        root = Path(out_dir) / "review"
        for sub in ("images", "premasks", "masks"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        name = Path(out_dir).name
        imwrite(root / "images" / f"{name}.png", work)
        imwrite(root / "premasks" / f"{name}.png", types, rgb=False)
        saved = root / "masks" / f"{name}.png"
        if saved.exists():
            saved.unlink()
        h, w = work.shape[:2]
        print(REVIEW_HELP)
        lab = Labeler([root / "images" / f"{name}.png"], root, zoom=max(0.5, min(1100 / w, 800 / h)))
        lab.loop()
        if not saved.exists():
            print("No correction saved - using the model's mask.")
            return None
        print("Using your corrected mask.")
        return imread_any(saved, cv2.IMREAD_GRAYSCALE)

    return edit


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
    ap.add_argument("--review-mask", action="store_true",
                    help="semi-automatic: open the predicted mask in the brush editor and correct it before inpainting")
    a = ap.parse_args()

    src = Path(a.input)
    paths = [src] if src.is_file() else list_images(src)
    if not paths:
        sys.exit(f"No images found at {src}")
    pipe = RestorationPipeline(a.seg_ckpt, None if a.no_color else a.color_ckpt, a.lama, a.sr, a.device)
    ref = imread_rgb(a.color_ref) if a.color_ref else None
    editor = make_mask_editor() if a.review_mask else None
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
            mask_editor=editor,
        )
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
