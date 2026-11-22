"""End-to-end restoration pipeline:

  deteriorated painting
     -> [M1] CLAHE + U-Net damage segmentation  -> damage mask
     -> [M2] LaMa inpainting (inside mask only)
     -> [M3] colour de-fading (Pix2Pix model or reference transfer)
     -> [M4] Real-ESRGAN super-resolution + spatial confidence map
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .color import auto_saturation_boost, defade_pix2pix, load_color_model, reference_color_transfer
from .inpainting import get_inpainter
from .preprocess import classical_crack_mask
from .segmentation import CHANNELS, load_seg_model, postprocess_mask, predict_probs
from .uncertainty import confidence_heatmap, confidence_map, inpaint_disagreement
from .utils import imwrite, overlay_mask, resize_max_side


# damage types as in the label tool: 1 crack, 2 loss, 3 stain, 255 unknown (RGB colours)
TYPE_COLORS = {1: (255, 0, 0), 2: (0, 80, 255), 3: (255, 220, 0), 255: (255, 0, 255)}
TYPE_NAMES = {1: "crack", 2: "loss", 3: "stain", 255: "unknown"}


def overlay_types(img: np.ndarray, types: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Damage overlay coloured by type, with a small legend in the top-left corner."""
    out = img.astype(np.float32).copy()
    for c, rgb in TYPE_COLORS.items():
        m = types == c
        out[m] = out[m] * (1 - alpha) + np.array(rgb, np.float32) * alpha
    out = out.clip(0, 255).astype(np.uint8)
    s = max(1.0, img.shape[1] / 1400)
    y = int(8 * s)
    for c in (1, 2, 3):
        cv2.rectangle(out, (int(8 * s), y), (int(26 * s), y + int(14 * s)), TYPE_COLORS[c], -1)
        cv2.rectangle(out, (int(8 * s), y), (int(26 * s), y + int(14 * s)), (0, 0, 0), 1)
        cv2.putText(out, TYPE_NAMES[c], (int(32 * s), y + int(12 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s,
                    (0, 0, 0), max(1, int(3 * s)), cv2.LINE_AA)
        cv2.putText(out, TYPE_NAMES[c], (int(32 * s), y + int(12 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s,
                    (255, 255, 255), max(1, int(s)), cv2.LINE_AA)
        y += int(20 * s)
    return out


class RestorationPipeline:
    def __init__(
        self,
        seg_ckpt: str | None = "checkpoints/seg_best.pt",
        color_ckpt: str | None = "checkpoints/color_best.pt",
        lama_weights: str = "weights/big-lama.pt",
        sr_weights: str = "weights/RealESRGAN_x4plus.pth",
        device: str | None = None,
    ):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seg = self.seg_ck = None
        if seg_ckpt and Path(seg_ckpt).exists():
            self.seg, self.seg_ck = load_seg_model(seg_ckpt, self.device)
        else:
            print(f"[pipeline] No segmentation checkpoint at {seg_ckpt} -> using classical crack detector (weak!).")
        self.color = load_color_model(color_ckpt, self.device) if color_ckpt and Path(color_ckpt).exists() else None
        self.inpainter = get_inpainter(lama_weights, self.device)
        self.sr_weights = sr_weights
        self._sr = None

    @property
    def sr(self):
        if self._sr is None:
            from .superres import SuperResolver

            self._sr = SuperResolver(self.sr_weights, self.device)
        return self._sr

    # ------------------------------------------------------------------ stages
    def detect(self, img, threshold=None, classes=("crack", "loss", "stain"), tta=True, dilate=3):
        self.last_probs = None
        if self.seg is None:
            m = classical_crack_mask(img)
            return m, None, None, 0.5
        thr = float(threshold if threshold is not None else self.seg_ck.get("threshold", 0.5))
        probs, tta_std = predict_probs(self.seg, img, self.device, use_clahe=self.seg_ck.get("use_clahe", True), tta=tta)
        self.last_probs = probs
        if set(classes) >= {"crack", "loss", "stain"}:
            p = probs[..., 0]
        else:
            p = probs[..., [CHANNELS.index(c) for c in classes]].max(-1)
        return postprocess_mask(p, thr, dilate=dilate), p, tta_std, thr

    def type_map(self, mask: np.ndarray) -> np.ndarray:
        """Per-pixel damage type inside `mask` (1 crack, 2 loss, 3 stain): the most likely type."""
        types = np.zeros(mask.shape[:2], np.uint8)
        m = mask > 0
        if self.last_probs is None:
            types[m] = 1
        else:
            cls = (self.last_probs[..., 1:].argmax(-1) + 1).astype(np.uint8)
            types[m] = cls[m]
        return types

    def recolor(self, img, color_ref=None, strength=1.0, auto_boost=False, adaptive=True):
        self.color_info = {}
        if color_ref is not None:
            return reference_color_transfer(img, color_ref, strength), "reference"
        if self.color is not None:
            out, self.color_info = defade_pix2pix(self.color, img, self.device, strength=strength,
                                                  adaptive=adaptive, return_info=True)
            return out, "pix2pix" + ("_adaptive" if adaptive else "")
        if auto_boost:
            return auto_saturation_boost(img), "auto_boost"
        return img.copy(), "none"

    # ------------------------------------------------------------------ run
    def run(
        self,
        img: np.ndarray,
        out_dir: str | Path,
        work_max_side: int | None = 2048,
        threshold: float | None = None,
        classes=("crack", "loss", "stain"),
        color_ref: np.ndarray | None = None,
        color_strength: float = 1.0,
        color_adaptive: bool = True,
        auto_boost: bool = False,
        sr_scale: float = 2.0,
        uncertainty: bool = True,
        tta: bool = True,
        dilate: int = 4,
        mask_editor=None,
    ) -> dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        timings = {}
        work, s = resize_max_side(img, work_max_side)
        imwrite(out / "00_input.png", work)

        t = time.time()
        mask, prob, tta_std, thr = self.detect(work, threshold, classes, tta, dilate)
        timings["segmentation"] = time.time() - t
        types = self.type_map(mask)
        edited = False
        if mask_editor is not None:  # semi-automatic: a person corrects the predicted mask
            imwrite(out / "01_mask_types_model.png", overlay_types(work, types))
            new = mask_editor(work, types, out)
            if new is not None:
                edited = not np.array_equal(new, types)
                types = new.astype(np.uint8)
                mask = np.where(types > 0, 255, 0).astype(np.uint8)
        imwrite(out / "01_mask.png", mask, rgb=False)
        imwrite(out / "01_mask_types.png", types, rgb=False)
        imwrite(out / "01_mask_overlay.png", overlay_types(work, types))
        if prob is not None:
            imwrite(out / "01_damage_prob.png", (prob * 255).astype(np.uint8), rgb=False)

        t = time.time()
        inpainted = self.inpainter.inpaint(work, mask)
        timings["inpainting"] = time.time() - t
        imwrite(out / "02_inpainted.png", inpainted)

        t = time.time()
        colored, color_method = self.recolor(inpainted, color_ref, color_strength, auto_boost, color_adaptive)
        timings["color"] = time.time() - t
        imwrite(out / "03_color.png", colored)

        t = time.time()
        istd = inpaint_disagreement(self.inpainter, work, mask, inpainted) if uncertainty else None
        conf = confidence_map(mask, prob, tta_std, thr, istd)
        timings["uncertainty"] = time.time() - t

        t = time.time()
        final = self.sr.upscale(colored, outscale=sr_scale) if sr_scale and sr_scale > 1 else colored
        timings["super_resolution"] = time.time() - t
        conf_final = cv2.resize(conf, (final.shape[1], final.shape[0]), interpolation=cv2.INTER_LINEAR)
        imwrite(out / "04_final.png", final)
        np.save(out / "confidence.npy", conf_final.astype(np.float16))
        imwrite(out / "confidence_map.png", confidence_heatmap(final, conf_final))
        imwrite(out / "confidence_gray.png", (conf_final * 255).astype(np.uint8), rgb=False)

        m = mask > 0
        report = {
            "input_size": list(img.shape[:2]),
            "work_size": list(work.shape[:2]),
            "final_size": list(final.shape[:2]),
            "segmenter": "unet" if self.seg is not None else "classical",
            "threshold": thr,
            "damaged_fraction": float(m.mean()),
            "damage_by_type_%": {TYPE_NAMES[c]: round(float((types == c).mean()) * 100, 2) for c in TYPE_NAMES},
            "mask_edited_by_user": edited,
            "color_method": color_method,
            "color_adjust": self.color_info,
            "mean_confidence_in_infill": float(conf[m].mean()) if m.any() else None,
            "speculative_fraction_of_infill(conf<0.5)": float((conf[m] < 0.5).mean()) if m.any() else None,
            "timings_s": {k: round(v, 2) for k, v in timings.items()},
            "total_s": round(time.time() - t0, 2),
        }
        (out / "report.json").write_text(json.dumps(report, indent=2))
        return report
