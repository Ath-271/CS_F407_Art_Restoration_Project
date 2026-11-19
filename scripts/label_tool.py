"""Minimal brush tool for labelling damage on real tiles.

    python scripts/label_tool.py            # tiles without a saved mask
    python scripts/label_tool.py --all      # revisit everything

Mouse:  left-drag = paint current class    right-drag = erase    wheel = ZOOM at cursor
        middle-button (wheel-click) drag = pan
Keys:   1 crack (red)   2 loss (blue)   3 stain/mold (yellow)   4 damage, type unknown (magenta)
        + / -  zoom in/out at cursor     arrow keys  pan      0  reset zoom
        [ / ]  brush size      h  hide/show overlay      r  reset to suggestion
        c  clear all           s  SAVE + next            n  next (skip)   p  previous
        q / Esc  quit

Tips: you don't need perfect masks. Cover every crack/flake roughly, slightly
generously. Leave painted linework UNlabelled - that is exactly what teaches the
model the difference. Tiles with no damage: press c then s (an empty mask is a
valuable negative example).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from restoration.utils import imread_any, imread_rgb, imwrite, list_images

COLORS = {1: (0, 0, 255), 2: (255, 80, 0), 3: (0, 220, 255), 255: (255, 0, 255)}  # BGR
NAMES = {1: "crack", 2: "loss", 3: "stain", 255: "unknown"}
KEY2CLS = {ord("1"): 1, ord("2"): 2, ord("3"): 3, ord("4"): 255}


class Labeler:
    BAR = 26  # status bar height (px) above the image

    def __init__(self, items, root: Path, zoom: float, view=(1100, 800)):
        self.items, self.root, self.zoom, self.start_zoom = items, root, zoom, zoom
        self.vw, self.vh = view
        self.idx, self.cls, self.brush, self.show = 0, 1, 4, True
        self.drawing = 0
        self.mouse = (0, 0)  # last mouse position in IMAGE coords
        self.pan_from = None
        cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("label", self.on_mouse)
        self.load()

    def load(self):
        ip = self.items[self.idx]
        self.img = cv2.cvtColor(imread_rgb(ip), cv2.COLOR_RGB2BGR)
        mp, pp = self.root / "masks" / f"{ip.stem}.png", self.root / "premasks" / f"{ip.stem}.png"
        src = mp if mp.exists() else pp if pp.exists() else None
        self.suggest = imread_any(src, cv2.IMREAD_GRAYSCALE) if src else np.zeros(self.img.shape[:2], np.uint8)
        self.mask = self.suggest.copy()
        self.status = "saved" if mp.exists() else "new"
        h, w = self.img.shape[:2]
        self.cx, self.cy = w / 2, h / 2  # view centre (image coords)

    # ---------------------------------------------------------------- view geometry
    def viewport(self):
        """Visible image region (x0, y0, x1, y1) in image pixels."""
        h, w = self.img.shape[:2]
        vw_img, vh_img = min(w, self.vw / self.zoom), min(h, self.vh / self.zoom)
        x0 = float(np.clip(self.cx - vw_img / 2, 0, w - vw_img))
        y0 = float(np.clip(self.cy - vh_img / 2, 0, h - vh_img))
        return x0, y0, x0 + vw_img, y0 + vh_img

    def to_img(self, x, y):
        x0, y0, _, _ = self.viewport()
        return x0 + x / self.zoom, y0 + (y - self.BAR) / self.zoom

    def set_zoom(self, z, anchor=None):
        """Zoom keeping `anchor` (image coords) under the same screen point."""
        z = float(np.clip(z, 0.5, 8.0))
        ax, ay = anchor if anchor else (self.cx, self.cy)
        x0, y0, _, _ = self.viewport()
        sx, sy = (ax - x0) * self.zoom, (ay - y0) * self.zoom
        self.zoom = z
        self.cx = ax - sx / z + (self.vw / z) / 2
        self.cy = ay - sy / z + (self.vh / z) / 2
        self.pan(0, 0)

    def pan(self, dx_img, dy_img):
        h, w = self.img.shape[:2]
        self.cx = float(np.clip(self.cx + dx_img, 0, w))
        self.cy = float(np.clip(self.cy + dy_img, 0, h))

    def reset_view(self):
        self.zoom = self.start_zoom
        h, w = self.img.shape[:2]
        self.cx, self.cy = w / 2, h / 2

    # ---------------------------------------------------------------- input
    def on_mouse(self, ev, x, y, flags, _):
        if ev in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
            # Plain wheel = zoom at the cursor. Wheel events on Windows can carry
            # screen (not window) coordinates, so anchor on the last MOVE position.
            delta = (int(flags) >> 16) & 0xFFFF  # wheel delta lives in the high 16 bits (signed)
            if delta >= 0x8000:
                delta -= 0x10000
            if delta != 0:
                self.set_zoom(self.zoom * (1.25 if delta > 0 else 0.8), anchor=self.mouse)
            return
        ix, iy = self.to_img(x, y)
        self.mouse = (ix, iy)
        if ev == cv2.EVENT_MBUTTONDOWN:
            self.pan_from = (x, y)
            return
        if ev == cv2.EVENT_MBUTTONUP:
            self.pan_from = None
            return
        if ev == cv2.EVENT_MOUSEMOVE and self.pan_from is not None:
            px, py = self.pan_from
            self.pan(-(x - px) / self.zoom, -(y - py) / self.zoom)
            self.pan_from = (x, y)
            return
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.drawing = 1
        elif ev == cv2.EVENT_RBUTTONDOWN:
            self.drawing = 2
        elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drawing = 0
        if self.drawing and y >= self.BAR and ev in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            cv2.circle(self.mask, (int(ix), int(iy)), self.brush, self.cls if self.drawing == 1 else 0, -1)

    # ---------------------------------------------------------------- drawing
    def render(self):
        x0, y0, x1, y1 = self.viewport()
        h, w = self.img.shape[:2]
        ix0, iy0 = int(x0), int(y0)
        ix1, iy1 = min(w, int(np.ceil(x1)) + 1), min(h, int(np.ceil(y1)) + 1)
        vis = self.img[iy0:iy1, ix0:ix1].copy()
        if self.show:
            m = self.mask[iy0:iy1, ix0:ix1]
            col = np.zeros_like(vis)
            for c, bgr in COLORS.items():
                col[m == c] = bgr
            sel = m > 0
            vis[sel] = (vis[sel] * 0.45 + col[sel] * 0.55).astype(np.uint8)
        out_w, out_h = int(round((x1 - x0) * self.zoom)), int(round((y1 - y0) * self.zoom))
        M = np.float32([[self.zoom, 0, -(x0 - ix0) * self.zoom], [0, self.zoom, -(y0 - iy0) * self.zoom]])
        vis = cv2.warpAffine(vis, M, (out_w, out_h), flags=cv2.INTER_NEAREST)
        mx, my = (self.mouse[0] - x0) * self.zoom, (self.mouse[1] - y0) * self.zoom
        cv2.circle(vis, (int(mx), int(my)), max(1, int(self.brush * self.zoom)), COLORS[self.cls], 1, cv2.LINE_AA)
        bar = np.zeros((self.BAR, max(out_w, 780), 3), np.uint8)
        txt = (f"[{self.idx + 1}/{len(self.items)}] {self.items[self.idx].name} ({self.status}) | "
               f"{NAMES[self.cls]} | brush {self.brush} | zoom {self.zoom:.1f}x")
        cv2.putText(bar, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[self.cls], 1, cv2.LINE_AA)
        if bar.shape[1] > out_w:
            vis = np.pad(vis, ((0, 0), (0, bar.shape[1] - out_w), (0, 0)))
        return np.concatenate([bar, vis], 0)

    def save(self):
        imwrite(self.root / "masks" / f"{self.items[self.idx].stem}.png", self.mask, rgb=False)
        print(f"saved {self.items[self.idx].stem}  (damage {(self.mask > 0).mean() * 100:.1f}%)")

    def loop(self):
        while True:
            cv2.imshow("label", self.render())
            kx = cv2.waitKeyEx(20)
            k = kx & 0xFF if kx != -1 else 255
            step = 60 / self.zoom
            arrows = {2424832: (-step, 0), 65361: (-step, 0), 2555904: (step, 0), 65363: (step, 0),
                      2490368: (0, -step), 65362: (0, -step), 2621440: (0, step), 65364: (0, step)}
            if kx in arrows:
                self.pan(*arrows[kx])
                continue
            if k in (ord("+"), ord("=")):
                self.set_zoom(self.zoom * 1.25, anchor=self.mouse)
                continue
            if k == ord("-"):
                self.set_zoom(self.zoom * 0.8, anchor=self.mouse)
                continue
            if k == ord("0"):
                self.reset_view()
                continue
            if k in (ord("q"), 27):
                break
            if k in KEY2CLS:
                self.cls = KEY2CLS[k]
            elif k == ord("["):
                self.brush = max(1, self.brush - 1)
            elif k == ord("]"):
                self.brush = min(60, self.brush + 1)
            elif k == ord("h"):
                self.show = not self.show
            elif k == ord("r"):
                self.mask = self.suggest.copy()
            elif k == ord("c"):
                self.mask[:] = 0
            elif k in (ord("s"), ord("n"), ord("p")):
                if k == ord("s"):
                    self.save()
                step = -1 if k == ord("p") else 1
                if not 0 <= self.idx + step < len(self.items):
                    print("End of list.")
                    if k == ord("s"):
                        break
                    continue
                self.idx += step
                self.load()
            if cv2.getWindowProperty("label", cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data/real_labeled"))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--zoom", type=float, default=1.5, help="starting zoom")
    ap.add_argument("--view", default="1100x800", help="max image area on screen, WxH px")
    a = ap.parse_args()
    root = Path(a.root)
    items = sorted(list_images(root / "images"), key=lambda p: (not p.name.startswith("calib_"), p.name))  # calibration first
    if not a.all:
        items = [p for p in items if not (root / "masks" / f"{p.stem}.png").exists()]
    if not items:
        sys.exit("Nothing to label. Run scripts/make_label_tiles.py first (or use --all).")
    print(__doc__)
    Labeler(items, root, a.zoom, tuple(int(v) for v in a.view.lower().split("x"))).loop()


if __name__ == "__main__":
    main()
