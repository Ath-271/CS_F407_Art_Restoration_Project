"""Split unlabelled tiles into per-person labelling kits (zip files).

    python scripts/make_label_kits.py --people "Atharva,Riya,Sam,Neha"
    python scripts/make_label_kits.py --people "..." --calib 4      # shared calibration tiles

* Splits by SOURCE PAINTING (one painting's tiles go to one person) and balances
  the number of tiles per person.
* `--calib N` tiles (from different paintings) go to EVERYONE, prefixed
  `calib_`, to measure inter-annotator agreement; merged later by majority vote.
* Re-runnable: tiles already assigned (data/real_labeled/assignment.csv) or
  already labelled are skipped, so new tiles (e.g. frescoes) become a new batch.
* Each kit is self-contained: label tool + minimal code + guide + .bat
  launchers; teammates only need Python (opencv-python/numpy auto-install).

Merge returned zips with scripts/merge_labels.py.
"""
import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from restoration.dataset import painting_key
from restoration.utils import list_images

RUN_BAT = r"""@echo off
cd /d "%~dp0"
rem Needs OpenCV WITH window support. The "headless" build (common) cannot open windows.
python -c "import cv2, numpy; cv2.namedWindow('t'); cv2.destroyAllWindows()" >nul 2>&1
if errorlevel 1 (
  echo Setting up OpenCV with window support ...
  python -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless >nul 2>&1
  python -m pip install --user --force-reinstall opencv-python numpy
)
python scripts\label_tool.py %*
if errorlevel 1 pause
"""
RETURN_BAT = r"""@echo off
cd /d "%~dp0"
set NAME={name}
powershell -NoProfile -Command "Compress-Archive -Force -Path 'data\real_labeled\masks\*' -DestinationPath '..\%NAME%_labels.zip'"
echo Created ..\%NAME%_labels.zip  - send this file back.
pause
"""
RUN_SH = """#!/bin/sh
cd "$(dirname "$0")"
python3 -c "import cv2, numpy; cv2.namedWindow('t'); cv2.destroyAllWindows()" 2>/dev/null || { python3 -m pip uninstall -y opencv-python-headless; python3 -m pip install --user --force-reinstall opencv-python numpy; }
python3 scripts/label_tool.py "$@"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", required=True, help="comma-separated names")
    ap.add_argument("--calib", type=int, default=4, help="shared calibration tiles (0 = none)")
    ap.add_argument("--root", default=str(ROOT / "data/real_labeled"))
    ap.add_argument("--out", default=str(ROOT / "outputs/label_kits"))
    ap.add_argument("--kit-name", default=None, help="kit suffix instead of batchN, e.g. 'test'")
    a = ap.parse_args()
    people = [p.strip() for p in a.people.split(",") if p.strip()]
    root, out = Path(a.root), Path(a.out)
    assign_csv = root / "assignment.csv"
    assigned = {}
    if assign_csv.exists():
        assigned = {r["tile"]: r for r in csv.DictReader(open(assign_csv, encoding="utf-8"))}
    batch = 1 + max([int(r["batch"]) for r in assigned.values()], default=0)

    tiles = [p for p in list_images(root / "images")
             if p.stem not in assigned and not (root / "masks" / f"{p.stem}.png").exists()]
    if not tiles:
        sys.exit("No unassigned, unlabelled tiles.")
    groups = defaultdict(list)
    for t in tiles:
        groups[painting_key(t)].append(t)

    # calibration tiles: one tile from each of N different paintings (spread out)
    calib = []
    if a.calib and len(people) > 1:
        keys = sorted(groups)
        for k in keys[:: max(1, len(keys) // a.calib)][: a.calib]:
            calib.append(groups[k].pop(0))
            if not groups[k]:
                del groups[k]

    # balance by painting: biggest groups first, to whoever has fewest tiles
    share = {p: [] for p in people}
    for k in sorted(groups, key=lambda k: -len(groups[k])):
        who = min(people, key=lambda p: len(share[p]))
        share[who] += groups[k]

    rows = list(assigned.values())
    for person in people:
        tag = a.kit_name or f"batch{batch}"
        kit = out / f"{person}_{tag}"
        if kit.exists():
            shutil.rmtree(kit)
        d = kit / "data/real_labeled"
        for sub in ("images", "premasks", "masks"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        for t, is_cal in [(t, True) for t in calib] + [(t, False) for t in share[person]]:
            name = ("calib_" if is_cal else "") + t.name
            shutil.copy2(t, d / "images" / name)
            pm = root / "premasks" / f"{t.stem}.png"
            if pm.exists() and not is_cal:  # calibration: no suggestion, label from scratch
                shutil.copy2(pm, d / "premasks" / name)
            rows.append({"tile": t.stem, "person": person, "batch": batch, "calib": int(is_cal)})
        (kit / "scripts").mkdir(exist_ok=True)
        (kit / "restoration").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "scripts/label_tool.py", kit / "scripts/label_tool.py")
        shutil.copy2(ROOT / "restoration/utils.py", kit / "restoration/utils.py")
        (kit / "restoration/__init__.py").write_text("")
        shutil.copy2(ROOT / "docs/LABELING_GUIDE.md", kit / "LABELING_GUIDE.md")
        (kit / "run_label.bat").write_text(RUN_BAT)
        (kit / "return_labels.bat").write_text(RETURN_BAT.replace("{name}", f"{person}_{tag}"))
        (kit / "run_label.sh").write_text(RUN_SH)
        zp = shutil.make_archive(str(kit), "zip", kit)
        print(f"{person:>12}: {len(share[person])} tiles + {len(calib)} calibration -> {zp}")

    # de-duplicate rows (calib tiles appear once per person)
    with open(assign_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["tile", "person", "batch", "calib"])
        w.writeheader()
        w.writerows(rows)
    print(f"Batch {batch}: assignment saved to {assign_csv}")


if __name__ == "__main__":
    main()
