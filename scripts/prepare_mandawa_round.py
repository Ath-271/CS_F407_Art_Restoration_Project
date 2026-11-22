"""Build the Mandawa training / development folders after the mandawaA labels are merged.

    python scripts/merge_labels.py outputs/label_returns_mandawaA --root data/mandawa_A_label \
        --agreement-out outputs/mandawaA_annotator_agreement.csv
    python scripts/prepare_mandawa_round.py

Uses data/mandawa_A_label/roles.csv (made once, by panel group, before training):
  role == train -> data/mandawa_train/{images,masks}   (new tiles + Test A answer-key tile)
  role == dev   -> data/mandawa_dev/{images,masks}     (model comparison only, never trained on)
Only Test A material is used; Test B is never read.
"""
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data/mandawa_A_label"
KEY = ROOT / "data/test_holdout/label"


def main():
    roles = list(csv.DictReader(open(LAB / "roles.csv", encoding="utf-8")))
    split = {r["tile"]: r["split"] for r in csv.DictReader(open(ROOT / "data/test_holdout/split.csv", encoding="utf-8"))}
    count, missing = {"train": 0, "dev": 0}, []
    for role in count:
        d = ROOT / f"data/mandawa_{role}"
        if d.exists():
            shutil.rmtree(d)
        (d / "images").mkdir(parents=True)
        (d / "masks").mkdir()
    for r in roles:
        src = KEY if r["kind"] == "answer_key" else LAB
        if r["kind"] == "answer_key" and split.get(r["tile"]) != "A":
            sys.exit(f"{r['tile']} is not a Test A tile - refusing.")
        im, mk = src / "images" / f"{r['tile']}.png", src / "masks" / f"{r['tile']}.png"
        if not (im.exists() and mk.exists()):
            missing.append(r["tile"])
            continue
        d = ROOT / f"data/mandawa_{r['role']}"
        shutil.copy2(im, d / "images" / im.name)
        shutil.copy2(mk, d / "masks" / mk.name)
        count[r["role"]] += 1
    print(f"train tiles: {count['train']} -> data/mandawa_train | dev tiles: {count['dev']} -> data/mandawa_dev")
    if missing:
        print(f"WARNING: {len(missing)} tiles have no mask yet (not labelled?): {missing}")


if __name__ == "__main__":
    main()
