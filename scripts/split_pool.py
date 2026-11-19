"""Split data/pool into data/clean (training source) and data/damaged (real damage).

    python scripts/split_pool.py

Reads data/pool/damaged_ids.txt  (ids of genuinely damaged paintings -> data/damaged)
      data/pool/exclude_ids.txt  (ids kept out of BOTH, e.g. aged pieces with
                                  unclear damage, which must not be "clean" negatives)
Edit those two files and rerun any time; data/clean and data/damaged are rebuilt.
"""
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data/pool"


def read_ids(p: Path) -> set[int]:
    if not p.exists():
        return set()
    ids = set()
    for line in p.read_text().splitlines():
        line = line.split("#")[0].strip()
        ids |= {int(x) for x in line.replace(",", " ").split()}
    return ids


def main():
    damaged, exclude = read_ids(POOL / "damaged_ids.txt"), read_ids(POOL / "exclude_ids.txt")
    rows = list(csv.DictReader(open(POOL / "manifest.csv", encoding="utf-8")))
    if not rows:
        sys.exit("Run scripts/import_dataset.py first.")
    for d in ("clean", "damaged"):
        tgt = ROOT / "data" / d
        if tgt.exists():
            for f in tgt.glob("*.png"):
                f.unlink()
        tgt.mkdir(parents=True, exist_ok=True)
    n = {"clean": 0, "damaged": 0, "excluded": 0}
    for r in rows:
        i = int(r["id"])
        if i in damaged:
            shutil.copy2(POOL / r["file"], ROOT / "data/damaged" / r["file"]); n["damaged"] += 1
        elif i in exclude:
            n["excluded"] += 1
        else:
            shutil.copy2(POOL / r["file"], ROOT / "data/clean" / r["file"]); n["clean"] += 1
    print(n)


if __name__ == "__main__":
    main()
