"""Collect fresco / mural images from Wikimedia Commons with license metadata.

    python scripts/collect_commons.py --discover          # counts per seed category
    python scripts/collect_commons.py                     # download

* Walks seed categories (+ subcategories to --depth).
* EXCLUDES anything whose title or categories match --exclude (default: Mandawa),
  because the held-out test set is from Mandawa: other photos of the same walls
  would leak the test set into training.
* Keeps only free licenses (CC0 / PD / CC BY / CC BY-SA) and records author +
  license in attribution.csv for the report.
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://commons.wikimedia.org/w/api.php"
UA = "FolkArtRestorationStudentProject/0.1 (educational, non-commercial research)"

SEEDS = [
    # Shekhawati towns other than Mandawa
    "Painted havelis in Nawalgarh", "Dr. Ramnath A. Podar Haveli Museum", "Morarka Haveli Museum",
    "Roop Niwas Kothi", "Char Chauk ki Haveli", "Mahansar", "Tain, Shekhawati", "Fatehpur, Rajasthan",
    "Churu", "Dundlod", "Jhunjhunu", "Laxmangarh", "Painted havelis",
    # other Rajasthani / Indian wall paintings (fresco technique, real damage)
    "Wall paintings of Hardoti School in Garh Palace, Bundi", "Chitrasala, Garh Palace, Bundi",
    "Chhatra Mahal, Garh Palace, Bundi", "Murals in Rajasthan",
    "Frescos in Main Sikh Haveli", "Paintings in Main Sikh Haveli",
    "Paintings in the Ajanta Caves",  # Kerala murals skipped: mostly modern reproductions, already in Kaggle set
]
LICENSE_OK = re.compile(r"^(cc0|public domain|pd|cc[ -]by(-sa)?([ -][\d.]+)?)", re.I)
SKIP_CATS = re.compile(r"street art|graffiti|hotel|people|portrait|dance|music|puppet|vehicle|food|map", re.I)


def api(params, retries=8):
    params = {**params, "format": "json", "maxlag": "5"}
    url = API + "?" + urllib.parse.urlencode(params)
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode("utf-8"))
            if "error" in d and d["error"].get("code") == "maxlag":
                time.sleep(5 * (k + 1)); continue
            time.sleep(1.0)
            return d
        except Exception as e:
            time.sleep(min(60, 5 * 2 ** k))
    raise RuntimeError(f"API failed: {url}")


def members(cat, cmtype):
    out, cont = [], {}
    while True:
        d = api({"action": "query", "list": "categorymembers", "cmtitle": "Category:" + cat, "cmtype": cmtype, "cmlimit": "500", **cont})
        out += d.get("query", {}).get("categorymembers", [])
        if "continue" not in d:
            return out
        cont = d["continue"]


def walk(seed, depth, exclude):
    seen, files, frontier = {seed}, {}, [(seed, 0)]
    while frontier:
        cat, dpt = frontier.pop()
        for f in members(cat, "file"):
            files.setdefault(f["title"], cat)
        if dpt < depth:
            for s in members(cat, "subcat"):
                name = s["title"].split(":", 1)[1]
                if name in seen or exclude.search(name) or SKIP_CATS.search(name):
                    continue
                seen.add(name)
                frontier.append((name, dpt + 1))
    return files


def file_info(titles):
    d = api({"action": "query", "titles": "|".join(titles), "prop": "imageinfo|categories", "cllimit": "max",
             "iiprop": "url|size|mime|extmetadata", "iiurlwidth": "1280"})
    return list(d.get("query", {}).get("pages", {}).values())


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/external/commons_frescoes"))
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--exclude", default="mandawa", help="regex; matching titles/categories are skipped")
    ap.add_argument("--max-per-seed", type=int, default=150)
    ap.add_argument("--min-side", type=int, default=600)
    ap.add_argument("--discover", action="store_true")
    a = ap.parse_args()
    exclude = re.compile(a.exclude, re.I)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    plan = {}
    for seed in SEEDS:
        try:
            fs = walk(seed, a.depth, exclude)
        except RuntimeError as e:
            print(f"{seed}: failed ({e})"); continue
        fs = {t: c for t, c in fs.items() if not exclude.search(t) and t.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))}
        print(f"{seed}: {len(fs)} image files")
        plan[seed] = fs
    if a.discover:
        return

    rows, done = [], set()
    att = out / "attribution.csv"
    if att.exists():
        rows = list(csv.DictReader(open(att, encoding="utf-8")))
        done = {r["title"] for r in rows}
    stats = {"downloaded": 0, "license": 0, "small": 0, "excluded_cat": 0, "dup": 0}
    for seed, fs in plan.items():
        titles = [t for t in fs if t not in done][: a.max_per_seed]
        for i in range(0, len(titles), 40):
            for pg in file_info(titles[i : i + 40]):
                t = pg.get("title")
                if t in done or "imageinfo" not in pg:
                    stats["dup"] += 1; continue
                cats = " ".join(c["title"] for c in pg.get("categories", []))
                if exclude.search(cats):
                    stats["excluded_cat"] += 1; continue
                ii = pg["imageinfo"][0]
                meta = ii.get("extmetadata", {})
                lic = strip_html(meta.get("LicenseShortName", {}).get("value", ""))
                if not LICENSE_OK.search(lic):
                    stats["license"] += 1; continue
                if min(ii.get("width", 0), ii.get("height", 0)) < a.min_side:
                    stats["small"] += 1; continue
                url = ii.get("thumburl") or ii["url"]
                slug = re.sub(r"[^A-Za-z0-9]+", "_", seed).strip("_")[:40]
                name = re.sub(r"[^A-Za-z0-9._-]+", "_", t.split(":", 1)[1])[:120]
                dst = out / slug / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                ok = False
                for k in range(6):
                    try:
                        req = urllib.request.Request(url, headers={"User-Agent": UA})
                        dst.write_bytes(urllib.request.urlopen(req, timeout=120).read())
                        ok = True
                        time.sleep(1.0)
                        break
                    except urllib.error.HTTPError as e:
                        if e.code != 429:
                            break
                        time.sleep(min(300, 15 * 2 ** k))  # rate-limited: back off
                    except Exception:
                        time.sleep(5)
                if not ok:
                    print("  download failed", t.encode("ascii", "replace").decode()); continue
                done.add(t)
                rows.append({"title": t, "file": dst.relative_to(out).as_posix(), "seed": seed, "license": lic,
                             "artist": strip_html(meta.get("Artist", {}).get("value", ""))[:200],
                             "source_url": ii.get("descriptionurl", ""), "width": ii.get("width"), "height": ii.get("height")})
                stats["downloaded"] += 1
        print(f"  {seed}: total so far {stats}")
        with open(att, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, list(rows[0].keys()) if rows else ["title"])
            w.writeheader(); w.writerows(rows)
    print("Done:", stats, "->", out)


if __name__ == "__main__":
    main()
