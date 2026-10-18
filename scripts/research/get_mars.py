"""Fetch the MARS attribute labels, index the MARS images, check the labels.

MARS itself cannot be downloaded by a script: its authors distribute it by
hand for research use. Get `bbox_train.zip` and `bbox_test.zip` from the MARS
project page (Zheng et al., ECCV 2016) and unzip them into data/mars/, so that
data/mars/bbox_train/0001/0001C1T0001F001.jpg exists. Then:

    python scripts/research/get_mars.py labels   # data/mars/mars_attributes.csv
    python scripts/research/get_mars.py index    # data/mars/tracklets.jsonl
    python scripts/research/get_mars.py sheet    # data/mars/gender_check.jpg

`sheet` draws 12 tracklets per gender code. Look at it and confirm that
GENDER_CODE in src/research/mars.py is right before reporting any MARS number.
Neither the images nor the labels may be redistributed; data/ is git-ignored.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.research.mars import GENDER_CODE, LABELS_URL, build_index  # noqa: E402

DATA = ROOT / "data" / "mars"


def cmd_labels(a):
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "mars_attributes.csv"
    urllib.request.urlretrieve(LABELS_URL, out)
    n = sum(1 for _ in open(out, encoding="utf-8")) - 1
    print(f"{n} labelled tracklets -> {out}")


def cmd_index(a):
    csv_path = DATA / "mars_attributes.csv"
    if not csv_path.exists():
        sys.exit("run `labels` first")
    if not (DATA / "bbox_train").is_dir():
        sys.exit(f"MARS images not found: unzip bbox_train and bbox_test into {DATA}")
    index = build_index(DATA, csv_path, min_frames=a.min_frames)
    with open(DATA / "tracklets.jsonl", "w", encoding="utf-8") as f:
        for t in index:
            f.write(json.dumps(t) + "\n")
    by = Counter((t["split"], t["gender"]) for t in index)
    print(f"{len(index)} tracklets, {sum(len(t['frames']) for t in index)} frames -> {DATA / 'tracklets.jsonl'}")
    for k, v in sorted(by.items()):
        print(f"  {k[0]:5} {k[1]:6} {v}")


def cmd_sheet(a):
    from PIL import Image, ImageDraw
    index = [json.loads(l) for l in open(DATA / "tracklets.jsonl", encoding="utf-8")]
    rng = random.Random(0)
    w, h, n = 64, 128, 12
    sheet = Image.new("RGB", (w * n, (h + 16) * len(GENDER_CODE)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, gender in enumerate(GENDER_CODE.values()):
        pick = rng.sample([t for t in index if t["gender"] == gender], n)
        y = row * (h + 16)
        draw.text((2, y), f"labelled {gender!r} (code {row})", fill="black")
        for col, t in enumerate(pick):
            img = Image.open(DATA / t["frames"][len(t["frames"]) // 2]).convert("RGB").resize((w, h))
            sheet.paste(img, (col * w, y + 16))
    out = DATA / "gender_check.jpg"
    sheet.save(out, quality=90)
    print(f"check this by eye: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("labels")
    i = sub.add_parser("index")
    i.add_argument("--min-frames", type=int, default=1)
    sub.add_parser("sheet")
    a = p.parse_args()
    {"labels": cmd_labels, "index": cmd_index, "sheet": cmd_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
