"""Split PETA into train / val / test with no person on two sides.

    python scripts/research/make_peta_split.py --peta-root data/PETA --out data/PETA/split_identity.csv

PETA has 19,000 images of 8,705 people. A random split by image, which is what
the paper's numbers used, puts many people in both training and test data and
flatters the results (Jia et al., arXiv:2107.03576). This script groups images
by person (subset + person id from each subset's Label.txt) and assigns whole
people to a split, in the PETA protocol's 50 / 10 / 40 proportions by image
count, stratified by gender.

`--mode image` writes the old image-level split instead, so the effect of the
overlap can be measured with the same models.

Nine subsets label people (`<pid> tags`, images named `<pid>_<n>.<ext>` or
`<pid>.<ext>`). CUHK labels files instead (`0606.png tags`) and gives no person
ids, so its people cannot be kept on one side. By default its 4,563 images are
left out of the identity split (`--unidentified exclude`); `--unidentified keep`
puts each of them in as its own unit, accepting that some CUHK people may then
appear on two sides. The summary reports how many images were left out.

Both training notebooks' older loaders matched on `<pid>` alone and so missed
CUHK (age) or everything except CUHK (gender); see paper/README.md.

Output CSV columns: image (relative to the CSV's own folder, so readers need no
other path), gender, age, subset, pid, split.
`gender` is female/male; `age` is one of the four groups the age model uses, or
empty (under 15, or no age tag).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

AGE_TAGS = {
    "personalLess15": None,
    "personalLess30": "Age16-30",
    "personalLess45": "Age31-45",
    "personalLess60": "Age46-60",
    "personalLarger60": "AgeAbove61",
}
GENDER_TAGS = {"personalMale": "male", "personalFemale": "female"}
IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg", ".jfif")
FRACTIONS = {"train": 0.5, "val": 0.1, "test": 0.4}


def parse_peta(root):
    """One record per image: subset, person id, gender, age, path relative to root.

    `identified` is False for images labelled by file name (CUHK), whose person
    is unknown; their `pid` is the file name.
    """
    root = Path(root)
    records = []
    for label_file in sorted(root.glob("*/archive/Label.txt")) + sorted(root.glob("*/Label.txt")):
        folder = label_file.parent
        subset = folder.parent.name if folder.name == "archive" else folder.name
        files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT]
        by_pid = defaultdict(list)
        for p in files:
            by_pid[p.stem.split("_")[0]].append(p)      # images are named <pid>_<n>.<ext> or <pid>.<ext>
        with open(label_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                pid, tags = parts[0], set(parts[1:])
                gender = next((g for t, g in GENDER_TAGS.items() if t in tags), None)
                age = next((AGE_TAGS[t] for t in AGE_TAGS if t in tags), None)
                by_name = folder / pid
                if by_name.suffix.lower() in IMG_EXT and by_name.exists():
                    images, identified = [by_name], False         # labelled by file (CUHK)
                else:
                    images, identified = by_pid.get(pid, []), True
                for img in images:
                    records.append({"image": img.relative_to(root).as_posix(), "gender": gender or "",
                                    "age": age or "", "subset": subset, "pid": pid,
                                    "identified": identified})
    return records


def assign(records, mode="identity", seed=42):
    """Add a "split" field. Identity mode keeps every identified person on one
    side; an unidentified image is its own unit."""
    rng = np.random.default_rng(seed)
    if mode == "image":
        units = [[i] for i in range(len(records))]
    else:
        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[(r["subset"], r["pid"])].append(i)       # unidentified: pid is the file name
        units = list(groups.values())
    # Stratify by gender: fill each split to its share of images, gender by gender.
    by_gender = defaultdict(list)
    for u in units:
        by_gender[records[u[0]]["gender"]].append(u)
    for gender, us in by_gender.items():
        order = rng.permutation(len(us))
        total = sum(len(u) for u in us)
        cuts = np.cumsum([FRACTIONS["train"], FRACTIONS["val"]]) * total
        seen = 0
        for k in order:
            split = "train" if seen < cuts[0] else ("val" if seen < cuts[1] else "test")
            for i in us[k]:
                records[i]["split"] = split
            seen += len(us[k])
    return records


def summary(records):
    out = {"images": len(records), "splits": {}}
    people = defaultdict(set)
    for r in records:
        people[r["split"]].add((r["subset"], r["pid"]))
    for s in FRACTIONS:
        rs = [r for r in records if r["split"] == s]
        out["splits"][s] = {"images": len(rs), "people": len(people[s]),
                            "gender": dict(Counter(r["gender"] or "none" for r in rs)),
                            "age": dict(Counter(r["age"] or "none" for r in rs))}
    out["people_in_train_and_test"] = len(people["train"] & people["test"])
    out["people_in_train_and_val"] = len(people["train"] & people["val"])
    out["unidentified_images"] = sum(1 for r in records if not r.get("identified", True))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--peta-root", required=True, help="folder holding the ten PETA subsets")
    p.add_argument("--out", required=True, help="CSV to write")
    p.add_argument("--mode", choices=["identity", "image"], default="identity")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unidentified", choices=["exclude", "keep"], default="exclude",
                   help="images with no person id (CUHK): leave out of an identity split, or keep "
                        "each as its own unit")
    a = p.parse_args()

    records = parse_peta(a.peta_root)
    if not records:
        raise SystemExit(f"No PETA images found under {a.peta_root} (expected <subset>/archive/Label.txt)")
    records = [r for r in records if r["gender"]]
    excluded = 0
    if a.mode == "identity" and a.unidentified == "exclude":
        excluded = sum(1 for r in records if not r["identified"])
        records = [r for r in records if r["identified"]]
    assign(records, a.mode, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "gender", "age", "subset", "pid", "split"],
                           extrasaction="ignore")
        w.writeheader()
        root = Path(a.peta_root).resolve()
        for r in records:
            w.writerow(dict(r, image=Path(os.path.relpath(root / r["image"], out.parent.resolve())).as_posix()))
    info = summary(records)
    info.update(mode=a.mode, seed=a.seed, unidentified=a.unidentified,
                unidentified_images_left_out=excluded)
    with open(out.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
