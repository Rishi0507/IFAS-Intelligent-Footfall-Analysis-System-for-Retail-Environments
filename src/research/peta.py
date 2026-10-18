"""PETA records and the train / val / test splits used by every RQ1 and RQ2 run.

PETA (Deng et al., 2014) gathers 19,000 images from ten surveillance datasets.
Each subset folder holds a Label.txt with one line per person,

    <pid> <attribute> <attribute> ...     e.g.  "5 personalMale personalLess30 ..."

and that person's images are named `<pid>_*` or `<pid>.*` next to it.

Two splits are built, both 9,500 / 1,900 / 7,600 in proportion:

* `identity` (primary): every image of a person lands in the same split, so the
  test set contains only people the model never saw. PETA has several images
  per person, and an image-level split lets a model score well by
  recognising the person (Jia et al., "Rethinking of Pedestrian Attribute
  Recognition", 2021, introduced the person-disjoint PETA-ZS for this reason).
* `image` (for comparison only): the image-level random split the original
  notebooks use, so the paper can say how much the leak inflated their numbers.
"""
from __future__ import annotations
import csv
import random
from collections import defaultdict
from pathlib import Path

AGE_TAGS = {
    "personalLess15": None,          # dropped: no matching age group
    "personalLess30": "Age16-30",
    "personalLess45": "Age31-45",
    "personalLess60": "Age46-60",
    "personalLarger60": "AgeAbove61",
}
GENDER_TAGS = {"personalMale": "male", "personalFemale": "female"}
SPLIT_SIZES = {"train": 9500, "val": 1900, "test": 7600}
IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}


def parse_peta(root):
    """Every labelled image under `root` (any depth), as dicts with
    image (path relative to root), subset, pid, gender ("male"/"female"/None),
    age (one of the four groups or None). Sorted by image path."""
    root = Path(root)
    records = []
    for label_file in sorted(root.rglob("Label.txt")):
        d = label_file.parent
        subset = d.parent.name if d.name.lower() == "archive" else d.name
        files = [p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        by_pid = defaultdict(list)
        for p in files:
            stem = p.stem
            by_pid[stem.split("_")[0] if "_" in stem else stem].append(p)
        for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if not parts:
                continue
            pid, tags = parts[0], set(parts[1:])
            gender = next((g for t, g in GENDER_TAGS.items() if t in tags), None)
            age = next((AGE_TAGS[t] for t in AGE_TAGS if t in tags), None)
            for p in sorted(by_pid.get(pid, [])):
                records.append({"image": p.relative_to(root).as_posix(), "subset": subset,
                                "pid": pid, "gender": gender, "age": age})
    records.sort(key=lambda r: r["image"])
    return records


def _targets(n_total):
    total = sum(SPLIT_SIZES.values())
    t = {k: round(n_total * v / total) for k, v in SPLIT_SIZES.items()}
    t["test"] = n_total - t["train"] - t["val"]
    return t


def identity_split(records, seed=42):
    """{image: split}, with every image of one (subset, pid) in one split.

    People are shuffled within each gender and dealt into train, val and test
    until each split holds its share of that gender's images, so the gender
    balance is the same in every split.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in records:
        groups[(r["subset"], r["pid"])].append(r)
    by_gender = defaultdict(list)
    for key, rs in sorted(groups.items()):
        by_gender[rs[0]["gender"]].append(key)
    out = {}
    for gender in sorted(by_gender, key=str):
        keys = by_gender[gender]
        rng.shuffle(keys)
        n = sum(len(groups[k]) for k in keys)
        targets, filled, order = _targets(n), defaultdict(int), ["train", "val", "test"]
        for k in keys:
            # first split still short of its target; test takes the remainder
            split = next((s for s in order[:-1] if filled[s] < targets[s]), "test")
            for r in groups[k]:
                out[r["image"]] = split
            filled[split] += len(groups[k])
    return out


def image_split(records, seed=42):
    """{image: split}: the image-level random split, stratified by gender.
    People leak across splits; kept only to measure that leak."""
    rng = random.Random(seed)
    by_gender = defaultdict(list)
    for r in records:
        by_gender[r["gender"]].append(r["image"])
    out = {}
    for gender in sorted(by_gender, key=str):
        imgs = sorted(by_gender[gender])
        rng.shuffle(imgs)
        t = _targets(len(imgs))
        for i, img in enumerate(imgs):
            out[img] = "train" if i < t["train"] else "val" if i < t["train"] + t["val"] else "test"
    return out


def write_split_csv(records, path, seed=42):
    """One CSV with both splits as columns: image, subset, pid, gender, age,
    split_identity, split_image."""
    ident, img = identity_split(records, seed), image_split(records, seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["image", "subset", "pid", "gender", "age", "split_identity", "split_image"])
        w.writeheader()
        for r in records:
            w.writerow({**r, "gender": r["gender"] or "", "age": r["age"] or "",
                        "split_identity": ident[r["image"]], "split_image": img[r["image"]]})


def read_split_csv(path, split, column="split_identity", require_gender=True):
    """Rows of one split from a CSV written by `write_split_csv`."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r[column] == split]
    return [r for r in rows if r["gender"]] if require_gender else rows
