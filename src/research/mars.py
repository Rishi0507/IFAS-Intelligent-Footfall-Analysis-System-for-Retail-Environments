"""MARS tracklets with attribute labels: the public video set for RQ1 and RQ3.

MARS (Zheng et al., ECCV 2016) holds pedestrian crops cut by a detector and
followed by a tracker, so its tracklets have the real-world mix of good and bad
frames the pooling study needs. Attribute labels per tracklet come from
github.com/yuange250/MARS-Attribute (`mars_attributes.csv`), which uses the
Market-1501 attribute definitions.

Image files are named `<pid:4><C><cam><T><tracklet:4><F><frame:3>.jpg`, e.g.
`0001C1T0001F001.jpg`, under `bbox_train/<pid>/` and `bbox_test/<pid>/`.
Person ids 0000 and -1 are junk and distractors and carry no labels.

Two label facts are not documented by the source and are handled here:

* The gender code. The CSV stores 0/1 without saying which is male.
  `GENDER_CODE` is a working assumption (0 = male: 59 % of tracklets, as
  expected on a campus), to be confirmed by eye with the contact sheet from
  `scripts/research/get_mars.py sheet` before any result is reported.
* The age groups. MARS uses young / teenager / adult / old, where college-age
  people count as teenagers. They map onto PETA only coarsely; see AGE_TO_PETA.
"""
from __future__ import annotations
import csv
import re
from collections import defaultdict
from pathlib import Path

LABELS_URL = "https://raw.githubusercontent.com/yuange250/MARS-Attribute/master/mars_attributes.csv"

# Working assumption, confirm visually (see module docstring).
GENDER_CODE = {0: "male", 1: "female"}
AGE_NAMES = ["young", "teenager", "adult", "old"]
# Coarse correspondence with the PETA groups the age model predicts. Only the
# old / not-old split is reliable enough to report.
AGE_TO_PETA = {"young": None, "teenager": "Age16-30", "adult": "Age31-60", "old": "AgeAbove61"}

_NAME = re.compile(r"^(\d{4})C(\d)T(\d{4})F(\d{3})\.jpg$", re.IGNORECASE)


def parse_name(name):
    """'0001C1T0001F001.jpg' -> (pid, cam, tracklet, frame) as ints, or None."""
    m = _NAME.match(name)
    return tuple(int(g) for g in m.groups()) if m else None


def load_labels(csv_path):
    """{(pid, cam, tracklet): {"gender": str, "age": str, "raw": dict}}"""
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (int(r["person-id"]), int(r["camera"]), int(r["tracklet"]))
            out[key] = {"gender": GENDER_CODE[int(r["gender"])],
                        "age": AGE_NAMES[int(r["Age"])], "raw": r}
    return out


def build_index(mars_root, csv_path, min_frames=1):
    """Join MARS image folders with the attribute CSV.

    Returns a list of tracklets, each
    {"pid", "cam", "tracklet", "split", "gender", "age", "frames": [relative paths]},
    sorted by (split, pid, cam, tracklet), with frames in time order.
    Tracklets without labels (junk, distractors) are left out.
    """
    root = Path(mars_root)
    labels = load_labels(csv_path)
    groups = defaultdict(list)
    for split in ("train", "test"):
        for p in sorted((root / f"bbox_{split}").rglob("*.jpg")):
            parsed = parse_name(p.name)
            if parsed is None:
                continue
            pid, cam, trk, frame = parsed
            groups[(split, pid, cam, trk)].append((frame, p.relative_to(root).as_posix()))
    index = []
    for (split, pid, cam, trk), frames in sorted(groups.items()):
        lab = labels.get((pid, cam, trk))
        if lab is None or len(frames) < min_frames:
            continue
        index.append({"pid": pid, "cam": cam, "tracklet": trk, "split": split,
                      "gender": lab["gender"], "age": lab["age"],
                      "frames": [f for _, f in sorted(frames)]})
    return index
