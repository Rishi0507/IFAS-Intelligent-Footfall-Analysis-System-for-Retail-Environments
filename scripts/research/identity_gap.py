"""How far does one person's box move between samples, against how much two
neighbours' boxes overlap? The answer sets the tracker's matching threshold.

    python scripts/research/identity_gap.py --run outputs/runs/<run> [--out paper/results/identity_gap.json]

From the run's track_boxes.json: the IoU between consecutive samples of the
same track, and the IoU between every pair of different people's boxes in the
same frame that overlap at all. A threshold that nearly every same-person pair
clears and no neighbour pair reaches keeps a missed person from being
continued on the box of the person beside them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.tracker import iou  # noqa: E402
from src.research.data import results_meta  # noqa: E402

THRESHOLDS = [0.3, 0.4, 0.5, 0.55, 0.6]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--every", type=int, default=10, help="use every Nth sampled frame for neighbour pairs")
    p.add_argument("--out", default=str(ROOT / "paper" / "results" / "identity_gap.json"))
    a = p.parse_args()
    tb = json.loads((Path(a.run) / "track_boxes.json").read_text(encoding="utf-8"))
    stride = tb["stride"]
    same, by_frame = [], {}
    for obs in tb["tracks"].values():
        for x, y in zip(obs, obs[1:]):
            if y[0] - x[0] == stride:
                same.append(iou(x[1:5], y[1:5]))
        for o in obs:
            by_frame.setdefault(o[0], []).append(o[1:5])
    diff = []
    for f in sorted(by_frame)[::a.every]:
        boxes = by_frame[f]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                v = iou(boxes[i], boxes[j])
                if v > 0.05:
                    diff.append(v)
    same, diff = np.array(same), np.array(diff)
    rows = [{"iou": th, "same_person_kept": round(float(np.mean(same >= th)), 4),
             "neighbour_pairs_at_or_above": round(float(np.mean(diff >= th)), 4)} for th in THRESHOLDS]
    doc = {"meta": results_meta(run=Path(a.run).name, stride=stride, every=a.every),
           "same_person": {"pairs": int(len(same)), "percentiles": {str(q): round(float(np.percentile(same, q)), 3)
                                                                     for q in (1, 5, 10, 25, 50)}},
           "neighbours": {"pairs": int(len(diff)), "percentiles": {str(q): round(float(np.percentile(diff, q)), 3)
                                                                    for q in (50, 75, 90, 95, 99)}},
           "rows": rows}
    Path(a.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    for r in rows:
        print(f"IoU {r['iou']}: keeps {100 * r['same_person_kept']:.1f}% of same-person pairs, "
              f"{100 * r['neighbour_pairs_at_or_above']:.1f}% of neighbour pairs reach it")
    print(f"neighbour 99th percentile {doc['neighbours']['percentiles']['99']}; written {a.out}")


if __name__ == "__main__":
    main()
