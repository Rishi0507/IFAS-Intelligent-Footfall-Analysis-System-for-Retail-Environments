"""How often does a track leave the person it started on? Two measures from a
run's track_boxes.json, no labels needed.

    python scripts/research/identity_drift.py --run outputs/runs/<run> [--out paper/results/identity_drift.json]

* wander: tracks whose box centre moves more than `widths` median box widths
  from where the track began. In a seated room almost nobody moves that far,
  so most of these are boxes passing from one person to the next.
* handoffs: places where a track ended and a new one began within
  `max_gap_sec` in the same spot (IoU of the two boxes at least 0.5): one
  person the tracker lost and counted again (src/inference/proof.py).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.proof import handoffs  # noqa: E402
from src.research.data import results_meta  # noqa: E402


def wander(tb, widths=1.5):
    """Tracks whose centre moves more than `widths` median box widths from its start."""
    out = []
    for tid, obs in tb["tracks"].items():
        obs = sorted(obs)
        w = statistics.median(o[3] - o[1] for o in obs)
        x0, y0 = (obs[0][1] + obs[0][3]) / 2, (obs[0][2] + obs[0][4]) / 2
        far = max(abs((o[1] + o[3]) / 2 - x0) + abs((o[2] + o[4]) / 2 - y0) for o in obs)
        if w and far > widths * w:
            out.append({"track": int(tid), "widths": round(far / w, 2)})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--widths", type=float, default=1.5)
    p.add_argument("--max-gap-sec", type=float, default=30.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    tb = json.loads((Path(a.run) / "track_boxes.json").read_text(encoding="utf-8"))
    far = wander(tb, a.widths)
    found = handoffs(tb, max_gap_sec=a.max_gap_sec)
    doc = {"meta": results_meta(run=Path(a.run).name, widths=a.widths, max_gap_sec=a.max_gap_sec),
           "tracks": len(tb["tracks"]), "wandered": len(far), "handoffs": len(found),
           "rows": far, "handoff_rows": found}
    print(f"{len(far)} of {len(tb['tracks'])} tracks wander more than {a.widths} box widths; "
          f"{len(found)} restarts in the same spot within {a.max_gap_sec:.0f} s")
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
