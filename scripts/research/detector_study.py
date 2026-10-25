"""RQ4: how many of the people in a dense, seated room does the detector find?

    python scripts/research/detector_study.py --video clip.avi \
        [--frames 20] [--models yolov8n.pt yolov8m.pt] [--tiles none 2x2]

Zone occupancy can only report a seat as taken if the person in it is
detected, and in a classroom most people are small, seated and half hidden by
the row in front. This runs each detector, with and without tiling (the
pipeline's `detect_people`, unchanged), on frames spread evenly through the
clip, and records how many people each finds at several confidence
thresholds, and how long a frame takes.

There are no box labels here, so a count is not a recall: a higher count can
include false positives. With `--counts counts.json` ({"<frame>": people in
view, counted by a person}), each row also gets the count as a share of the
people really there, which bounds recall from above.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.pipeline import detect_people, parse_tiles  # noqa: E402
from src.research.data import results_meta  # noqa: E402

THRESHOLDS = [0.25, 0.4, 0.6]


def sample_frames(video, n):
    import cv2
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = [round((i + 0.5) * total / n) for i in range(n)]
    frames = []
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            frames.append((f, img))
    cap.release()
    return frames, total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8m.pt"])
    p.add_argument("--tiles", nargs="+", default=["none", "2x2"])
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--source-label", default=None,
                   help="what to record as the footage's source; recorder exports often carry a network address")
    p.add_argument("--counts", default=None, help="JSON of people in view per frame, counted by a person")
    p.add_argument("--out", default=str(ROOT / "paper" / "results" / "rq4_detection.json"))
    a = p.parse_args()

    from ultralytics import YOLO
    frames, total = sample_frames(a.video, a.frames)
    counts = None
    if a.counts:
        with open(a.counts, encoding="utf-8") as f:
            counts = {int(k): int(v) for k, v in json.load(f).items()}

    rows = []
    for name in a.models:
        model = YOLO(name)
        model(frames[0][1], classes=[0], verbose=False)              # warm up before timing
        for tiles in a.tiles:
            grid = parse_tiles(tiles)
            per_frame, secs = [], []
            for f, img in frames:
                t0 = time.perf_counter()
                found = detect_people(model, img, min(THRESHOLDS), grid, imgsz=a.imgsz)
                secs.append(time.perf_counter() - t0)
                per_frame.append((f, [c for _b, c in found]))
            for th in THRESHOLDS:
                n = [sum(c >= th for c in confs) for _f, confs in per_frame]
                row = {"model": Path(name).stem, "tiles": tiles, "imgsz": a.imgsz, "conf": th,
                       "frames": len(n), "mean": round(mean(n), 2), "sd": round(pstdev(n), 2),
                       "min": min(n), "max": max(n), "sec_per_frame": round(mean(secs), 3),
                       "per_frame": {str(f): k for (f, _c), k in zip(per_frame, n)}}
                if counts:
                    share = [k / counts[f] for (f, _c), k in zip(per_frame, n) if counts.get(f)]
                    row["share_of_people"] = round(mean(share), 3) if share else None
                rows.append(row)
                print(f"{row['model']:>8} tiles {tiles:>4} conf >= {th:.2f}: "
                      f"{row['mean']:5.1f} people per frame (sd {row['sd']:.1f}), {row['sec_per_frame']:.2f}s")

    meta = results_meta(models=a.models, source=a.source_label or Path(a.video).name, frames=[f for f, _ in frames],
                        frames_total=total, thresholds=THRESHOLDS, counted_by_person=bool(counts))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    print(f"written {out}")


if __name__ == "__main__":
    main()
