"""Unlabelled store frames for Fourier Domain Adaptation: the A3 hand-off to B.

    python scripts/research/fda_frames.py footage/fda --per-clip 200 --out data/fda_frames

Takes frames evenly spread over each clip in a folder (or the clips named) and
writes them as JPEGs to `--out`, which is what `fda_target_dir` in
train_stage1.ipynb and `run_rq2.py --fda-target-dir` read. FDA uses only the
frames' low-frequency amplitude (colour and lighting), so no person in them is
labelled or scored. A manifest beside the frames records each clip's name,
SHA-1 and the frame numbers taken, so `check_data.py` can prove that no FDA clip
is also a test clip.

Frames stay under data/, which git ignores. They show members of the public:
keep them on the project machines (docs/research-data.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm"}


def sha1(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def clips_in(paths):
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in VIDEO_SUFFIXES)
        elif p.suffix.lower() in VIDEO_SUFFIXES:
            out.append(p)
    return out


def export(clip, out_dir, per_clip=200, max_width=640, quality=92):
    """Write up to `per_clip` frames of one clip; returns its manifest entry."""
    import cv2
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        cap.release()
        raise ValueError(f"cannot read frames from {clip}")
    picks = np.unique(np.linspace(0, n - 1, min(per_clip, n)).round().astype(int))
    stem = clip.stem.replace(" ", "_")[:60]
    written = []
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
        name = f"{stem}_{idx:07d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        written.append(int(idx))
    cap.release()
    return {"clip": clip.name, "sha1": sha1(clip), "frames_total": n, "frames": written}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips", nargs="+", help="FDA clips, or folders of them")
    p.add_argument("--out", default="data/fda_frames")
    p.add_argument("--per-clip", type=int, default=200)
    p.add_argument("--max-width", type=int, default=640)
    a = p.parse_args()
    clips = clips_in(a.clips)
    if not clips:
        raise SystemExit("No video files found.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"clips": []}
    done = {c["sha1"] for c in manifest["clips"]}
    for clip in clips:
        h = sha1(clip)
        if h in done:
            print(f"{clip.name}: already exported, skipped")
            continue
        entry = export(clip, out, a.per_clip, a.max_width)
        manifest["clips"].append(entry)
        done.add(h)
        print(f"{clip.name}: {len(entry['frames'])} frames")
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = sum(len(c["frames"]) for c in manifest["clips"])
    print(f"{total} frames from {len(manifest['clips'])} clips in {out}")


if __name__ == "__main__":
    main()
