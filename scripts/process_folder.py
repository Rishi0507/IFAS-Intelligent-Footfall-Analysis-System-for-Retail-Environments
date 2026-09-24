"""Process a folder of recordings into dated runs - the way history gets built.

Seasonal analytics needs many recordings from the same camera over weeks, not
one clip. That only happens if adding a day's footage is a single command:

    python scripts/process_folder.py footage/entrance \\
        --site "MG Road" --camera "Entrance" \\
        --gender-repo abhshkp/footfall-analysis-vit-stage1 \\
        --age-bundle models/age

Each video becomes `outputs/runs/<site>-<camera>-<timestamp>/`. Capture time
comes from the filename when it looks like a timestamp (most camera exports do:
`cam1_2026-09-05_17-30-00.mp4`), otherwise from the file's own mtime.

Already-processed videos are skipped, so re-running after adding new files only
does the new work.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.pipeline import process_video  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm"}

# Timestamps as camera exports actually write them.
PATTERNS = [
    (r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})[ _T-]+(\d{2})[-_:]?(\d{2})[-_:]?(\d{2})", 6),
    (r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})[ _T-]+(\d{2})[-_:]?(\d{2})", 5),
    (r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", 3),
]


def captured_from_name(path: Path):
    """Pull a timestamp out of the filename, else fall back to the file's mtime."""
    name = path.stem
    for pattern, n in PATTERNS:
        m = re.search(pattern, name)
        if not m:
            continue
        parts = [int(g) for g in m.groups()[:n]]
        parts += [0] * (6 - len(parts))
        try:
            return dt.datetime(*parts), "filename"
        except ValueError:
            continue
    return dt.datetime.fromtimestamp(path.stat().st_mtime), "file timestamp"


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "run"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    p.add_argument("--out-root", default="outputs/runs")
    p.add_argument("--site", default="Store")
    p.add_argument("--camera", default="Camera 1")
    p.add_argument("--gender-repo", default=None)
    p.add_argument("--gender-ckpt", default=None)
    p.add_argument("--age-bundle", default=None)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--det-conf", type=float, default=0.4)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-video", action="store_true",
                   help="skip the annotated MP4 - much faster for bulk history")
    p.add_argument("--limit", type=int, default=None, help="process at most N videos")
    p.add_argument("--dry-run", action="store_true", help="list what would be processed")
    a = p.parse_args()

    folder = Path(a.folder)
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")
    videos = sorted(f for f in folder.rglob("*") if f.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise SystemExit(f"No videos found under {folder}")

    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    done = set()
    for d in out_root.iterdir():
        doc = d / "analytics.json"
        if doc.exists():
            try:
                with open(doc, encoding="utf-8") as f:
                    done.add(json.load(f)["meta"].get("source"))
            except (json.JSONDecodeError, KeyError, OSError):
                pass

    queue = [v for v in videos if v.name not in done]
    skipped = len(videos) - len(queue)
    if a.limit:
        queue = queue[:a.limit]
    print(f"{len(videos)} videos found, {skipped} already processed, {len(queue)} to do.")

    if a.dry_run:
        for v in queue:
            when, how = captured_from_name(v)
            print(f"  {v.name}  ->  {when:%Y-%m-%d %H:%M}  (from {how})")
        return

    failures = []
    for i, video in enumerate(queue, 1):
        when, how = captured_from_name(video)
        run_dir = out_root / f"{slug(a.site)}-{slug(a.camera)}-{when:%Y%m%d-%H%M%S}"
        print(f"[{i}/{len(queue)}] {video.name}  ({when:%Y-%m-%d %H:%M} from {how})")
        try:
            doc = process_video(
                video, run_dir, gender_repo=a.gender_repo, gender_ckpt=a.gender_ckpt,
                age_bundle=a.age_bundle, det_conf=a.det_conf, stride=a.stride,
                device=a.device, write_video=not a.no_video, captured_at=when,
                site=a.site, camera=a.camera, source_name=video.name,
                progress=lambda **kw: print(f"    {kw.get('stage')}: {kw.get('pct')}%", end="\r"),
            )
            k = doc["kpis"]
            print(f"    {k['visitors']} visitors, {doc['meta']['duration_sec']:.0f}s of footage")
        except Exception as exc:                       # one bad file must not stop the batch
            failures.append((video.name, f"{type(exc).__name__}: {exc}"))
            print(f"    FAILED: {type(exc).__name__}: {exc}")

    print(f"\nDone. {len(queue) - len(failures)} processed, {len(failures)} failed.")
    for name, err in failures:
        print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
