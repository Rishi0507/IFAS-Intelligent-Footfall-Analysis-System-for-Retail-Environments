#!/usr/bin/env python3
"""cctv_gender.py — CLI entry point for the IPD pipeline (no age)."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.config import load_config
from src.pipeline import StoreAnalytics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="IPD CCTV gender analytics (no age)")
    p.add_argument("--video", required=True, help="input video path")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--config", default=None, help="YAML config (optional)")
    p.add_argument("--dump-every", type=int, default=30,
                   help="save annotated frame every N frames")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--end-frame", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.max_frames:
        cfg["pipeline"]["max_frames"] = args.max_frames
    cfg["pipeline"]["dump_annotated_every"] = args.dump_every

    sa = StoreAnalytics(cfg)
    summary = sa.process_video(
        args.video, args.output,
        dump_annotated_every=cfg["pipeline"]["dump_annotated_every"],
        process_every_n=cfg["pipeline"]["process_every_n_frames"],
        max_frames=cfg["pipeline"].get("max_frames"),
        start_frame=args.start_frame, end_frame=args.end_frame,
    )
    print("\n=== SUMMARY ===")
    print(json.dumps({
        "frames_processed": summary["frames_processed"],
        "entries": summary["entries"],
        "exits": summary["exits"],
        "unique_track_estimate": summary["unique_track_estimate"],
        "gender_counts": summary["gender_counts"],
        "reports": summary["reports"],
        "annotated_frame_samples": summary["annotated_frame_samples"],
    }, indent=2))


if __name__ == "__main__":
    main()
