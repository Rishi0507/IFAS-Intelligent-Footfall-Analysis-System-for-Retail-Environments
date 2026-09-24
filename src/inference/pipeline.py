"""Video -> annotated MP4 + analytics.json.

    python -m src.inference.pipeline --video store.mp4 --out outputs/runs/store \
        --gender-repo abhshkp/footfall-analysis-vit-stage1 --age-bundle models/age

Detection (YOLO) runs on every sampled frame; the two classifiers run on each
person crop; an IoU tracker turns boxes into people; `analytics.build_analytics`
turns people into the document the dashboard renders.
"""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import shutil
import subprocess
import time
from pathlib import Path

from .analytics import build_analytics
from .tracker import IoUTracker
from .trends import insights

# Same hues the dashboard uses, so the video and the charts agree.
BOX_BGR = {"male": (214, 120, 42), "female": (52, 104, 235), None: (129, 135, 137)}


def _device(pref="auto"):
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def process_video(video_path, out_dir, gender_ckpt=None, gender_repo=None,
                  age_bundle=None, detector_weights="yolov8n.pt", det_conf=0.4,
                  stride=2, device="auto", write_video=True, bin_sec=5.0,
                  min_track_frames=3, max_frames=None, progress=None,
                  vote_every_sec=0.4, max_votes=12, cache_dir=None, source_name=None,
                  captured_at=None, site=None, camera=None,
                  detector=None, gender_model=None, age_model=None):
    """Run the pipeline over one video.

    `detector`, `gender_model` and `age_model` accept already-constructed
    objects; leave them None to build them from the weight arguments.
    """
    import cv2
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(device)
    say = progress or (lambda **kw: None)

    say(stage="loading models", pct=0.0)
    if detector is None:
        from ultralytics import YOLO
        detector = YOLO(detector_weights)

    if gender_model is None and gender_repo:
        from .models import GenderModel
        # Shared cache, not the run directory: a 340 MB checkpoint should be
        # downloaded once, not once per run.
        cache = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[2] / "models" / "hf"
        say(stage="downloading checkpoint", pct=0.0)
        gender_model = GenderModel.from_hub(gender_repo, cache / gender_repo.replace("/", "__"),
                                            device=device)
    elif gender_model is None and gender_ckpt:
        from .models import GenderModel
        gender_model = GenderModel(gender_ckpt, device=device)
    if age_model is None and age_bundle:
        from .models import AgeModel
        age_model = AgeModel(age_bundle)
    if gender_model is None and age_model is None:
        raise ValueError("Nothing to classify: pass a gender checkpoint and/or an age bundle.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames:
        n_frames = min(n_frames, max_frames) if n_frames else max_frames

    writer, codec = None, None
    if write_video:
        writer, codec = _open_writer(cv2, out_dir / "video.mp4", fps, width, height)

    tracker = IoUTracker(iou_thresh=0.3, max_misses=int(fps))
    vote_gap = max(stride, int(round(vote_every_sec * fps)))
    occupancy = {}
    frame_area = float(width * height)
    frame_idx = 0
    t0 = time.time()
    n_classified = 0

    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and frame_idx >= max_frames):
            break

        if frame_idx % stride == 0:
            res = detector(frame, classes=[0], conf=det_conf, verbose=False)[0]
            dets = []
            for box in res.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 - x1 < 8 or y2 - y1 < 16:
                    continue
                dets.append(((x1, y1, x2, y2), float(box.conf[0])))

            matched = tracker.update(dets, frame_idx, frame_area)
            occupancy[frame_idx] = len(dets)

            due = [(t, b) for t, b, _c in matched if t.needs_vote(frame_idx, vote_gap, max_votes)]
            crops = [Image.fromarray(frame[b[1]:b[3], b[0]:b[2]][:, :, ::-1]) for _t, b in due]
            g_out = gender_model.predict(crops) if (gender_model and crops) else [None] * len(crops)
            a_out = age_model.predict(crops) if (age_model and crops) else [None] * len(crops)
            for (track, _b), gp, ap in zip(due, g_out, a_out):
                track.vote(gender_probs=gp, age_probs=ap, frame_idx=frame_idx)
            n_classified += len(crops)

            if writer is not None:
                _draw(cv2, frame, matched)

        elif writer is not None:
            _draw(cv2, frame, [(t, t.box, 0.0) for t in tracker.active])

        if writer is not None:
            writer.write(frame)
        if frame_idx == 0:
            cv2.imwrite(str(out_dir / "poster.jpg"), frame)

        frame_idx += 1
        if frame_idx % 10 == 0:
            done = time.time() - t0
            rate = frame_idx / done if done else 0
            say(stage="processing", frame=frame_idx, total=n_frames,
                pct=round(100 * frame_idx / n_frames, 1) if n_frames else None,
                eta_sec=round((n_frames - frame_idx) / rate) if (n_frames and rate) else None,
                classified=n_classified)

    cap.release()
    if writer is not None:
        writer.release()
        codec = _finalize_video(out_dir / "video.mp4", codec)

    elapsed = time.time() - t0
    say(stage="aggregating", frame=frame_idx, total=n_frames, pct=99.0)

    # When the footage was *recorded* - the axis every trend is grouped by.
    # Falls back to the file's own timestamp, which is right for camera exports.
    if not captured_at:
        try:
            captured_at = _dt.datetime.fromtimestamp(Path(video_path).stat().st_mtime)
        except OSError:
            captured_at = _dt.datetime.now()
    if isinstance(captured_at, str):
        try:
            captured_at = _dt.datetime.fromisoformat(captured_at)
        except ValueError:
            captured_at = _dt.datetime.now()

    meta = {
        "source": source_name or Path(video_path).name,
        "captured_at": captured_at.isoformat(timespec="minutes"),
        # Which shop and which camera: rates from different framings are not
        # comparable, so trends group by these rather than pooling everything.
        "site": site or "Store",
        "camera": camera or "Camera 1",
        "fps": round(fps, 3),
        "width": width, "height": height,
        "frames_total": frame_idx,
        "frames_sampled": len(occupancy),
        "duration_sec": round(frame_idx / fps, 2) if fps else 0.0,
        "processing_sec": round(elapsed, 2),
        "processing_fps": round(frame_idx / elapsed, 2) if elapsed else 0.0,
        "crops_classified": n_classified,
        "device": device,
        "has_video": bool(writer is not None),
        "video_codec": codec,
    }
    config = {
        "detector": detector_weights,
        "det_conf": det_conf,
        "frame_stride": stride,
        "vote_every_sec": vote_every_sec,
        "max_votes_per_person": max_votes,
        "iou_thresh": 0.3,
        "gender_model": gender_repo or (str(gender_ckpt) if gender_ckpt else None),
        "age_model": str(age_bundle) if age_bundle else None,
    }
    model_card = {
        "gender": {"family": "ViT-B/16 fine-tuned (2-stage)", "labels": ["female", "male"],
                   "source": config["gender_model"]},
        "age": {"family": "region colour + LBP histograms, one-vs-rest SVM per bucket",
                "labels": ["Age16-30", "Age31-45", "Age46-60", "AgeAbove61"],
                "reported_avg_mA": 0.835, "source": config["age_model"]},
    }

    doc = build_analytics(tracker.close(), meta, config, occupancy,
                          model_card=model_card, bin_sec=bin_sec,
                          min_frames=min_track_frames)
    # Stored, not computed in the browser: any consumer of the file gets them.
    doc["insights"] = insights(doc)
    with open(out_dir / "analytics.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    say(stage="done", frame=frame_idx, total=n_frames, pct=100.0)
    return doc


def _open_writer(cv2, path, fps, width, height):
    """Prefer H.264: it is the only codec every browser can play.

    Not every OpenCV build ships an H.264 encoder, so fall back to MPEG-4 and
    let `_to_h264` fix it up afterwards if ffmpeg is around.
    """
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
        if writer.isOpened():
            return writer, fourcc
        writer.release()
    raise ValueError(f"Could not open a video writer for {path}")


def _probe_codec(path):
    """Ask ffprobe what actually landed in the file. None if ffprobe is absent."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)]
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip() or None
    except (subprocess.CalledProcessError, OSError):
        return None


def _finalize_video(path, requested):
    """Make sure the MP4 is browser-playable, and report what it actually is.

    `isOpened()` lies on builds whose H.264 encoder fails to load at write time,
    so trust ffprobe over the fourcc we asked for.
    """
    actual = _probe_codec(path)
    if actual == "h264":
        return "avc1"
    if shutil.which("ffmpeg"):
        return _to_h264(path) or actual or requested
    return actual or requested


def _to_h264(path):
    """Transcode in place with ffmpeg if available. Returns the resulting codec."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    tmp = path.with_suffix(".h264.mp4")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError):
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(path)
    return "avc1"


def _draw(cv2, frame, matched):
    for track, box, _conf in matched:
        label, conf = track.gender()
        age, _ = track.age()
        color = BOX_BGR.get(label, BOX_BGR[None])
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        parts = [f"#{track.id}"]
        if label:
            parts.append(f"{label} {conf:.2f}")
        if age:
            parts.append(age)
        text = "  ".join(parts)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, text, (x1 + 4, max(10, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="outputs/runs/latest")
    p.add_argument("--gender-repo", default=None, help="HuggingFace repo id")
    p.add_argument("--gender-ckpt", default=None, help="local checkpoint directory")
    p.add_argument("--age-bundle", default=None, help="directory with classifiers.pkl")
    p.add_argument("--detector", default="yolov8n.pt")
    p.add_argument("--det-conf", type=float, default=0.4)
    p.add_argument("--stride", type=int, default=2, help="classify every Nth frame")
    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--min-track-frames", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--vote-every-sec", type=float, default=0.4,
                   help="classify each tracked person at most this often")
    p.add_argument("--max-votes", type=int, default=12,
                   help="stop classifying a person after this many votes")
    p.add_argument("--site", default=None, help="shop name, e.g. 'MG Road store'")
    p.add_argument("--camera", default=None, help="camera name, e.g. 'Entrance'")
    p.add_argument("--captured-at", default=None,
                   help="when the footage was recorded, ISO format "
                        "(default: the video file's timestamp)")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-video", action="store_true", help="analytics only, skip the MP4")
    a = p.parse_args()

    try:
        doc = process_video(
            a.video, a.out, gender_ckpt=a.gender_ckpt, gender_repo=a.gender_repo,
            age_bundle=a.age_bundle, detector_weights=a.detector, det_conf=a.det_conf,
            stride=a.stride, device=a.device, write_video=not a.no_video,
            bin_sec=a.bin_sec, min_track_frames=a.min_track_frames,
            max_frames=a.max_frames, vote_every_sec=a.vote_every_sec,
            max_votes=a.max_votes, captured_at=a.captured_at,
            site=a.site, camera=a.camera,
            progress=lambda **kw: print(f"  {kw.get('stage')}: {kw.get('pct')}%", end="\r"),
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    k = doc["kpis"]
    print(f"\n{k['visitors']} visitors · peak occupancy {k['peak_occupancy']} · "
          f"avg dwell {k['avg_dwell_sec']}s · {k['male_pct']}% male / {k['female_pct']}% female")
    print(f"written to {a.out}/analytics.json")


if __name__ == "__main__":
    main()
