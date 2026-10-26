"""Label a processed run again, from crops taken through each person's whole stay.

The pipeline votes on each person a bounded number of times as it goes (24
votes, spread over roughly the first 100 s of a stay), and draws a box's label
from that person's latest votes. In a crowded, seated room a box can crawl
from one person to the next over a minute: it widens over two people, then
latches onto the neighbour. The label then rides on someone it was never
measured on.

This pass runs after the pipeline, on the same tracks (track_boxes.json) and
the original video, with the pipeline's own classifiers (load_classifiers):

* every person is classified every `every_sec` seconds of their whole stay; a
  crop overlapping another box by more than `max_overlap` (IoU) is skipped,
  as it shows two people;
* in the annotated video each box is labelled from that person's votes within
  `window_sec` of the moment, so the label describes whoever is in the box
  then, and says "?" when those votes are weak;
* each person's label is pooled over all their votes with the pipeline's
  rules (under the confidence floor, or votes whose halves disagree: "not
  sure"), and analytics.json is rebuilt from the same tracks.

The previous analytics.json and video.mp4 are kept as analytics.v1.json and
video.v1.mp4.

    python -m src.inference.relabel --run outputs/runs/X --video original.avi
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .analytics import build_analytics
from .pipeline import (BOX_BGR, _device, _finalize_video, _open_writer, load_classifiers,
                       make_model_card)
from .tracker import Track, iou


def schedule(tb, every_sec=2.0, max_overlap=0.3, frames=None):
    """{frame: [(track, box)]} to classify, and how many crops were skipped
    for overlapping a neighbour. A person is due every `every_sec`; when the
    due crop overlaps someone, the next clear sighting is taken instead.
    `frames` (first, last) limits it to part of the recording."""
    fps = float(tb["fps"])
    gap = every_sec * fps
    by_frame = {}
    for tid, obs in tb["tracks"].items():
        for o in obs:
            by_frame.setdefault(o[0], []).append((int(tid), o[1:5]))
    due, skipped = {}, 0
    for tid, obs in tb["tracks"].items():
        last = -10 ** 9
        for o in sorted(obs):
            f, box = o[0], o[1:5]
            if frames and not frames[0] <= f <= frames[1]:
                continue
            if f - last < gap:
                continue
            if max((iou(box, b) for t, b in by_frame[f] if t != int(tid)), default=0.0) > max_overlap:
                skipped += 1
                continue
            due.setdefault(f, []).append((int(tid), [int(round(v)) for v in box]))
            last = f
    return due, skipped


def classify(video, due, gender_model, age_model, batch=48, say=None):
    """{track: [(t, gender_probs, age_probs)]} over the scheduled crops."""
    import cv2
    from PIL import Image
    say = say or (lambda **kw: None)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    votes, pending = {}, []
    total = sum(len(v) for v in due.values())
    done, t0 = 0, time.time()

    def flush():
        nonlocal done
        if not pending:
            return
        crops = [c for _t, _s, c in pending]
        g = gender_model.predict(crops) if gender_model else [None] * len(crops)
        a = age_model.predict(crops) if age_model else [None] * len(crops)
        for (tid, sec, _c), gp, ap in zip(pending, g, a):
            votes.setdefault(tid, []).append((sec, gp, ap))
        done += len(pending)
        pending.clear()
        rate = done / max(time.time() - t0, 1e-6)
        say(stage="classifying", done=done, total=total, eta_sec=round((total - done) / rate) if rate else None)

    last = max(due) if due else -1
    frame_idx = 0
    if due and min(due) > 250:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(due))
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    while frame_idx <= last:
        if frame_idx not in due:
            if not cap.grab():
                break
            frame_idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        for tid, (x1, y1, x2, y2) in due[frame_idx]:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            pending.append((tid, frame_idx / fps, Image.fromarray(frame[y1:y2, x1:x2][:, :, ::-1].copy())))
        if len(pending) >= batch:
            flush()
        frame_idx += 1
    flush()
    cap.release()
    for v in votes.values():
        v.sort(key=lambda x: x[0])
    return votes


def window_label(votes, which, t, window_sec):
    """(label, confidence) from one person's votes within `window_sec` of t;
    the nearest three votes when none are that close; (None, 0) without any."""
    if not votes:
        return None, 0.0
    times = [v[0] for v in votes]
    lo, hi = bisect.bisect_left(times, t - window_sec), bisect.bisect_right(times, t + window_sec)
    near = votes[lo:hi]
    if not near:
        i = bisect.bisect_left(times, t)
        near = sorted(votes[max(0, i - 3):i + 3], key=lambda v: abs(v[0] - t))[:3]
    probs = [v[which + 1] for v in near if v[which + 1]]
    if not probs:
        return None, 0.0
    mean = {k: sum(p.get(k, 0.0) for p in probs) / len(probs) for k in probs[0]}
    top = max(mean, key=mean.get)
    return top, mean[top]


def rebuild_tracks(doc, tb, votes):
    """The pipeline's Track objects again, from the stored boxes, with the
    new votes: what build_analytics needs."""
    w, h = doc["meta"]["width"], doc["meta"]["height"]
    area = float(w * h)
    old = {t["id"]: t for t in doc.get("tracks", [])}
    tracks = []
    for tid, obs in tb["tracks"].items():
        tid = int(tid)
        obs = [tuple(o) for o in sorted(obs)]
        f0, *b0, c0 = obs[0]
        t = Track(id=tid, box=tuple(obs[-1][1:5]), first_frame=f0, last_frame=obs[-1][0],
                  frames=len(obs), det_conf_sum=sum(o[5] for o in obs),
                  rejoins=old.get(tid, {}).get("rejoins", 0),
                  path=[(o[0], round((o[1] + o[3]) / 2.0, 1), round((o[2] + o[4]) / 2.0, 1),
                         round((o[3] - o[1]) * (o[4] - o[2]) / area, 5)) for o in obs],
                  box_log=list(obs))
        for sec, gp, ap in votes.get(tid, []):
            t.vote(gender_probs=gp, age_probs=ap, frame_idx=int(round(sec * tb["fps"])))
        tracks.append(t)
    return tracks


def _draw_labelled(cv2, frame, boxes, min_gender_conf, min_age_conf):
    """The pipeline's box style (pipeline._draw), with labels given."""
    for tid, box, (label, conf), (age, age_conf) in boxes:
        unsure = label is not None and conf < min_gender_conf
        if unsure:
            label = None
        if age is not None and age_conf < min_age_conf:
            age = None
        color = BOX_BGR.get(label, BOX_BGR[None])
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        parts = [f"#{tid}"]
        if unsure:
            parts.append("?")
        if label:
            parts.append(f"{label}{'?' if conf < 0.7 else ''} {conf:.2f}")
        if age:
            parts.append(age)
        text = "  ".join(parts)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, text, (x1 + 4, max(10, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def render(video, out_path, tb, votes, window_sec, min_gender_conf, min_age_conf, say=None, frames=None):
    """The annotated video again: every counted person's box on each frame
    they were seen, labelled from their votes around that moment. `video`
    may be smaller than the frames the boxes were found on (clean.mp4):
    boxes are scaled to it. Encoded through ffmpeg when it is installed."""
    import cv2
    say = say or (lambda **kw: None)
    fps = float(tb["fps"])
    stride = int(tb.get("stride") or 1)
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    k = w / float(tb.get("width") or w)
    by_frame = {}
    for tid, obs in tb["tracks"].items():
        for o in obs:
            by_frame.setdefault(o[0], []).append((int(tid), [v * k for v in o[1:5]]))
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        pipe = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                 "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                                 "-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p",
                                 "-movflags", "+faststart", "-an", str(out_path)], stdin=subprocess.PIPE)
        writer, codec = None, "avc1"
    else:
        writer, codec = _open_writer(cv2, out_path, fps, w, h)
    labels = {}
    frame_idx = 0
    if frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    while not frames or frame_idx <= frames[1]:
        ok, frame = cap.read()
        if not ok:
            break
        s = frame_idx - frame_idx % stride
        if s not in labels:
            labels = {s: [(tid, box, window_label(votes.get(tid), 0, s / fps, window_sec),
                           window_label(votes.get(tid), 1, s / fps, window_sec))
                          for tid, box in by_frame.get(s, [])]}
        _draw_labelled(cv2, frame, labels[s], min_gender_conf, min_age_conf)
        if writer is None:
            pipe.stdin.write(frame.tobytes())
        else:
            writer.write(frame)
        frame_idx += 1
        if frame_idx % 250 == 0:
            say(stage="drawing", frame=frame_idx, total=n)
    cap.release()
    if writer is None:
        pipe.stdin.close()
        if pipe.wait() != 0:
            raise RuntimeError("ffmpeg could not write the video")
        return codec
    writer.release()
    return _finalize_video(out_path, codec)


def classifiers(gender_repo, attributes, age, device, say):
    """The pipeline's classifiers for `attributes`, with CLIP answering age
    when `age` is "clip" and they have no age model of their own (the
    original SVMs are at chance on held-out people)."""
    gender_model, age_model, clip_meta = load_classifiers(gender_repo=gender_repo, attributes=attributes,
                                                          device=device, say=say)
    if age == "clip" and age_model is None:
        from .clip_attrs import load_clip_attributes
        both = load_clip_attributes(device=device, say=say)
        age_model, clip_meta = both.age, {"kind": both.kind, **both.meta}
    return gender_model, age_model, clip_meta


def clip(run_dir, video, out_path, start_sec, end_sec, gender_repo=None, attributes="vit", age="clip",
         every_sec=1.5, window_sec=8.0, max_overlap=0.3, min_gender_conf=0.6, min_age_conf=0.4,
         device="auto", pad_sec=10.0, say=None):
    """One stretch of the recording as its own annotated video, labelled like
    relabel() but from votes taken only around that stretch. The run itself
    is not changed. Returns the votes, for checking."""
    say = say or (lambda **kw: None)
    run_dir = Path(run_dir)
    doc = json.loads((run_dir / "analytics.json").read_text(encoding="utf-8"))
    tb = json.loads((run_dir / "track_boxes.json").read_text(encoding="utf-8"))
    fps = float(tb["fps"])
    gender_repo = gender_repo or doc["config"].get("gender_model")
    gender_model, age_model, _ = classifiers(gender_repo, attributes, age, _device(device), say)
    span = (int((start_sec - pad_sec) * fps), int((end_sec + pad_sec) * fps))
    due, _ = schedule(tb, every_sec, max_overlap, frames=span)
    votes = classify(video, due, gender_model, age_model, say=say)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    render(video, Path(out_path), tb, votes, window_sec, min_gender_conf, min_age_conf, say=say,
           frames=(int(start_sec * fps), int(end_sec * fps)))
    return votes


def relabel(run_dir, video, gender_repo=None, attributes="ensemble", every_sec=2.0, window_sec=8.0,
            max_overlap=0.3, min_gender_conf=0.6, min_age_conf=0.4, device="auto", draw=True, say=None):
    """Classify a run's people again and rewrite analytics.json (and video.mp4)."""
    from .trends import insights
    from .guide import shop_guide
    say = say or (lambda **kw: None)
    run_dir = Path(run_dir)
    doc = json.loads((run_dir / "analytics.json").read_text(encoding="utf-8"))
    tb = json.loads((run_dir / "track_boxes.json").read_text(encoding="utf-8"))
    gender_repo = gender_repo or doc["config"].get("gender_model")
    device = _device(device)
    gender_model, age_model, clip_meta = load_classifiers(gender_repo=gender_repo, attributes=attributes,
                                                          device=device, say=say)
    t0 = time.time()
    due, skipped = schedule(tb, every_sec, max_overlap)
    votes = classify(video, due, gender_model, age_model, say=say)
    # Saved first: the classifying is the expensive part.
    (run_dir / "votes.json").write_text(json.dumps(
        {str(k): [[round(s, 2), g, a] for s, g, a in v] for k, v in votes.items()}), encoding="utf-8")
    tracks = rebuild_tracks(doc, tb, votes)

    meta = dict(doc["meta"])
    config = dict(doc["config"])
    config.update({"attributes": attributes, "gender_model": gender_repo, "age_model": None,
                   "min_gender_conf": min_gender_conf, "min_age_conf": min_age_conf,
                   "relabel": {"every_sec": every_sec, "window_sec": window_sec, "max_overlap": max_overlap,
                               "votes": sum(len(v) for v in votes.values()), "skipped_overlap": skipped}})
    card = make_model_card(config, attributes, gender_model, age_model, clip_meta)
    stride = int(tb.get("stride") or config.get("frame_stride") or 1)
    sampled = range(0, int(meta["frames_total"]), stride)
    occupancy = {f: sum(1 for t in tracks if t.first_frame <= f <= t.last_frame) for f in sampled}
    new = build_analytics(tracks, meta, config, occupancy, model_card=card,
                          bin_sec=config.get("bin_sec", 5.0), min_frames=config.get("min_track_frames", 3),
                          min_gender_conf=min_gender_conf, min_age_conf=min_age_conf)
    for k in ("tracks_rejoined", "hunches_checked", "hunches_dropped"):
        if k in doc["kpis"]:
            new["kpis"][k] = doc["kpis"][k]
    new["meta"]["crops_classified"] = config["relabel"]["votes"]
    new["meta"]["relabelled_sec"] = round(time.time() - t0, 1)
    if doc.get("zones"):
        new["zones"] = doc["zones"]
    new["insights"] = insights(new)
    new["guide"] = shop_guide(new)

    old = run_dir / "analytics.json"
    if not (run_dir / "analytics.v1.json").exists():
        shutil.copyfile(old, run_dir / "analytics.v1.json")
    tmp_doc = run_dir / "analytics.json.part"
    tmp_doc.write_text(json.dumps(new, indent=1), encoding="utf-8")
    tmp_doc.replace(old)
    if draw:
        tmp = run_dir / "video.relabel.mp4"
        # The light clean copy when there is one: drawing 1080p is the slow part.
        src = run_dir / "clean.mp4" if (run_dir / "clean.mp4").exists() else video
        render(src, tmp, tb, votes, window_sec, min_gender_conf, min_age_conf, say=say)
        old_video = run_dir / "video.mp4"
        try:
            if old_video.exists() and not (run_dir / "video.v1.mp4").exists():
                old_video.replace(run_dir / "video.v1.mp4")
            tmp.replace(old_video)
        except PermissionError:
            # Windows will not move a file the dashboard is streaming.
            say(stage="video kept as video.relabel.mp4: video.mp4 is open elsewhere; swap it after closing that")
    return new


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="a run directory with analytics.json and track_boxes.json")
    p.add_argument("--video", required=True, help="the original recording the run was made from")
    p.add_argument("--gender-repo", default=None, help="defaults to the run's gender model")
    p.add_argument("--attributes", default="ensemble", choices=("vit", "clip", "ensemble"))
    p.add_argument("--every-sec", type=float, default=2.0)
    p.add_argument("--window-sec", type=float, default=8.0)
    p.add_argument("--max-overlap", type=float, default=0.3)
    p.add_argument("--min-gender-conf", type=float, default=0.6)
    p.add_argument("--min-age-conf", type=float, default=0.4)
    p.add_argument("--no-video", action="store_true", help="rewrite analytics.json only")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    import torch
    torch.set_num_threads(os.cpu_count() or 1)

    def say(**kw):
        print("  ".join(f"{k}={v}" for k, v in kw.items()), flush=True)

    doc = relabel(a.run, a.video, gender_repo=a.gender_repo, attributes=a.attributes, every_sec=a.every_sec,
                  window_sec=a.window_sec, max_overlap=a.max_overlap, min_gender_conf=a.min_gender_conf,
                  min_age_conf=a.min_age_conf, device=a.device, draw=not a.no_video, say=say)
    t = doc["tracks"]
    print(f"{len(t)} people: {sum(x['gender'] == 'female' for x in t)} women, "
          f"{sum(x['gender'] == 'male' for x in t)} men, {sum(x['gender'] is None for x in t)} not sure")


if __name__ == "__main__":
    main()
