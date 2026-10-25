"""Video -> annotated MP4 + analytics.json.

    python -m src.inference.pipeline --video store.mp4 --out outputs/runs/store \
        --gender-repo abhshkp/footfall-analysis-vit-stage2 --age-bundle abhshkp/footfall-analysis-age-svm

Detection (YOLO) runs on every sampled frame; the two classifiers run on each
person crop; an IoU tracker turns boxes into people; `analytics.build_analytics`
turns people into the document the dashboard renders.
"""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from .analytics import build_analytics
from .tracker import IoUTracker, iou, overlap_of_smaller
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
                  vote_every_sec=0.4, max_votes=24, cache_dir=None, source_name=None,
                  captured_at=None, site=None, camera=None,
                  track_conf=0.25, rejoin_sec=8.0, weak_conf=0.10, max_checks=4,
                  verifier_weights="yolov8m.pt", verify_sizes=(640, 960), verify_conf=0.25,
                  detector=None, gender_model=None, age_model=None, verifier=None,
                  export_crops=False, pooling="majority", quality_model=None,
                  tiles=None, tile_overlap=0.2, imgsz=None, zones=None, auto_zones=True,
                  zone_params=None, frame_every_sec=30.0, attributes=None, clip_heads=None,
                  match_iou=None, vote_spread=True, max_vote_gap_sec=45.0, max_vote_overlap=0.3,
                  min_gender_conf=0.6, min_age_conf=0.4):
    """Run the pipeline over one video.

    `detector`, `gender_model` and `age_model` accept already-constructed
    objects; leave them None to build them from the weight arguments.

    Detections are read at three levels of confidence:

    * `det_conf` and up starts a person outright.
    * `track_conf` and up may re-join a person the detector lost, for up to
      `rejoin_sec` seconds.
    * `weak_conf` and up may extend someone already being followed, or open a
      tentative track. A tentative track is a hunch: it only becomes a person
      if a later clear detection confirms it, or if a second, larger detector
      agrees (`verifier_weights`, asked at most `max_checks` times per hunch).
      The larger model is slow, which is why it is only consulted about
      hunches and never run on every frame. Pass `verifier_weights=None` to
      do without it, or set `weak_conf` equal to `track_conf` to turn hunches
      off altogether.

    `export_crops` saves every crop that is classified to `crops/<track>/` and
    writes `crops.jsonl` (one line per crop, with its quality cues) next to
    analytics.json, for labelling and the pooling study (docs/research-plan.md).
    It uses the same vote sampling as classification, so it saves exactly the
    crops the product classified.

    `pooling` picks how a person's votes become one label: "majority" (the
    default), "mean", "confidence" or "quality". "quality" needs `quality_model`,
    a fitted src.research.pooling.QualityModel or the path of its JSON file.

    `tiles` ("2x2", or (cols, rows)) runs the detector on overlapping tiles as
    well as the whole frame and merges the boxes. Seated, half-hidden people in
    a crowded room are small to a 640-pixel detector; on a quarter of the frame
    they are not. It costs about five detector passes per sampled frame.

    Every observed box is written to `track_boxes.json`, and zone occupancy
    (src/inference/zones.py) is computed from it: `zones` is a zones document
    or the path of one; without it, `auto_zones` proposes seats from where
    people sat still. Zones can be redrawn later without running the models.

    Every `frame_every_sec` seconds a clean frame (no boxes drawn) is saved to
    `frames/`: the background for drawing zones, and what annotators label
    zone occupancy on, since they must not see the model's boxes.

    `attributes` picks the classifiers (src/inference/clip_attrs.py):
    "vit" is the fine-tuned ViT for gender and the colour/texture SVMs for age;
    "clip" is CLIP zero-shot for both; "ensemble" averages the ViT and CLIP for
    gender and uses CLIP for age. On held-out PETA identities and on the
    labelled classroom recording, "ensemble" is the most accurate for gender
    and CLIP the better one for age (docs/research-plan.md, Section 10).
    `clip_heads` replaces the zero-shot prompts with heads fitted by
    scripts/research/clip_probe.py.

    Labels are only as good as the crops and the votes behind them, so:

    * `vote_spread` spaces a person's votes out over their whole stay (the gap
      doubles every four votes, up to `max_vote_gap_sec`), up to `max_votes`.
    * A crop that overlaps another person's box by more than
      `max_vote_overlap` (IoU) is not voted on: it shows two people.
    * `match_iou` is how much a box must overlap a person's last box to
      continue them. Seated neighbours' boxes overlap by up to ~0.5, so the
      default is 0.5 with `tiles` (crowded rooms) and 0.3 otherwise.
    * A label under `min_gender_conf` / `min_age_conf`, or one whose votes
      disagree between the first and second half of the stay, is reported as
      unknown rather than guessed.
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

    # Shared cache, not the run directory: a 340 MB checkpoint should be
    # downloaded once, not once per run.
    cache = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[2] / "models" / "hf"
    if gender_model is None and gender_repo:
        from .models import GenderModel
        local = cache / gender_repo.replace("/", "__")
        if (local / "pytorch_model.bin").exists() and (local / "config.json").exists():
            # Already downloaded: load it without touching the network.
            gender_model = GenderModel(local, device=device)
            gender_model.source = gender_repo
        else:
            say(stage="downloading checkpoint", pct=0.0)
            gender_model = GenderModel.from_hub(gender_repo, local, device=device)
    elif gender_model is None and gender_ckpt:
        from .models import GenderModel
        gender_model = GenderModel(gender_ckpt, device=device)
    if age_model is None and age_bundle and attributes not in ("clip", "ensemble"):
        # (with CLIP answering age, the SVM bundle would be loaded for nothing)
        from .models import AgeModel
        if Path(age_bundle).exists():
            age_model = AgeModel(age_bundle)
        else:
            # Not a local path, so a Hub repo id, cached like the gender checkpoint.
            local = cache / str(age_bundle).replace("/", "__")
            if (local / AgeModel.HUB_FILE).exists():
                age_model = AgeModel(local / AgeModel.HUB_FILE)
                age_model.source = age_bundle
            else:
                say(stage="downloading age model", pct=0.0)
                age_model = AgeModel.from_hub(age_bundle, local)
    if attributes not in (None, "vit", "clip", "ensemble"):
        raise ValueError("attributes must be 'vit', 'clip' or 'ensemble'")
    clip_meta = None
    if attributes in ("clip", "ensemble"):
        from .clip_attrs import Average, load_clip_attributes
        both = load_clip_attributes(clip_heads or "zero-shot", device=device, say=say)
        clip_meta = {"kind": both.kind, **both.meta}
        # Gender: the fine-tuned ViT and CLIP each win on one kind of footage
        # and their average wins on both. Age: CLIP alone (the SVMs are at
        # chance on held-out identities).
        if attributes == "ensemble" and gender_model is not None:
            gender_model = Average(gender_model, both.gender)
        else:
            gender_model = both.gender
        age_model = both.age
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

    low_conf = min(det_conf, track_conf)
    floor_conf = min(low_conf, weak_conf)
    quality_model_src = None
    if isinstance(quality_model, (str, Path)):
        from ..research.pooling import QualityModel
        quality_model_src = str(quality_model)
        quality_model = QualityModel.load(quality_model)
    grid = parse_tiles(tiles)
    if match_iou is None:
        match_iou = 0.5 if grid else 0.3
    tracker = IoUTracker(iou_thresh=match_iou, max_misses=int(fps),
                         rejoin_frames=int(round(rejoin_sec * fps)),
                         new_track_conf=det_conf, rejoin_conf=low_conf,
                         tentative_conf=floor_conf if floor_conf < low_conf else None,
                         frame_size=(width, height), pooling=pooling,
                         quality_model=quality_model)
    # Quality cues are measured only when something will use them.
    need_cues = export_crops or pooling == "quality"
    if need_cues:
        from ..research.quality import crop_cues, flow_stats, frame_flow, occlusion
    crop_records = []
    prev_gray = gray = None
    vote_gap = max(stride, int(round(vote_every_sec * fps)))
    max_gap = int(round(max_vote_gap_sec * fps)) if max_vote_gap_sec else None
    skipped_votes = 0
    check_gap = max(stride, int(round(fps)))          # ask about a hunch once a second at most
    n_checks = 0
    rejected = []                                     # spots the verifier has ruled out
    if verifier is None and verifier_weights in (None, "", "none", detector_weights):
        verifier = False                              # no second opinion available
    frame_gap = max(1, int(round(frame_every_sec * fps))) if frame_every_sec else 0
    saved_frames = []
    sampled = []
    frame_area = float(width * height)
    frame_idx = 0
    t0 = time.time()
    n_classified = 0

    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and frame_idx >= max_frames):
            break
        if need_cues:
            # Before any box is drawn on the frame. The previous frame, sampled
            # or not, is what motion is measured against.
            prev_gray, gray = gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_gap and frame_idx % frame_gap == 0:
            saved_frames.append(_save_clean_frame(cv2, out_dir, frame, frame_idx, fps))

        if frame_idx % stride == 0:
            dets = detect_people(detector, frame, floor_conf, tiles=grid,
                                 overlap=tile_overlap, imgsz=imgsz)

            matched = tracker.update(dets, frame_idx, frame_area)
            sampled.append(frame_idx)

            # A hunch that keeps turning up earns a closer look.
            # A hunch that keeps turning up earns a second opinion.
            unsure = [(t, b) for t, b, _c in matched
                      if not t.confirmed and t.frames >= 8 and t.checks < max_checks
                      and frame_idx - t.last_check_frame >= check_gap]
            for t, b in unsure:                       # ...unless that spot was ruled out before
                if t.checks == 0 and any(iou(b, r) >= 0.3 for r in rejected):
                    t.checks = max_checks
            unsure = [(t, b) for t, b in unsure if t.checks < max_checks]
            agreed = {}
            if unsure and verifier is not False:
                if verifier is None:
                    say(stage="loading second detector")
                    verifier = _load_verifier(verifier_weights)
                if verifier is not False:
                    known = [b for t, b, _c in matched if t.confirmed]
                    opinions = _second_opinion(verifier, frame, verify_sizes, verify_conf, known)
                    n_checks += 1
                    for t, b in unsure:
                        t.checks += 1
                        t.last_check_frame = frame_idx
                        if any(overlap_of_smaller(o, b) >= 0.5 for o in opinions):
                            agreed[id(t)] = tracker.confirm(t)    # may fold into a lost person's track
                        elif t.checks >= max_checks:
                            rejected.append(b)
            matched = [(agreed.get(id(t), t), b, c) for t, b, c in matched]
            matched = [m for m in matched if m[0].confirmed]

            due = [(t, b, c) for t, b, c in matched
                   if t.needs_vote(frame_idx, vote_gap, max_votes, spread=vote_spread, max_gap=max_gap)]
            if max_vote_overlap is not None and due:
                # A crop that half shows a neighbour is a vote for two people.
                boxes = [b for _t, b, _c in matched]
                clear = [(t, b, c) for t, b, c in due
                         if max((iou(b, o) for o in boxes if o is not b), default=0.0) <= max_vote_overlap]
                skipped_votes += len(due) - len(clear)
                due = clear
            crops = [Image.fromarray(frame[b[1]:b[3], b[0]:b[2]][:, :, ::-1]) for _t, b, _c in due]
            g_out = gender_model.predict(crops) if (gender_model and crops) else [None] * len(crops)
            a_out = age_model.predict(crops) if (age_model and crops) else [None] * len(crops)
            measured = [(None, None, None)] * len(due)
            if need_cues and due:
                mag = scale = None
                if prev_gray is not None:
                    mag, scale = frame_flow(prev_gray, gray)
                people_boxes = [b for _t, b, _c in matched]
                measured = []
                for (_t, b, c), crop in zip(due, crops):
                    others = [o for o in people_boxes if o != b]
                    flow = flow_stats(mag, scale, b, others) if mag is not None else None
                    overlap = occlusion(b, others)
                    measured.append((crop_cues(crop, b, width, height, det_conf=c, flow=flow,
                                               overlap=overlap), flow, overlap))
            for (track, b, c), crop, gp, ap, (cues, flow, overlap) in zip(
                    due, crops, g_out, a_out, measured):
                track.vote(gender_probs=gp, age_probs=ap, frame_idx=frame_idx, cues=cues)
                if export_crops:
                    # PNG, not JPEG: research scripts re-read these crops, and the age
                    # histograms shift by up to 0.06 in probability on recompressed pixels.
                    rel = f"crops/{track.id}/{frame_idx:06d}.png"
                    (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                    crop.save(out_dir / rel)
                    crop_records.append({
                        "track": track.id, "frame": frame_idx, "t": round(frame_idx / fps, 3),
                        "box": list(b), "det_conf": round(float(c), 4),
                        "frame_w": width, "frame_h": height, "crop": rel,
                        "flow": flow, "overlap": round(float(overlap), 4), "cues": cues,
                        "pred": {"gender": _round_probs(gp), "age": _round_probs(ap)},
                    })
            n_classified += len(crops)

            if writer is not None:
                _draw(cv2, frame, matched, min_gender_conf, min_age_conf)

        elif writer is not None:
            # Only people seen on the last sampled frame, or a box lingers
            # where someone was a second ago.
            _draw(cv2, frame, [(t, t.box, 0.0) for t in tracker.active
                               if t.misses == 0 and t.confirmed], min_gender_conf, min_age_conf)

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
    clean = False
    if writer is not None:
        writer.release()
        codec = _finalize_video(out_dir / "video.mp4", codec)
        clean = _write_clean_video(video_path, out_dir / "clean.mp4", frame_idx / fps if fps else None)

    elapsed = time.time() - t0
    say(stage="aggregating", frame=frame_idx, total=n_frames, pct=99.0)

    # When the footage was *recorded* - the axis every trend is grouped by.
    # Recorder exports put the start time in the file name; failing that, the
    # file's own timestamp, which is usually when it was copied, not recorded.
    captured_from = "given"
    if not captured_at:
        captured_at = time_from_name(source_name or Path(video_path).name)
        captured_from = "file name"
    if not captured_at:
        captured_from = "file time"
        try:
            captured_at = _dt.datetime.fromtimestamp(Path(video_path).stat().st_mtime)
        except OSError:
            captured_at = _dt.datetime.now()
    if isinstance(captured_at, str):
        try:
            captured_at = _dt.datetime.fromisoformat(captured_at)
        except ValueError:
            captured_at = _dt.datetime.now()
            captured_from = "processing time"

    meta = {
        "source": source_name or Path(video_path).name,
        "captured_at": captured_at.isoformat(timespec="seconds"),
        "captured_from": captured_from,
        # Which shop and which camera: rates from different framings are not
        # comparable, so trends group by these rather than pooling everything.
        "site": site or "Store",
        "camera": camera or "Camera 1",
        "fps": round(fps, 3),
        "width": width, "height": height,
        "frames_total": frame_idx,
        "frames_sampled": len(sampled),
        "duration_sec": round(frame_idx / fps, 2) if fps else 0.0,
        "processing_sec": round(elapsed, 2),
        "processing_fps": round(frame_idx / elapsed, 2) if elapsed else 0.0,
        "crops_classified": n_classified,
        "device": device,
        "has_video": bool(writer is not None),
        "has_clean_video": clean,
        "video_codec": codec,
        "frames": saved_frames,
    }
    config = {
        "detector": detector_weights,
        "tiles": f"{grid[0]}x{grid[1]}" if grid else None,
        "imgsz": imgsz,
        "det_conf": det_conf,
        "track_conf": low_conf,
        "weak_conf": floor_conf,
        "verifier": verifier_weights if verifier not in (None, False) else None,
        "rejoin_sec": rejoin_sec,
        "frame_stride": stride,
        "vote_every_sec": vote_every_sec,
        "max_votes_per_person": max_votes,
        "vote_spread": vote_spread,
        "max_vote_gap_sec": max_vote_gap_sec,
        "max_vote_overlap": max_vote_overlap,
        "votes_skipped_overlap": skipped_votes,
        "match_iou": match_iou,
        "attributes": attributes or "vit",
        "min_gender_conf": min_gender_conf,
        "min_age_conf": min_age_conf,
        "pooling": pooling,
        "quality_model": quality_model_src,
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
    if clip_meta is not None:
        # Measured by scripts/research/attribute_study.py, which writes this
        # summary; the guide quotes it.
        summary = Path(__file__).with_name("assets") / "attribute_study.json"
        studied = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
        g_kind = "ensemble" if attributes == "ensemble" and isinstance(gender_model, Average) else "clip"
        studied = {"gender": (studied.get("gender") or {}).get(g_kind),
                   "age": (studied.get("age") or {}).get("clip")}
        model_card = {
            "gender": {"family": ("fine-tuned ViT-B/16 averaged with CLIP ViT-B/32 zero-shot"
                                  if g_kind == "ensemble" else "CLIP ViT-B/32 zero-shot"),
                       "labels": ["female", "male"], "source": getattr(gender_model, "source", None),
                       "test": studied["gender"]},
            "age": {"family": "CLIP ViT-B/32 zero-shot", "labels": ["Age16-30", "Age31-45", "Age46-60", "AgeAbove61"],
                    "source": getattr(age_model, "source", None), "test": studied["age"]},
            "clip": clip_meta,
        }
        config["gender_model"] = model_card["gender"]["source"]
        config["age_model"] = model_card["age"]["source"]

    people = tracker.close()
    # Occupancy is people present, counted once everything is known: someone
    # hidden behind a shelf for a moment is still in the room, and a person
    # confirmed late was there from their first sighting.
    counted = [t for t in people if t.frames >= min_track_frames]
    occupancy = {f: sum(1 for t in counted if t.first_frame <= f <= t.last_frame) for f in sampled}
    doc = build_analytics(people, meta, config, occupancy,
                          model_card=model_card, bin_sec=bin_sec,
                          min_frames=min_track_frames,
                          min_gender_conf=min_gender_conf, min_age_conf=min_age_conf)
    doc["kpis"]["tracks_rejoined"] = tracker.rejoined
    doc["kpis"]["hunches_checked"] = n_checks
    doc["kpis"]["hunches_dropped"] = tracker.unconfirmed
    if export_crops:
        doc["meta"]["crops_exported"] = _write_crops(out_dir, crop_records,
                                                     {t["id"] for t in doc["tracks"]})
    tb = _write_track_boxes(out_dir, counted, fps, width, height, stride, frame_idx)
    say(stage="zones", frame=frame_idx, total=n_frames, pct=99.5)
    zone_block = _zones(out_dir, tb, zones, auto_zones, zone_params)
    if zone_block is not None:
        doc["zones"] = zone_block
    # Stored, not computed in the browser: any consumer of the file gets them.
    doc["insights"] = insights(doc)
    from .guide import shop_guide
    doc["guide"] = shop_guide(doc)
    with open(out_dir / "analytics.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    say(stage="done", frame=frame_idx, total=n_frames, pct=100.0)
    return doc


_NAME_TIME = re.compile(r"(?<!\d)(20\d\d)[-_]?(\d\d)[-_]?(\d\d)[T_ -]?(\d\d)[-_:.]?(\d\d)[-_:.]?(\d\d)(?!\d)")


def time_from_name(name):
    """The recording start in a recorder's export name, or None.

    NVRs name exports like `CH021_20260914_165019_170000.avi` (start, then
    end); the first date-and-time in the name is the start."""
    for m in _NAME_TIME.finditer(str(name)):
        try:
            return _dt.datetime(*(int(v) for v in m.groups()))
        except ValueError:
            continue
    return None


def parse_tiles(tiles):
    """'2x2' / (2, 2) / None -> (cols, rows) or None."""
    if not tiles or tiles in ("none", "off", "1x1", (1, 1)):
        return None
    if isinstance(tiles, str):
        try:
            cols, rows = (int(v) for v in tiles.lower().split("x"))
        except ValueError:
            raise ValueError(f"tiles must look like '2x2', not {tiles!r}") from None
    else:
        cols, rows = (int(v) for v in tiles)
    if not (1 <= cols <= 6 and 1 <= rows <= 6) or cols * rows == 1:
        raise ValueError("tiles must be between 1x2 and 6x6")
    return cols, rows


def _boxes_of(res, ox=0, oy=0):
    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        yield (x1 + ox, y1 + oy, x2 + ox, y2 + oy), float(box.conf[0])


def merge_boxes(found, iou_thresh=0.5, inside_thresh=0.7):
    """Greedy NMS over boxes from overlapping tiles, strongest first. A box
    lying mostly inside a stronger one is dropped too: that is a person cut in
    half by a tile edge, already found whole by the tile beside it."""
    kept = []
    for box, conf in sorted(found, key=lambda bc: -bc[1]):
        area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
        clash = False
        for k, _c in kept:
            if iou(box, k) >= iou_thresh:
                clash = True
                break
            iw = max(0.0, min(box[2], k[2]) - max(box[0], k[0]))
            ih = max(0.0, min(box[3], k[3]) - max(box[1], k[1]))
            if iw * ih / area >= inside_thresh:
                clash = True
                break
        if not clash:
            kept.append((box, conf))
    return kept


def detect_people(detector, frame, conf, tiles=None, overlap=0.2, imgsz=None):
    """Person boxes [((x1, y1, x2, y2), conf)] in whole-frame pixels.

    With `tiles`, the detector also looks at overlapping tiles and the boxes
    are merged (see `merge_boxes`)."""
    height, width = frame.shape[:2]
    kw = {"classes": [0], "conf": conf, "verbose": False}
    if imgsz:
        kw["imgsz"] = imgsz
    found = list(_boxes_of(detector(frame, **kw)[0]))
    if tiles:
        cols, rows = tiles
        tw = min(width, int(round(width / cols * (1 + overlap))))
        th = min(height, int(round(height / rows * (1 + overlap))))
        xs = [round(i * (width - tw) / max(1, cols - 1)) for i in range(cols)]
        ys = [round(j * (height - th) / max(1, rows - 1)) for j in range(rows)]
        for y in ys:
            for x in xs:
                found.extend(_boxes_of(detector(frame[y:y + th, x:x + tw], **kw)[0], x, y))
        found = merge_boxes(found)
    dets = []
    for (x1, y1, x2, y2), c in found:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(x2)), min(height, int(y2))
        if x2 - x1 < 8 or y2 - y1 < 16:
            continue
        dets.append(((x1, y1, x2, y2), c))
    return dets


def _save_clean_frame(cv2, out_dir, frame, frame_idx, fps, max_width=1280):
    """A frame as the camera saw it, at most `max_width` wide. Returns its entry."""
    (out_dir / "frames").mkdir(exist_ok=True)
    h, w = frame.shape[:2]
    k = min(1.0, max_width / float(w))
    img = cv2.resize(frame, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame
    name = f"{frame_idx:06d}.jpg"
    cv2.imwrite(str(out_dir / "frames" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"frame": frame_idx, "t": round(frame_idx / fps, 2), "file": f"frames/{name}"}


def _write_track_boxes(out_dir, tracks, fps, width, height, stride, frames_total):
    """Every observed box of every counted person, for zones.py."""
    tb = {
        "schema": 1, "fps": round(fps, 3), "width": width, "height": height,
        "stride": stride, "frames_total": frames_total,
        "tracks": {str(t.id): [list(o) for o in sorted(t.box_log)] for t in tracks},
    }
    with open(out_dir / "track_boxes.json", "w", encoding="utf-8") as f:
        json.dump(tb, f, separators=(",", ":"))
    return tb


def _zones(out_dir, tb, zones, auto_zones, zone_params):
    """Zone occupancy for this run, or None when there is nothing to measure."""
    from .zones import analyse, normalise_zones, propose_zones, write_json_atomic
    if isinstance(zones, (str, Path)):
        with open(zones, encoding="utf-8") as f:
            zones = json.load(f)
    if zones is None and auto_zones and tb["tracks"]:
        zones = propose_zones(tb, zone_params)
    if not zones or not zones.get("zones"):
        return None
    zones = normalise_zones(zones, tb["width"], tb["height"])
    write_json_atomic(out_dir / "zones.json", zones)
    return analyse(tb, zones, zone_params)


def _round_probs(probs):
    return None if probs is None else {k: round(float(v), 5) for k, v in probs.items()}


def _write_crops(out_dir, records, kept_ids):
    """Write crops.jsonl for the people the analytics kept, and drop the rest.

    A track seen in too few frames is dropped as noise after it may already
    have been classified; its crops go too, so the file holds exactly the crops
    behind the `classified_frames` of the people in analytics.json.
    """
    keep = [r for r in records if r["track"] in kept_ids]
    for tid in {r["track"] for r in records} - set(kept_ids):
        shutil.rmtree(out_dir / "crops" / str(tid), ignore_errors=True)
    with open(out_dir / "crops.jsonl", "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    return len(keep)


def _load_verifier(weights):
    """The larger detector, or False if it cannot be had (no network on first
    use, say). Without it hunches are simply never confirmed."""
    try:
        from ultralytics import YOLO
        return YOLO(weights)
    except Exception as exc:                          # noqa: BLE001 - optional component
        print(f"second detector unavailable ({type(exc).__name__}); carrying on without it")
        return False


def _second_opinion(verifier, frame, sizes, conf, known=()):
    """Person boxes a larger detector finds in this frame.

    The fast detector scores a person behind glass, or small and far away, at
    0.1 - the same as a poster or a stack of boxes. A larger model tells them
    apart. It is run at more than one input size because no single size finds
    everyone: the small one is better on dark, low-contrast people, the large
    one on people who are tiny in the frame. Boxes lying on someone already
    confirmed are left out, so a hunch next to a real person cannot borrow
    that person's box.
    """
    found = []
    for size in sizes:
        res = verifier(frame, classes=[0], conf=conf, imgsz=size, verbose=False)[0]
        for b in res.boxes:
            box = tuple(b.xyxy[0].tolist())
            if not any(overlap_of_smaller(box, k) >= 0.5 for k in known):
                found.append(box)
    return found


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


# Above this the file is too heavy to scrub in a browser; OpenCV's H.264
# writer on Windows lands near 40 Mbit/s for 1080p, ffmpeg at CRF 23 near 3.
MAX_BITRATE = 8_000_000


def _probe_codec(path):
    """Ask ffprobe what actually landed in the file: (codec, bit/s) or (None, None)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None, None
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=codec_name,bit_rate", "-of", "csv=p=0", str(path)]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None, None
    codec, _, rate = out.partition(",")
    try:
        rate = int(rate)
    except ValueError:
        rate = None
    return codec or None, rate


def _finalize_video(path, requested):
    """Make sure the MP4 is browser-playable, and report what it actually is.

    `isOpened()` lies on builds whose H.264 encoder fails to load at write time,
    so trust ffprobe over the fourcc we asked for. An H.264 file that is far
    too heavy to stream is re-encoded as well, when ffmpeg is around.
    """
    actual, rate = _probe_codec(path)
    if actual == "h264" and (rate is None or rate <= MAX_BITRATE or not shutil.which("ffmpeg")):
        return "avc1"
    if shutil.which("ffmpeg"):
        return _to_h264(path) or ("avc1" if actual == "h264" else actual) or requested
    return actual or requested


def _to_h264(path):
    """Transcode in place with ffmpeg if available. Returns the resulting codec."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    tmp = path.with_suffix(".h264.mp4")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError):
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(path)
    return "avc1"


def _write_clean_video(src, path, duration=None, max_width=1280):
    """A light copy of the footage with nothing drawn on it, for the zones view,
    where burned-in boxes would fight the zone overlay. Needs ffmpeg; returns
    whether the file was written."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    tmp = path.with_suffix(".part.mp4")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vf", f"scale='min({max_width},iw)':-2", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError):
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(path)
    return True


def _draw(cv2, frame, matched, min_gender_conf=0.0, min_age_conf=0.0):
    for track, box, _conf in matched:
        # What this box shows now: the latest votes, not the whole stay's.
        # Under the thresholds the box says "?" rather than show a guess.
        label, conf = track.recent(0)
        age, age_conf = track.recent(1)
        unsure = label is not None and conf < min_gender_conf
        if unsure:
            label = None
        if age is not None and age_conf < min_age_conf:
            age = None
        color = BOX_BGR.get(label, BOX_BGR[None])
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        parts = [f"#{track.id}"]
        if unsure:
            parts.append("?")
        if label:
            # A question mark while the votes are still close: the label can flip.
            parts.append(f"{label}{'?' if conf < 0.7 else ''} {conf:.2f}")
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
    p.add_argument("--age-bundle", default=None,
                   help="HuggingFace repo id, or a local bundle file or directory")
    p.add_argument("--detector", default="yolov8n.pt")
    p.add_argument("--det-conf", type=float, default=0.4)
    p.add_argument("--track-conf", type=float, default=0.25,
                   help="weaker detections may extend a tracked person, never start one")
    p.add_argument("--rejoin-sec", type=float, default=8.0,
                   help="how long a lost person can be picked up again; 0 turns it off")
    p.add_argument("--weak-conf", type=float, default=0.10,
                   help="weakest detection worth a second opinion; set it equal to "
                        "--track-conf to turn hunches off")
    p.add_argument("--verifier", default="yolov8m.pt",
                   help="larger detector consulted about weak detections; 'none' to do without")
    p.add_argument("--stride", type=int, default=2, help="classify every Nth frame")
    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--min-track-frames", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--vote-every-sec", type=float, default=0.4,
                   help="classify each tracked person at most this often")
    p.add_argument("--max-votes", type=int, default=24,
                   help="stop classifying a person after this many votes")
    p.add_argument("--no-vote-spread", action="store_true",
                   help="vote at a fixed pace instead of spreading votes over the whole stay")
    p.add_argument("--max-vote-gap-sec", type=float, default=45.0,
                   help="longest wait between two votes on one person when votes are spread")
    p.add_argument("--max-vote-overlap", type=float, default=0.3,
                   help="skip a vote when the crop overlaps another person's box by more (IoU)")
    p.add_argument("--match-iou", type=float, default=None,
                   help="overlap a box needs to continue a person (default 0.5 with --tiles, else 0.3)")
    p.add_argument("--attributes", choices=["ensemble", "clip", "vit"], default="ensemble",
                   help="ensemble (default): gender from the ViT (--gender-repo) averaged with CLIP "
                        "zero-shot, age from CLIP; clip: CLIP zero-shot for both; vit: the ViT and "
                        "the age SVMs (--age-bundle)")
    p.add_argument("--clip-heads", default=None,
                   help="heads fitted by scripts/research/clip_probe.py instead of zero-shot prompts")
    p.add_argument("--min-gender-conf", type=float, default=0.6,
                   help="report a person's gender as unknown below this confidence")
    p.add_argument("--min-age-conf", type=float, default=0.4,
                   help="report a person's age group as unknown below this confidence")
    p.add_argument("--site", default=None, help="shop name, e.g. 'MG Road store'")
    p.add_argument("--camera", default=None, help="camera name, e.g. 'Entrance'")
    p.add_argument("--captured-at", default=None,
                   help="when the footage was recorded, ISO format "
                        "(default: the video file's timestamp)")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-video", action="store_true", help="analytics only, skip the MP4")
    p.add_argument("--export-crops", action="store_true",
                   help="save every classified crop and crops.jsonl, for labelling and research")
    p.add_argument("--pooling", default="majority",
                   choices=["majority", "mean", "confidence", "quality"],
                   help="how a person's votes become one label")
    p.add_argument("--quality-model", default=None,
                   help="fitted quality model JSON, needed for --pooling quality")
    p.add_argument("--tiles", default=None,
                   help="also detect on overlapping tiles, e.g. 2x2: for crowded, seated scenes")
    p.add_argument("--imgsz", type=int, default=None, help="detector input size (default: the model's)")
    p.add_argument("--zones", default=None,
                   help="zones JSON (tables, seats, queues); default: propose seats automatically")
    p.add_argument("--no-auto-zones", action="store_true",
                   help="without --zones, do not propose seats")
    a = p.parse_args()

    try:
        doc = process_video(
            a.video, a.out, gender_ckpt=a.gender_ckpt, gender_repo=a.gender_repo,
            age_bundle=a.age_bundle, detector_weights=a.detector, det_conf=a.det_conf,
            stride=a.stride, device=a.device, write_video=not a.no_video,
            bin_sec=a.bin_sec, min_track_frames=a.min_track_frames,
            max_frames=a.max_frames, vote_every_sec=a.vote_every_sec,
            max_votes=a.max_votes, captured_at=a.captured_at,
            track_conf=a.track_conf, rejoin_sec=a.rejoin_sec, weak_conf=a.weak_conf,
            verifier_weights=a.verifier,
            site=a.site, camera=a.camera, export_crops=a.export_crops,
            pooling=a.pooling, quality_model=a.quality_model,
            tiles=a.tiles, imgsz=a.imgsz, zones=a.zones, auto_zones=not a.no_auto_zones,
            attributes=a.attributes, clip_heads=a.clip_heads, match_iou=a.match_iou,
            vote_spread=not a.no_vote_spread, max_vote_gap_sec=a.max_vote_gap_sec,
            max_vote_overlap=a.max_vote_overlap, min_gender_conf=a.min_gender_conf,
            min_age_conf=a.min_age_conf,
            progress=lambda **kw: print(f"  {kw.get('stage')}: {kw.get('pct')}%", end="\r"),
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    k = doc["kpis"]
    print(f"\n{k['visitors']} visitors · peak occupancy {k['peak_occupancy']} · "
          f"avg dwell {k['avg_dwell_sec']}s · {k['male_pct']}% male / {k['female_pct']}% female")
    z = doc.get("zones")
    if z:
        s = z["summary"]
        print(f"{s['zones']} zones ({z['source']}) · mean utilisation {100 * s['mean_utilization']:.0f}% · "
              f"peak {s['peak_occupied']} occupied")
    print(f"written to {a.out}/analytics.json")


if __name__ == "__main__":
    main()
