"""What the guide's claims rest on, taken from the recording itself.

The guide (guide.py) says which moment, seat or people each claim is about;
this module says what was there. For a moment: every person the system counted
then, drawn where it saw them, and every seat with its state. A claim like
"45 people at the busiest" then comes with the frame it was counted on, and
anyone can count the boxes.

A person counts as in view from their first sighting to their last, so at any
moment some counted people are not detected in that very frame (hidden behind
someone, or missed). They are returned with their last box and how long ago it
was seen, never dropped and never invented, so the boxes always add up to the
number the dashboard reports.

    tb = zones.load_track_boxes(run_dir)
    moment(doc, tb, 443.8)          # people and seats at that moment
    handoffs(tb)                    # tracks that ended and restarted in place
    portraits(rows, [12, 40])       # the clearest crop of each person
    still(video, 443.8, 960, cache) # the frame itself, as JPEG bytes
    snap(video, 443.8, box, 1920, cache)  # one person, cut from that frame
"""
from __future__ import annotations

import bisect
from pathlib import Path

STILL_WIDTHS = (480, 960, 1600)


def occupancy_peak(tracks):
    """(t, n): the first moment the most people were in view, counting a
    person from their first to their last sighting (as analytics.py does)."""
    events = []
    for t in tracks:
        events.append((t["first_sec"], 0))    # arrivals sort before departures at the same instant,
        events.append((t["last_sec"], 1))     # so someone leaving as another arrives overlaps them
    events.sort()
    n = best = 0
    at = 0.0
    for sec, kind in events:
        n += 1 if kind == 0 else -1
        if n > best:
            best, at = n, sec
    return at, best


def taken_at(zone, t, eps=1e-6):
    """The session holding a zone at time t, or None."""
    for s in zone.get("sessions", []):
        if s["start"] - eps <= t <= s["end"] + eps:
            return s
    return None


def _index(tb):
    """Per track: sorted frame numbers and boxes, for bisecting."""
    cache = tb.get("_index")
    if cache is None:
        cache = {}
        for tid, obs in tb["tracks"].items():
            obs = sorted(obs)
            cache[int(tid)] = ([o[0] for o in obs], obs)
        tb["_index"] = cache
    return cache


def moment(doc, tb, t):
    """Everyone counted and every zone's state at time t (seconds).

    people: {track, box [x1, y1, x2, y2] in pixels, seen (detected in this
    frame), ago (seconds since last seen, 0 when seen), gender, age}.
    zones: {id, name, kind, polygon (0-1 of the frame), taken, since, people}.
    """
    fps = float(tb.get("fps") or doc["meta"].get("fps") or 25.0)
    stride = int(tb.get("stride") or 1)
    frame = int(round(t * fps / stride)) * stride
    info = {x["id"]: x for x in doc.get("tracks", [])}
    people = []
    for tid, (frames, obs) in _index(tb).items():
        if not frames or not frames[0] <= frame <= frames[-1]:
            continue
        i = bisect.bisect_right(frames, frame) - 1
        f, x1, y1, x2, y2 = obs[i][:5]
        p = info.get(tid, {})
        people.append({"track": tid, "box": [round(x1), round(y1), round(x2), round(y2)],
                       "seen": f == frame, "ago": round((frame - f) / fps, 1),
                       "gender": p.get("gender"), "age": p.get("age")})
    people.sort(key=lambda p: (not p["seen"], p["track"]))
    zones = []
    for z in (doc.get("zones") or {}).get("zones", []):
        s = taken_at(z, frame / fps)
        zones.append({"id": z["id"], "name": z["name"], "kind": z["kind"], "polygon": z["polygon"],
                      "taken": s is not None, "since": s["start"] if s else None,
                      "people": s["people"] if s else []})
    return {"t": round(frame / fps, 2), "frame": frame,
            "width": tb.get("width") or doc["meta"].get("width"),
            "height": tb.get("height") or doc["meta"].get("height"),
            "people": people, "zones": zones,
            "seen": sum(p["seen"] for p in people), "hidden": sum(not p["seen"] for p in people)}


def handoffs(tb, max_gap_sec=30.0, min_overlap=0.5):
    """Places where one track ended and a new one began moments later in the
    same spot: most likely one person the tracker lost and counted again.

    A start is matched to the ended track whose last box overlaps its first
    box most (intersection over union, at least `min_overlap`), within
    `max_gap_sec`; each ended track is matched at most once. On the labelled
    classroom footage the same person's boxes overlap this much in 98% of
    samples and neighbours' in 0.1% (paper/results/identity_gap.json).
    """
    fps = float(tb.get("fps") or 25.0)
    tracks = {int(k): sorted(v) for k, v in tb["tracks"].items() if v}
    ends = sorted((obs[-1][0], tid) for tid, obs in tracks.items())
    end_frames = [f for f, _ in ends]
    used, out = set(), []
    for tid, obs in sorted(tracks.items(), key=lambda kv: kv[1][0][0]):
        start = obs[0]
        lo = bisect.bisect_left(end_frames, start[0] - max_gap_sec * fps)
        hi = bisect.bisect_left(end_frames, start[0])
        best = None
        for f, other in ends[lo:hi]:
            if other == tid or other in used:
                continue
            j = _iou(tracks[other][-1][1:5], start[1:5])
            if j >= min_overlap and (best is None or j > best[0]):
                best = (j, other, f)
        if best:
            j, other, f = best
            used.add(other)
            out.append({"a": other, "b": tid, "a_t": round(f / fps, 2), "b_t": round(start[0] / fps, 2),
                        "gap": round((start[0] - f) / fps, 1), "overlap": round(j, 2),
                        "a_box": [round(v) for v in tracks[other][-1][1:5]],
                        "b_box": [round(v) for v in start[1:5]]})
    return out


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def portraits(rows, track_ids):
    """The clearest crop of each person: big, sharp, least covered.

    rows: crops.jsonl records. Returns {track: {"crop", "t"}}.
    """
    want = set(track_ids)
    best = {}
    for r in rows:
        if r["track"] not in want:
            continue
        c = r.get("cues") or {}
        score = ((c.get("size") or 0.0) * (r.get("det_conf") or 0.0)
                 * (1.0 - (c.get("occlusion") or 0.0)) * (1.0 - (c.get("truncated") or 0.0)))
        if r["track"] not in best or score > best[r["track"]][0]:
            best[r["track"]] = (score, {"crop": r["crop"], "t": r["t"]})
    return {k: v[1] for k, v in best.items()}


def _frame(video, t):
    """(frame number, BGR image) at t seconds of a video."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame = max(0, min(int(round(t * fps)), n - 1 if n else 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f"no frame at {t:.2f} s")
    return frame, img


def snap(video, t, box, src_width, cache_dir, height=220):
    """One person at t seconds: the frame cut to their box (given in the
    coordinates of a `src_width`-wide frame) with a margin, as JPEG bytes."""
    import cv2
    x1, y1, x2, y2 = box
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{t:.2f}_{x1}_{y1}_{x2}_{y2}.jpg"
    if path.exists():
        return path.read_bytes()
    _, img = _frame(video, t)
    k = img.shape[1] / float(src_width)
    w, h = (x2 - x1) * k, (y2 - y1) * k
    m = 0.25 * max(w, h)
    cx1, cy1 = int(max(0, x1 * k - m)), int(max(0, y1 * k - m))
    cx2, cy2 = int(min(img.shape[1], x2 * k + m)), int(min(img.shape[0], y2 * k + m))
    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError("box outside the frame")
    crop = img[cy1:cy2, cx1:cx2]
    crop = cv2.resize(crop, (max(1, int(round(crop.shape[1] * height / crop.shape[0]))), height),
                      interpolation=cv2.INTER_CUBIC if crop.shape[0] < height else cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    data = buf.tobytes()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def still(video, t, width, cache_dir):
    """The frame at t seconds, `width` pixels wide, as JPEG bytes. Frames are
    cached on disk: a guide asks for the same few again and again."""
    import cv2
    if width not in STILL_WIDTHS:
        raise ValueError(f"width must be one of {STILL_WIDTHS}")
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{t:.2f}_{width}.jpg"
    if path.exists():
        return path.read_bytes()
    _, img = _frame(video, t)
    h = int(round(img.shape[0] * width / img.shape[1]))
    img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 84])
    data = buf.tobytes()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data
