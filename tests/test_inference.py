"""Unit tests for the inference-side pipeline: tracking and analytics.

Model wrappers are not covered here - they need weights. These cover the logic
that turns detections into the numbers the dashboard shows.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.analytics import build_analytics  # noqa: E402
from src.inference.tracker import IoUTracker, Track, iou  # noqa: E402

FRAME_AREA = 1280 * 720
META = {"source": "t.mp4", "fps": 10.0, "width": 1280, "height": 720,
        "frames_total": 100, "duration_sec": 10.0}
CONFIG = {"detector": "yolov8n.pt", "det_conf": 0.4, "frame_stride": 1}


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert round(iou((0, 0, 10, 10), (5, 0, 15, 10)), 3) == 0.333


def test_tracker_keeps_one_identity_across_frames():
    t = IoUTracker(iou_thresh=0.3, max_misses=2)
    for i in range(5):
        t.update([((i, 0, 100 + i, 200), 0.9)], i, FRAME_AREA)
    tracks = t.close()
    assert len(tracks) == 1
    assert tracks[0].frames == 5
    assert tracks[0].first_frame == 0 and tracks[0].last_frame == 4


def test_tracker_starts_a_new_track_for_a_distant_box():
    t = IoUTracker()
    t.update([((0, 0, 50, 100), 0.9)], 0, FRAME_AREA)
    t.update([((600, 0, 650, 100), 0.9)], 1, FRAME_AREA)
    assert len(t.close()) == 2


def test_tracker_drops_a_track_after_max_misses():
    t = IoUTracker(max_misses=1)
    t.update([((0, 0, 50, 100), 0.9)], 0, FRAME_AREA)
    for f in (1, 2, 3):
        t.update([], f, FRAME_AREA)
    assert len(t.active) == 0 and len(t.finished) == 1


def test_votes_beat_a_single_bad_frame():
    t = Track(id=1, box=(0, 0, 10, 10), first_frame=0, last_frame=0)
    t.vote(gender_probs={"male": 0.9, "female": 0.1})
    t.vote(gender_probs={"male": 0.8, "female": 0.2})
    t.vote(gender_probs={"male": 0.2, "female": 0.8})
    label, conf = t.gender()
    assert label == "male"
    assert 0.5 < conf < 1.0


def _make_track(tid, f0, f1, gender, age, x=640):
    t = Track(id=tid, box=(x, 300, x + 60, 500), first_frame=f0, last_frame=f1)
    t.frames = f1 - f0 + 1
    t.det_conf_sum = 0.9 * t.frames
    t.path = [(f, float(x + 30), 400.0, 0.01) for f in range(f0, f1 + 1)]
    for _ in range(3):
        t.vote(gender_probs={gender: 0.9, "male" if gender == "female" else "female": 0.1},
               age_probs={age: 0.7, "Age46-60": 0.3})
    return t


def test_build_analytics_shape_and_kpis():
    tracks = [
        _make_track(1, 0, 20, "female", "Age16-30"),
        _make_track(2, 10, 90, "male", "Age31-45", x=100),
        _make_track(3, 0, 1, "male", "Age16-30"),   # too short - dropped
    ]
    occ = {f: 2 for f in range(0, 100)}
    doc = build_analytics(tracks, META, CONFIG, occ, bin_sec=5.0, min_frames=3)

    assert doc["kpis"]["visitors"] == 2
    assert doc["kpis"]["tracks_dropped"] == 1
    assert doc["distributions"]["gender"] == {"female": 1, "male": 1}
    assert doc["distributions"]["age_by_gender"]["male"]["Age31-45"] == 1
    assert len(doc["tracks"]) == 2
    assert doc["timeline"]["bins"][0]["entries"] == 2   # both enter inside 0-5s
    assert doc["kpis"]["peak_occupancy"] == 2
    # track 2 spans 8 seconds at 10fps
    assert doc["tracks"][1]["dwell_sec"] == 8.0
    assert doc["tracks"][1]["tier"] == "Glancing"


def test_build_analytics_handles_an_empty_run():
    doc = build_analytics([], META, CONFIG, {}, bin_sec=5.0)
    assert doc["kpis"]["visitors"] == 0
    assert doc["kpis"]["dominant_age"] is None
    assert doc["heatmap"]["grid"]


def test_zones_split_the_frame_into_thirds():
    left = _make_track(1, 0, 20, "female", "Age16-30", x=10)
    right = _make_track(2, 0, 20, "male", "Age16-30", x=1150)
    doc = build_analytics([left, right], META, CONFIG, {0: 2}, min_frames=3)
    zones = {t["id"]: t["zone"] for t in doc["tracks"]}
    assert zones[1] == "Left" and zones[2] == "Right"
