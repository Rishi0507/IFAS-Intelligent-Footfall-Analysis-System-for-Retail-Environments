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

# ------------------------------------------------- re-joining lost people
SIZE = (1280, 720)
MID = (600, 300, 660, 460)          # a person in the middle of the frame


def _rejoining(**kw):
    return IoUTracker(iou_thresh=0.3, max_misses=2, rejoin_frames=40,
                      new_track_conf=0.4, frame_size=SIZE, **kw)


def test_person_hidden_mid_frame_is_the_same_person_when_they_reappear():
    t = _rejoining()
    for i in range(4):
        t.update([(MID, 0.9)], i, FRAME_AREA)
    for i in range(4, 20):                        # hidden far longer than max_misses
        t.update([], i, FRAME_AREA)
    assert len(t.active) == 0 and len(t.finished) == 1
    moved = (630, 290, 690, 450)                  # came out half a step to the side
    t.update([(moved, 0.9)], 20, FRAME_AREA)
    tracks = t.close()
    assert len(tracks) == 1
    assert tracks[0].rejoins == 1 and t.rejoined == 1
    assert tracks[0].first_frame == 0 and tracks[0].last_frame == 20


def test_without_rejoining_the_same_person_counts_twice():
    t = IoUTracker(iou_thresh=0.3, max_misses=2)
    for i in range(4):
        t.update([(MID, 0.9)], i, FRAME_AREA)
    for i in range(4, 20):
        t.update([], i, FRAME_AREA)
    t.update([((630, 290, 690, 450), 0.9)], 20, FRAME_AREA)
    assert len(t.close()) == 2


def test_rejoining_gives_up_after_the_window():
    t = _rejoining()
    t.update([(MID, 0.9)], 0, FRAME_AREA)
    t.update([(MID, 0.9)], 1, FRAME_AREA)
    for i in range(2, 60):
        t.update([], i, FRAME_AREA)
    t.update([(MID, 0.9)], 60, FRAME_AREA)        # 59 frames later, window is 40
    assert len(t.close()) == 2


def test_someone_far_away_is_a_new_person_not_a_rejoin():
    t = _rejoining()
    for i in range(3):
        t.update([(MID, 0.9)], i, FRAME_AREA)
    for i in range(3, 10):
        t.update([], i, FRAME_AREA)
    t.update([((900, 300, 960, 460), 0.9)], 10, FRAME_AREA)   # two body heights away
    assert len(t.close()) == 2


def test_leaving_at_the_edge_and_someone_else_arriving_stay_two_people():
    t = _rejoining()
    # Walks out through the left edge...
    for i, x in enumerate((110, 90, 70, 50, 30, 10)):
        t.update([((x, 300, x + 60, 460), 0.9)], i, FRAME_AREA)
    assert len(t.active) == 1                     # one walker so far
    for i in range(6, 12):
        t.update([], i, FRAME_AREA)
    # ...and a box shows up in that same doorway a moment later.
    t.update([((10, 300, 70, 460), 0.9)], 12, FRAME_AREA)
    assert len(t.close()) == 2


def test_someone_sitting_at_the_edge_is_rejoined_in_place():
    t = _rejoining()
    desk = (0, 300, 50, 440)
    for i in range(5):
        t.update([(desk, 0.9)], i, FRAME_AREA)
    for i in range(5, 15):
        t.update([], i, FRAME_AREA)
    t.update([((2, 302, 52, 442), 0.9)], 15, FRAME_AREA)
    tracks = t.close()
    assert len(tracks) == 1 and tracks[0].rejoins == 1


def test_weak_detection_extends_a_person_but_never_starts_one():
    t = _rejoining()
    t.update([(MID, 0.3)], 0, FRAME_AREA)         # weak and unknown: ignored
    assert len(t.active) == 0
    t.update([(MID, 0.9)], 1, FRAME_AREA)         # confident: a person
    t.update([(MID, 0.3)], 2, FRAME_AREA)         # weak, but it is that person
    tracks = t.close()
    assert len(tracks) == 1 and tracks[0].frames == 2


def test_two_boxes_cannot_both_rejoin_one_lost_person():
    t = _rejoining()
    for i in range(3):
        t.update([(MID, 0.9)], i, FRAME_AREA)
    for i in range(3, 10):
        t.update([], i, FRAME_AREA)
    t.update([((610, 300, 670, 460), 0.9), ((680, 300, 740, 460), 0.8)], 10, FRAME_AREA)
    assert len(t.close()) == 2                    # one rejoins, the other is new

def test_second_box_on_a_tracked_person_is_not_a_second_person():
    t = _rejoining()
    t.update([(MID, 0.9)], 0, FRAME_AREA)
    torso = (605, 300, 655, 380)                  # the same person, upper half only
    t.update([(MID, 0.9), (torso, 0.6)], 1, FRAME_AREA)
    t.update([(MID, 0.9), (torso, 0.6)], 2, FRAME_AREA)
    assert len(t.close()) == 1


def test_two_people_standing_side_by_side_are_still_two():
    t = _rejoining()
    beside = (650, 300, 710, 460)                 # shoulders overlapping a little
    for i in range(3):
        t.update([(MID, 0.9), (beside, 0.9)], i, FRAME_AREA)
    assert len(t.close()) == 2

# ------------------------------------------------------ tentative tracks
def _hunching(**kw):
    return IoUTracker(iou_thresh=0.3, max_misses=2, rejoin_frames=40, new_track_conf=0.4,
                      rejoin_conf=0.25, tentative_conf=0.1, frame_size=SIZE, **kw)


def test_weak_boxes_alone_never_become_a_person():
    t = _hunching()
    for i in range(10):
        matched = t.update([(MID, 0.15)], i, FRAME_AREA)
        assert len(matched) == 1 and matched[0][0].confirmed is False
    assert t.close() == []
    assert t.unconfirmed == 1


def test_a_clear_sighting_confirms_a_hunch_and_keeps_its_history():
    t = _hunching()
    for i in range(6):
        t.update([(MID, 0.15)], i, FRAME_AREA)
    t.update([(MID, 0.8)], 6, FRAME_AREA)
    tracks = t.close()
    assert len(tracks) == 1
    assert tracks[0].first_frame == 0 and tracks[0].frames == 7     # stay starts at the first weak sighting


def test_caller_can_confirm_a_hunch():
    t = _hunching()
    for i in range(5):
        matched = t.update([(MID, 0.15)], i, FRAME_AREA)
    assert t.confirm(matched[0][0]) is matched[0][0]
    assert len(t.close()) == 1 and t.unconfirmed == 0


def test_confirmed_hunch_folds_into_the_person_lost_at_that_spot():
    t = _hunching()
    for i in range(4):
        t.update([(MID, 0.9)], i, FRAME_AREA)                       # a known person...
    for i in range(4, 12):
        t.update([], i, FRAME_AREA)                                 # ...lost for a while
    nearby = (650, 300, 710, 460)
    for i in range(12, 18):
        matched = t.update([(nearby, 0.15)], i, FRAME_AREA)         # weak boxes a step away: a hunch
    assert matched[0][0].confirmed is False
    survivor = t.confirm(matched[0][0])
    assert survivor.id == 1 and survivor.rejoins == 1
    tracks = t.close()
    assert len(tracks) == 1
    assert tracks[0].first_frame == 0 and tracks[0].last_frame == 17


def test_weak_box_in_the_same_place_picks_a_seated_person_back_up():
    t = _hunching()
    for i in range(5):
        t.update([(MID, 0.9)], i, FRAME_AREA)
    for i in range(5, 15):
        t.update([], i, FRAME_AREA)
    matched = t.update([(MID, 0.12)], 15, FRAME_AREA)               # weak, but exactly where they sat
    assert matched[0][0].id == 1 and matched[0][0].rejoins == 1
    assert len(t.close()) == 1


def test_weak_box_a_step_away_does_not_revive_anyone():
    t = _hunching()
    for i in range(5):
        t.update([(MID, 0.9)], i, FRAME_AREA)
    for i in range(5, 15):
        t.update([], i, FRAME_AREA)
    matched = t.update([((690, 300, 750, 460), 0.12)], 15, FRAME_AREA)
    assert matched[0][0].id != 1 and matched[0][0].confirmed is False
    assert len(t.close()) == 1                                      # only the original person


def test_a_hunch_cannot_hide_a_real_person_standing_on_it():
    t = _hunching()
    t.update([(MID, 0.15)], 0, FRAME_AREA)                          # hunch
    matched = t.update([(MID, 0.15), ((605, 300, 665, 460), 0.9)], 1, FRAME_AREA)
    assert any(m[0].confirmed for m in matched)

def test_long_trail_keeps_its_end_when_thinned():
    from src.inference.analytics import _thin
    for n in (10, 60, 61, 119, 134, 1000):
        path = [(i, float(i), 0.0, 0.01) for i in range(n)]
        thin = _thin(path)
        assert len(thin) == min(n, 60)
        assert thin[0] == path[0] and thin[-1] == path[-1]
        assert [p[0] for p in thin] == sorted({p[0] for p in thin})   # in order, no repeats
