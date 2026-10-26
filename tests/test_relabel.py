"""Labelling a run again from whole-stay votes (src/inference/relabel.py)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.analytics import build_analytics  # noqa: E402
from src.inference.relabel import rebuild_tracks, schedule, window_label  # noqa: E402

FPS = 25.0


def tb_doc(tracks):
    return {"fps": FPS, "stride": 5, "width": 640, "height": 360,
            "tracks": {str(k): [[f, *box, 0.8] for f, box in v] for k, v in tracks.items()}}


def test_votes_cover_the_whole_stay_and_skip_crops_showing_two_people():
    alone = [(f, (10, 10, 60, 120)) for f in range(0, 2500, 5)]               # 100 s
    a = [(f, (300, 10, 350, 120)) for f in range(0, 500, 5)]
    b = [(f, (310, 10, 360, 120) if f < 250 else (500, 10, 550, 120)) for f in range(0, 500, 5)]
    due, skipped = schedule(tb_doc({1: alone, 2: a, 3: b}), every_sec=2.0, max_overlap=0.3)
    one = sorted(f for f, v in due.items() for t, _ in v if t == 1)
    assert len(one) == 50 and one[0] == 0 and one[-1] == 2450                 # every 2 s, start to end
    two = sorted(f for f, v in due.items() for t, _ in v if t == 2)
    assert two and min(two) >= 250                                            # only once 3 moved away
    assert skipped > 0


def test_a_box_is_labelled_from_the_votes_around_that_moment():
    man, woman = {"male": 0.9, "female": 0.1}, {"male": 0.2, "female": 0.8}
    votes = [(float(s), man, None) for s in range(0, 60, 2)] + [(float(s), woman, None) for s in range(60, 120, 2)]
    assert window_label(votes, 0, 20.0, 8.0) == ("male", pytest.approx(0.9))
    assert window_label(votes, 0, 100.0, 8.0) == ("female", pytest.approx(0.8))
    label, conf = window_label(votes, 0, 60.0, 8.0)                            # the hand-over: unsure
    assert conf < 0.6
    assert window_label(votes, 1, 20.0, 8.0) == (None, 0.0)                    # no age votes
    assert window_label([(500.0, man, None)], 0, 20.0, 8.0) == ("male", 0.9)    # nearest when none close
    assert window_label([], 0, 1.0, 8.0) == (None, 0.0)


def test_rebuilt_tracks_give_the_same_analytics_and_new_labels():
    box = (100, 50, 160, 200)
    tb = tb_doc({7: [(f, box) for f in range(0, 1000, 5)]})
    doc = {"meta": {"fps": FPS, "width": 640, "height": 360, "frames_total": 1000, "duration_sec": 40.0},
           "tracks": [{"id": 7, "rejoins": 2}]}
    man = {"male": 0.95, "female": 0.05}
    young = {"Age16-30": 0.7, "Age31-45": 0.1, "Age46-60": 0.1, "AgeAbove61": 0.1}
    tracks = rebuild_tracks(doc, tb, {7: [(float(s), man, young) for s in range(0, 40, 2)]})
    t = tracks[0]
    assert (t.first_frame, t.last_frame, t.frames, t.rejoins, t.n_classified) == (0, 995, 200, 2, 20)
    occ = {f: 1 for f in range(0, 1000, 5)}
    a = build_analytics(tracks, doc["meta"], {}, occ, min_gender_conf=0.6, min_age_conf=0.4)
    p = a["tracks"][0]
    assert (p["gender"], p["age"], p["gender_votes"]) == ("male", "Age16-30", {"male": 20})
    assert p["first_sec"] == 0.0 and p["last_sec"] == 39.8 and p["rejoins"] == 2
