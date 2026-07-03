import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.tracking import Tracker


def _cfg():
    return {"iou_threshold": 0.30, "centroid_max_dist": 80.0,
            "persist_frames": 3, "size_scale": 0.30}


def _det(box, gender="uncertain"):
    return {"box": box.astype(np.float32), "score": 0.9, "gender": gender,
            "face_label": "unknown", "face_conf": 0.0,
            "clip_gender": "unknown", "clip_ratio": 0.0, "clip_embedding": None}


def test_first_frame_mints_id():
    t = Tracker(_cfg())
    tracks = t.update([_det(np.array([0, 0, 50, 80]))], frame_idx=1)
    assert list(tracks.keys()) == [1]


def test_iou_match_keeps_id():
    t = Tracker(_cfg())
    t.update([_det(np.array([0, 0, 50, 80]))], 1)
    tracks = t.update([_det(np.array([2, 2, 52, 82]))], 2)
    assert list(tracks.keys()) == [1]


def test_expire_after_persist():
    t = Tracker(_cfg())
    t.update([_det(np.array([0, 0, 50, 80]))], 1)
    for fr in range(2, 7):
        t.update([], fr)
    assert len(t.tracks) == 0
