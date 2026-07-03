import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.tracking import Track
from src.counting import LineCounter


def test_crossing_counts_entry():
    lc = LineCounter([[0.0, 0.5], [1.0, 0.5]], direction="down")
    lc.set_resolution(100, 100)
    t1 = {1: Track(track_id=1, box=np.array([40, 10, 60, 30], dtype=np.float32), last_seen=1)}
    lc.update(t1, 1)
    t2 = {1: Track(track_id=1, box=np.array([40, 60, 60, 90], dtype=np.float32), last_seen=2)}
    events = lc.update(t2, 2)
    assert lc.entries == 1
    assert events and events[0]["type"] == "entry"


def test_no_crossing_no_count():
    lc = LineCounter([[0.0, 0.5], [1.0, 0.5]], direction="down")
    lc.set_resolution(100, 100)
    t1 = {1: Track(track_id=1, box=np.array([40, 10, 60, 30], dtype=np.float32), last_seen=1)}
    lc.update(t1, 1)
    t2 = {1: Track(track_id=1, box=np.array([40, 15, 60, 35], dtype=np.float32), last_seen=2)}
    events = lc.update(t2, 2)
    assert events == [] and lc.entries == 0
