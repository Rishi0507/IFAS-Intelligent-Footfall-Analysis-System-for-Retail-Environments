"""RQ4 evaluation helpers (src/research/zone_eval.py), on hand-made analyses."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")

from src.research import zone_eval as E  # noqa: E402


def zone(zid, sessions, kind="seat", util=0.5):
    return {"id": zid, "kind": kind, "metrics": {"utilization": util, "sessions": len(sessions)},
            "sessions": [{"start": a, "end": b} for a, b in sessions]}


def block(zones, duration=10.0):
    return {"duration_sec": duration, "zones": zones}


def test_taken_at_and_agreement():
    a = block([zone("S1", [(0, 5)]), zone("S2", [(2, 4)]), zone("Q", [(0, 10)], kind="queue")])
    st = E.taken_at(a, [0, 3, 5, 9])
    assert set(st) == {"S1", "S2"}                                   # queues are not seats
    assert st["S1"].tolist() == [True, True, False, False]           # end is exclusive
    assert st["S2"].tolist() == [False, True, False, False]
    assert E.agreement(a, a)["state_agreement"] == 1.0
    b = block([zone("S1", [(0, 6)], util=0.6), zone("S2", [(2, 4)])])
    ag = E.agreement(a, b, step=1.0)
    assert ag["state_agreement"] == pytest.approx(19 / 20)            # S1 differs at t=5 only
    assert ag["utilization_abs_err"] == pytest.approx(0.05)


def test_subsample_keeps_the_coarser_grid():
    tb = {"stride": 5, "tracks": {"1": [[0], [5], [10], [15], [20]]}}
    sub = E.subsample(tb, 2)
    assert sub["stride"] == 10 and [o[0] for o in sub["tracks"]["1"]] == [0, 10, 20]
    assert E.subsample(tb, 1) is tb


def test_score_against_labels():
    b = block([zone("S1", [(0, 5)]), zone("S2", [(0, 10)]), zone("T", [(0, 10)], kind="queue")])
    labels = {"frames": {
        "1.00": {"S1": True, "S2": True, "T": False},     # T is a queue: ignored
        "6.00": {"S1": True, "S2": None},                # S1 wrong (system says free), S2 unsure
        "8.00": {"S1": False, "S2": True},
    }}
    s = E.score(b, labels, n_boot=200)
    # Items: (1 s, S1) (1 s, S2) (6 s, S1) (8 s, S1) (8 s, S2); truth T T T F T, system T T F F T.
    assert s["items"] == 5 and s["unsure"] == 1 and s["zones"] == 2 and s["moments"] == 3
    assert s["accuracy"]["value"] == pytest.approx(4 / 5)
    assert s["precision"]["value"] == 1.0                             # every "taken" said was right
    assert s["recall"]["value"] == pytest.approx(3 / 4)
    assert s["f1"]["value"] == pytest.approx(6 / 7, abs=1e-4)
    assert s["base_rate_taken"] == pytest.approx(4 / 5)
    assert E.score(b, {"frames": {}})["items"] == 0


def test_inter_annotator_and_majority():
    a = {"frames": {"1.00": {"S1": True, "S2": False, "S3": True}}}
    b = {"frames": {"1.00": {"S1": True, "S2": True, "S3": None}}}
    c = {"frames": {"1.00": {"S1": False, "S2": True, "S3": True}}}
    pairs = E.inter_annotator({"a": a, "b": b})
    assert pairs[0]["items"] == 2 and pairs[0]["agreement"] == 0.5
    m = E.majority_labels({"a": a, "b": b, "c": c})["frames"]["1.00"]
    assert m == {"S1": True, "S2": True, "S3": True}
    tie = E.majority_labels({"a": a, "c": c})["frames"]["1.00"]
    assert tie["S1"] is None and tie["S3"] is True
