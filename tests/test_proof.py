"""Evidence for the guide's claims (src/inference/proof.py) and its endpoints."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.proof import handoffs, moment, occupancy_peak, portraits, snap, still  # noqa: E402

FPS = 25.0


def tb_doc(tracks):
    """track_boxes.json with boxes on every 5th frame: {id: [(frame, box), ...]}."""
    return {"schema": 1, "fps": FPS, "width": 640, "height": 360, "stride": 5,
            "tracks": {str(k): [[f, *box, 0.8] for f, box in v] for k, v in tracks.items()}}


def test_peak_is_the_first_moment_most_people_overlap():
    tracks = [{"first_sec": 0.0, "last_sec": 10.0}, {"first_sec": 4.0, "last_sec": 6.0},
              {"first_sec": 5.0, "last_sec": 20.0}, {"first_sec": 6.5, "last_sec": 7.0}]
    assert occupancy_peak(tracks) == (5.0, 3)
    # someone leaving as another arrives overlaps them, as in analytics.py
    assert occupancy_peak([{"first_sec": 0.0, "last_sec": 2.0}, {"first_sec": 2.0, "last_sec": 3.0}]) == (2.0, 2)
    assert occupancy_peak([]) == (0.0, 0)


def test_moment_counts_everyone_in_view_including_people_missed_in_that_frame():
    tb = tb_doc({1: [(0, (10, 10, 50, 90)), (5, (12, 10, 52, 90)), (10, (12, 10, 52, 90))],
                 2: [(0, (100, 10, 140, 90)), (10, (100, 10, 140, 90))],      # missed at frame 5
                 3: [(20, (200, 10, 240, 90))]})                              # not there yet
    doc = {"meta": {"fps": FPS}, "tracks": [{"id": 1, "gender": "female", "age": "Age16-30"}],
           "zones": {"zones": [
               {"id": "S1", "name": "Seat 1", "kind": "seat", "polygon": [[0, 0], [.1, 0], [.1, .3], [0, .3]],
                "sessions": [{"start": 0.0, "end": 0.4, "people": [1]}]},
               {"id": "S2", "name": "Seat 2", "kind": "seat", "polygon": [[.5, 0], [.6, 0], [.6, .3], [.5, .3]],
                "sessions": [{"start": 0.8, "end": 2.0, "people": [3]}]}]}}
    m = moment(doc, tb, 0.2)
    assert (m["frame"], m["seen"], m["hidden"]) == (5, 1, 1)
    hidden = next(p for p in m["people"] if not p["seen"])
    assert hidden["track"] == 2 and hidden["box"] == [100, 10, 140, 90] and hidden["ago"] == 0.2
    assert m["people"][0]["gender"] == "female"
    assert [z["taken"] for z in m["zones"]] == [True, False]
    assert m["zones"][0]["since"] == 0.0 and m["zones"][0]["people"] == [1]


def test_a_track_restarting_in_the_same_spot_is_a_handoff_and_a_neighbour_is_not():
    box = (100, 100, 160, 220)
    tb = tb_doc({1: [(0, box), (50, box)],
                 2: [(300, (102, 100, 162, 222)), (320, box)],          # 10 s later, same spot
                 3: [(0, (300, 100, 360, 220)), (50, (300, 100, 360, 220))],
                 4: [(300, (370, 100, 430, 220))],                      # beside 3, not on it
                 5: [(0, (500, 10, 560, 130))],
                 6: [(1200, (500, 10, 560, 130))]})                     # same spot, 48 s later
    found = handoffs(tb)
    assert [(h["a"], h["b"]) for h in found] == [(1, 2)]
    assert found[0]["gap"] == 10.0 and found[0]["overlap"] > 0.9
    assert [(h["a"], h["b"]) for h in handoffs(tb, max_gap_sec=60)] == [(1, 2), (5, 6)]


def test_portraits_pick_each_persons_clearest_crop():
    rows = [{"track": 1, "crop": "crops/1/a.png", "t": 0.0, "det_conf": 0.9, "cues": {"size": 0.01, "occlusion": 0.0}},
            {"track": 1, "crop": "crops/1/b.png", "t": 4.0, "det_conf": 0.9, "cues": {"size": 0.04, "occlusion": 0.1}},
            {"track": 1, "crop": "crops/1/c.png", "t": 8.0, "det_conf": 0.9, "cues": {"size": 0.05, "occlusion": 0.9}},
            {"track": 2, "crop": "crops/2/a.png", "t": 1.0, "det_conf": 0.5, "cues": {}}]
    assert portraits(rows, [1, 3]) == {1: {"crop": "crops/1/b.png", "t": 4.0}}


@pytest.fixture
def video(tmp_path):
    """Two seconds of 320x180 video whose frame k is filled with grey level 10*k."""
    path = tmp_path / "clean.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 180))
    for k in range(50):
        out.write(np.full((180, 320, 3), min(255, 5 * k), np.uint8))
    out.release()
    return path


def _decode(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def test_still_is_the_frame_at_that_moment_and_is_cached(video, tmp_path):
    cache = tmp_path / "stills"
    img = _decode(still(video, 1.0, 480, cache))
    assert img.shape[:2] == (270, 480)
    assert abs(float(img.mean()) - 5 * 25) < 8                    # frame 25
    assert len(list(cache.iterdir())) == 1
    assert still(video, 1.0, 480, cache) == (cache / "1.00_480.jpg").read_bytes()
    with pytest.raises(ValueError):
        still(video, 1.0, 123, cache)


def test_snap_cuts_one_box_out_of_the_frame(video, tmp_path):
    img = _decode(snap(video, 0.4, (320, 180, 640, 360), 640, tmp_path / "snaps", height=120))
    assert img.shape[0] == 120 and abs(float(img.mean()) - 50) < 8


# --------------------------------------------------------------- endpoints
@pytest.fixture
def client(tmp_path, monkeypatch, video):
    import app.server as server
    from fastapi.testclient import TestClient
    runs = tmp_path / "runs"
    run = runs / "r1"
    run.mkdir(parents=True)
    video.rename(run / "clean.mp4")
    (run / "track_boxes.json").write_text(json.dumps(tb_doc({
        1: [(0, (10, 10, 50, 90)), (5, (10, 10, 50, 90))],
        2: [(30, (11, 10, 51, 90)), (40, (11, 10, 51, 90))]})), encoding="utf-8")
    (run / "analytics.json").write_text(json.dumps({
        "meta": {"fps": FPS, "width": 640, "height": 360, "duration_sec": 2.0},
        "kpis": {"visitors": 2, "peak_occupancy": 1, "avg_occupancy": 1.0},
        "timeline": {"bin_sec": 1.0, "bins": []}, "distributions": {},
        "tracks": [{"id": 1, "gender": "male", "first_sec": 0.0, "last_sec": 0.2, "dwell_sec": 0.2},
                   {"id": 2, "gender": None, "first_sec": 1.2, "last_sec": 1.6, "dwell_sec": 0.4}]}), encoding="utf-8")
    (run / "crops.jsonl").write_text(json.dumps({"track": 1, "crop": "crops/1/000000.png", "t": 0.0,
                                                 "det_conf": 0.8, "cues": {"size": 0.02}}) + "\n", encoding="utf-8")
    monkeypatch.setattr(server, "RUNS", runs)
    return TestClient(server.app)


def test_evidence_endpoints(client):
    m = client.get("/api/runs/r1/moment?t=0.2").json()
    assert m["seen"] == 1 and m["people"][0]["gender"] == "male"
    r = client.get("/api/runs/r1/still?t=1&w=480")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/api/runs/r1/still?t=1&w=999").status_code == 400
    h = client.get("/api/runs/r1/handoffs").json()
    assert h["count"] == 1 and h["examples"][0]["a_url"].startswith("/api/runs/r1/snap?t=0.2&box=10,10,50,90")
    assert client.get(h["examples"][0]["b_url"]).headers["content-type"] == "image/jpeg"
    assert client.get("/api/runs/r1/snap?t=0&box=0,0,9999,10").status_code == 400
    assert client.get("/api/runs/r1/snap?t=0&box=a,b,c,d").status_code == 400
    assert client.get("/api/runs/r1/portraits?tracks=1,2").json() == {
        "1": {"url": "/api/runs/r1/crops/1/000000.png", "t": 0.0}}
    assert client.get("/api/runs/r1/portraits?tracks=x").status_code == 400
    assert client.get("/api/runs/..%2Fr1/moment?t=0").status_code == 404
