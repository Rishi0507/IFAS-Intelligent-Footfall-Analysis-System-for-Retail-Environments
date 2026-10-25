"""Zone occupancy (src/inference/zones.py), tiled detection and the zone API.

Synthetic tracks with known answers: who sat where, for how long, with how
much detection flicker. Every expected number below is worked out by hand.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")

from src.inference import zones as Z  # noqa: E402
from src.inference.pipeline import detect_people, merge_boxes, parse_tiles  # noqa: E402

FPS, STRIDE = 10.0, 5            # one sample every 0.5 s
W, H = 1000, 500


def seat(x, y, w=80, h=120):
    """A box whose 'lower' anchor (85% down) sits at (x, y)."""
    top = y - 0.85 * h
    return [x - w / 2, top, x + w / 2, top + h]


def obs(frames, x, y, conf=0.6, **kw):
    return [[f, *seat(x, y, **kw), conf] for f in frames]


def tb_doc(tracks, seconds):
    return {"schema": 1, "fps": FPS, "width": W, "height": H, "stride": STRIDE,
            "frames_total": int(seconds * FPS), "tracks": {str(k): v for k, v in tracks.items()}}


def rect(x1, y1, x2, y2):
    return [[x1 / W, y1 / H], [x2 / W, y1 / H], [x2 / W, y2 / H], [x1 / W, y2 / H]]


ZONES = {"zones": [
    {"id": "T1", "kind": "table", "polygon": rect(100, 200, 300, 400)},
    {"id": "T2", "kind": "table", "polygon": rect(600, 200, 800, 400)},
]}


# ---------------------------------------------------------------- geometry
def test_polygon_area_and_point_in_polygon():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert Z.polygon_area(sq) == 100
    ell = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]       # concave L
    assert Z.polygon_area(ell) == 64
    assert Z.point_in_polygon(2, 8, ell)
    assert not Z.point_in_polygon(8, 8, ell)                         # the notch
    pts = Z.points_in_polygon([[1, 1], [11, 1], [5, 5]], sq)
    assert pts.tolist() == [True, False, True]


def test_rect_polygon_overlap_is_exact():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert Z.rect_polygon_overlap((2, 2, 4, 4), sq) == pytest.approx(4)
    assert Z.rect_polygon_overlap((5, 5, 15, 15), sq) == pytest.approx(25)
    assert Z.rect_polygon_overlap((20, 20, 30, 30), sq) == 0
    tri = [(0, 0), (10, 0), (0, 10)]
    assert Z.rect_polygon_overlap((0, 0, 10, 10), tri) == pytest.approx(50)
    assert Z.rect_polygon_overlap((0, 0, 5, 5), tri) == pytest.approx(25)


def test_anchor_rules():
    box = (0, 0, 10, 100)
    assert Z.anchor_point(box, "foot") == (5, 100)
    assert Z.anchor_point(box, "center") == (5, 50)
    assert Z.anchor_point(box, "lower", 0.85) == (5, 85)


def test_normalise_zones_validates_and_converts_pixels():
    doc = Z.normalise_zones({"units": "px", "zones": [
        {"id": "A", "polygon": [[0, 0], [500, 0], [500, 250]]}]}, W, H)
    assert doc["zones"][0]["polygon"] == [[0, 0], [0.5, 0], [0.5, 0.5]]
    assert doc["zones"][0]["kind"] == "table"
    bad = [
        {"zones": [{"id": "A", "polygon": [[0, 0], [1, 0]]}]},                          # 2 points
        {"zones": [{"id": "A", "polygon": [[0, 0], [1, 0], [2, 0]]}]},                  # no area
        {"zones": [{"id": "bad id!", "polygon": rect(0, 0, 10, 10)}]},
        {"zones": [{"id": "A", "polygon": rect(0, 0, 10, 10)},
                   {"id": "A", "polygon": rect(20, 0, 30, 10)}]},                        # duplicate
        {"zones": [{"id": "A", "kind": "sofa", "polygon": rect(0, 0, 10, 10)}]},
        {"zones": "nope"},
        {"units": "px", "zones": [{"id": "A", "polygon": rect(0, 0, 10, 10)}]},          # no size
    ]
    for d in bad:
        with pytest.raises(ValueError):
            Z.normalise_zones(d, None, None) if d.get("units") == "px" else Z.normalise_zones(d)


def test_params_reject_nonsense():
    with pytest.raises(ValueError):
        Z.ZoneParams.from_dict({"anchor": "head"})
    with pytest.raises(ValueError):
        Z.ZoneParams.from_dict({"on_sec": -1})
    with pytest.raises(ValueError):
        Z.ZoneParams.from_dict({"off_sec": "soon"})
    assert Z.ZoneParams.from_dict({"on_sec": "4"}).on_sec == 4.0


# --------------------------------------------------------------- smoothing
def test_smooth_closes_gaps_then_drops_blips():
    m = np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    # dt 0.5 s: the 1-sample gap (0.5 s) closes under off 2 s, the 6-sample gap
    # (3 s) does not; the lone sample at 11 lasts 0.5 s < on 1 s and goes.
    assert Z.smooth(m, 0.5, on_sec=1.0, off_sec=2.0) == [(0, 4)]
    assert Z.smooth(m, 0.5, on_sec=0.5, off_sec=2.0) == [(0, 4), (11, 11)]
    assert Z.smooth(m, 0.5, on_sec=1.0, off_sec=4.0) == [(0, 11)]   # all one session
    assert Z.smooth(np.zeros(5, bool), 0.5, 1, 1) == []


def test_smooth_stretches_stays_cut_off_by_the_recording():
    # Seen from 0.5 s to 3 s of a 4.5 s clip: both edge gaps (0.5 s, 1.5 s) are
    # shorter than off 2 s, so they are missed detections, not arrivals/exits.
    m = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0], bool)
    assert Z.smooth(m, 0.5, on_sec=1.0, off_sec=2.0) == [(0, 8)]
    assert Z.smooth(m, 0.5, on_sec=1.0, off_sec=2.0, edges=False) == [(1, 5)]
    # A blip near the end is still dropped first, not stretched into a stay.
    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], bool)
    assert Z.smooth(b, 0.5, on_sec=1.0, off_sec=2.0) == []


# ---------------------------------------------------------------- analysis
def scenario():
    """60 s at 2 samples/s.

    A sits at T1 from 0 s to 30 s, missed on every third sample (flicker).
    B sits at T1 from 40 s to the end: still there when the recording stops.
    C sits at T2 from 0 s to 20 s, already there when it starts.
    D walks across T2 at 45-46 s: a passer-by, not a session.
    E stands outside both tables for the whole minute.
    """
    a = [f for f in range(0, 300, STRIDE) if (f // STRIDE) % 3 != 2]
    tracks = {
        1: obs(a, 200, 300),
        2: obs(range(400, 600, STRIDE), 210, 310),
        3: obs(range(0, 200, STRIDE), 700, 300),
        4: obs(range(450, 460, STRIDE), 700, 300),
        5: obs(range(0, 600, STRIDE), 450, 300),
    }
    return tb_doc(tracks, 60)


def test_analyse_sessions_censoring_and_metrics():
    out = Z.analyse(scenario(), ZONES, {"on_sec": 3, "off_sec": 5, "gap_fill_sec": 0})
    t1, t2 = out["zones"]
    s1 = t1["sessions"]
    assert [(s["start"], s["end"]) for s in s1] == [(0.0, 29.5), (40.0, 60.0)]
    assert s1[0]["censored_start"] and not s1[0]["censored_end"]
    assert s1[1]["censored_end"] and not s1[1]["censored_start"]
    assert s1[0]["people"] == [1] and s1[1]["people"] == [2]
    assert t1["metrics"]["occupied_sec"] == pytest.approx(49.5)
    assert t1["metrics"]["utilization"] == pytest.approx(49.5 / 60, abs=1e-4)
    # Flicker: A's missed samples flip raw presence constantly; smoothing keeps 3 changes.
    assert t1["metrics"]["raw_flips_per_min"] > 10 * t1["metrics"]["smoothed_flips_per_min"] / 3
    assert [(s["start"], s["end"]) for s in t2["sessions"]] == [(0.0, 20.0)]   # D did not count
    assert out["summary"]["sessions"] == 3
    assert out["summary"]["occupied_at_end"] == 1 and out["summary"]["free_at_end"] == 1
    assert out["summary"]["unassigned_share"] > 0.4                   # E is in no zone
    kinds = [(e["type"], e["zone"], e["t"]) for e in out["events"]]
    assert ("freed", "T2", 20.0) in kinds and ("freed", "T1", 29.5) in kinds
    assert ("occupied", "T1", 40.0) in kinds
    assert not any(e[0] == "occupied" and e[2] == 0.0 for e in kinds)   # censored starts are not events
    assert out["next_free"]["at_end"][0]["zone"] == "T1"
    assert out["latency_sec"] == {"occupied": 3.0, "freed": 5.0}


def test_gap_fill_bridges_misses_but_not_moves():
    # Seen at 0 s and 1.5 s in T1, and in between outside every zone at 0.5 s.
    tracks = {1: [[0, *seat(200, 300), .9], [5, *seat(450, 300), .9], [15, *seat(200, 300), .9]]}
    counts, _ = Z.presence(tb_doc(tracks, 2), Z.normalise_zones(ZONES)["zones"],
                           Z.ZoneParams(gap_fill_sec=2.0))
    assert counts[0].tolist() == [1, 0, 0, 1]                          # the move breaks the fill
    tracks = {1: [[0, *seat(200, 300), .9], [15, *seat(200, 300), .9]]}
    counts, _ = Z.presence(tb_doc(tracks, 2), Z.normalise_zones(ZONES)["zones"],
                           Z.ZoneParams(gap_fill_sec=2.0))
    assert counts[0].tolist() == [1, 1, 1, 1]


def test_smallest_zone_wins_and_overlap_rule():
    zs = Z.normalise_zones({"zones": [
        {"id": "big", "polygon": rect(0, 0, 1000, 500), "kind": "area"},
        {"id": "seat", "polygon": rect(150, 250, 250, 350), "kind": "seat"}]})["zones"]
    tracks = {1: obs([0], 200, 300)}
    got = Z.assign_observations(tb_doc(tracks, 1), zs, Z.ZoneParams())
    assert got["1"] == [(0, 1)]
    # Overlap rule: the lower half of the box must be mostly inside the zone.
    tracks = {1: [[0, 150, 200, 250, 400, .9]]}                        # lower half 150-250 x 300-400
    zs2 = Z.normalise_zones({"zones": [{"id": "t", "polygon": rect(150, 300, 250, 350)}]})["zones"]
    hit = Z.assign_observations(tb_doc(tracks, 1), zs2, Z.ZoneParams(anchor="overlap", min_overlap=0.4))
    miss = Z.assign_observations(tb_doc(tracks, 1), zs2, Z.ZoneParams(anchor="overlap", min_overlap=0.6))
    assert hit["1"] == [(0, 0)] and miss["1"] == [(0, -1)]


def test_queue_waits():
    zones = {"zones": [{"id": "Q", "kind": "queue", "polygon": rect(100, 200, 300, 400)}]}
    tracks = {1: obs(range(0, 100, STRIDE), 200, 300), 2: obs(range(50, 300, STRIDE), 220, 310)}
    out = Z.analyse(tb_doc(tracks, 40), zones, {"on_sec": 1, "off_sec": 2, "queue_wait_sec": 20})
    q = out["zones"][0]["queue"]
    assert [w["track"] for w in q["waits"]] == [1, 2]
    assert q["waits"][0]["wait"] == pytest.approx(10.0) and q["waits"][1]["wait"] == pytest.approx(25.0)
    assert q["max_length"] == 2
    assert [e["track"] for e in out["events"] if e["type"] == "queue_wait"] == [2]


# ---------------------------------------------------------------- survival
def test_kaplan_meier_matches_hand_computation():
    km = Z.kaplan_meier([2, 3, 3, 5], [True, True, False, True])
    assert km["t"] == [0.0, 2.0, 3.0, 5.0]
    assert km["s"] == pytest.approx([1.0, 0.75, 0.5, 0.0])
    assert Z.km_at(km, 2.5) == 0.75
    # E[T - 1 | T > 1] over [1, 5]: 1*1 + 0.75*1 + 0.5*2 = 2.75
    assert Z.mean_residual(km, 1.0) == pytest.approx(2.75)
    assert Z.mean_residual(km, 6.0) is None


def test_next_free_evaluation_is_causal_and_counts():
    def s(start, end, cs=False, ce=False):
        return {"start": start, "end": end, "duration": end - start, "censored_start": cs, "censored_end": ce}
    # Two tables; the one seated longer always empties first.
    sessions = [[s(0, 50, cs=True), s(60, 100)], [s(5, 80), s(85, 120, ce=True)]]
    ev = Z.evaluate_next_free(sessions, 120, step=10)
    assert ev["moments"] > 0
    assert ev["longest_seated"] == 1.0
    assert 0 < ev["random"] <= 0.5
    assert ev["km_fallbacks"] == ev["moments"]                         # never 3 ended sessions in time


# ---------------------------------------------------------------- proposal
def test_propose_zones_finds_seats_in_reading_order():
    tracks = {
        1: obs(range(0, 400, STRIDE), 600, 200),     # back row, right
        2: obs(range(0, 400, STRIDE), 300, 210),     # back row, left
        3: obs(range(0, 400, STRIDE), 450, 420),     # front row
        4: [[f, *seat(50 + 2 * f, 300), .8] for f in range(0, 400, STRIDE)],   # walks, never sits
        5: obs(range(100, 150, STRIDE), 450, 425),   # 5 s at the front seat: too short alone
    }
    prop = Z.propose_zones(tb_doc(tracks, 40), min_stay_sec=20, min_evidence_sec=30)
    ids = [z["id"] for z in prop["zones"]]
    assert ids == ["S01", "S02", "S03"]
    assert prop["zones"][0]["evidence_sec"] >= 30
    # Each place has about 40 s of sitting behind it; asking for more drops it.
    assert Z.propose_zones(tb_doc(tracks, 40), min_stay_sec=20, min_evidence_sec=60)["zones"] == []
    centres = [Z.polygon_centroid(z["polygon"]) for z in prop["zones"]]
    assert centres[0][0] < centres[1][0]                               # left before right
    assert centres[2][1] > centres[0][1]                               # front row last
    out = Z.analyse(tb_doc(tracks, 40), prop)
    assert all(z["metrics"]["utilization"] > 0.9 for z in out["zones"])


def test_stick_keeps_a_leaning_person_in_their_seat():
    # Two seats side by side, 100 px wide. The person sits in the left one and
    # leans 10 px over the boundary every other second.
    zones = {"units": "px", "zones": [
        {"id": "L", "kind": "seat", "polygon": [[0, 0], [100, 0], [100, 200], [0, 200]]},
        {"id": "R", "kind": "seat", "polygon": [[100, 0], [200, 0], [200, 200], [100, 200]]}]}
    frames = range(0, 600, STRIDE)
    tracks = {1: [[f, *((60, 20, 100, 180) if (f // 10) % 2 == 0 else (90, 20, 130, 180)), .9]
                  for f in frames]}
    tb = tb_doc(tracks, 60)
    tb["width"], tb["height"] = 200, 200
    loose = Z.analyse(tb, zones, {"stick": 0, "on_sec": 0.5, "off_sec": 0.5})
    tight = Z.analyse(tb, zones, {"stick": 0.5, "on_sec": 0.5, "off_sec": 0.5})
    util = lambda out: {z["id"]: z["metrics"]["utilization"] for z in out["zones"]}
    assert util(loose)["R"] > 0.3                                      # flickers into the neighbour's seat
    assert util(tight)["R"] == 0 and util(tight)["L"] > 0.95
    # Sitting down in the right seat's core does move them.
    moved = {1: [[f, *(60, 20, 100, 180), .9] for f in range(0, 300, STRIDE)] +
                [[f, *(130, 20, 170, 180), .9] for f in range(300, 600, STRIDE)]}
    tb["tracks"] = {str(k): v for k, v in moved.items()}
    after = util(Z.analyse(tb, zones, {"stick": 0.5, "on_sec": 0.5, "off_sec": 0.5}))
    assert 0.4 < after["L"] < 0.6 and 0.4 < after["R"] < 0.6


def test_analyse_run_saves_and_reuses(tmp_path):
    tb = scenario()
    (tmp_path / "track_boxes.json").write_text(json.dumps(tb))
    (tmp_path / "analytics.json").write_text(json.dumps({"meta": {}, "kpis": {}}))
    block = Z.analyse_run(tmp_path, ZONES, {"on_sec": 3, "off_sec": 5})
    assert (tmp_path / "zones.json").exists()
    assert json.loads((tmp_path / "analytics.json").read_text())["zones"]["summary"] == block["summary"]
    again = Z.analyse_run(tmp_path, None, {"on_sec": 3, "off_sec": 5}, save=False)   # reads zones.json
    assert again["summary"] == block["summary"]
    with pytest.raises(FileNotFoundError):
        Z.analyse_run(tmp_path / "missing", ZONES)


# ------------------------------------------------------------------ tiling
class _Box:
    def __init__(self, xyxy, conf):
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.conf = [conf]


class _Res:
    def __init__(self, boxes):
        self.boxes = boxes


class TileDetector:
    """Sees the white person drawn in the frame, but only on a tile: on the
    whole frame they are too small, which is what tiling is for."""

    def __init__(self):
        self.shapes = []

    def __call__(self, img, **kw):
        h, w = img.shape[:2]
        self.shapes.append((w, h))
        ys, xs = np.nonzero(img[:, :, 0] > 128)
        if (w, h) == (400, 300) or not len(xs):
            return [_Res([])]
        return [_Res([_Box([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], 0.7)])]


def test_detect_people_with_tiles_maps_back_and_merges():
    frame = np.zeros((300, 400, 3), np.uint8)
    frame[100:170, 180:220] = 255                 # one person, inside the tiles' overlap
    det = TileDetector()
    assert detect_people(det, frame, 0.1) == []
    found = detect_people(det, frame, 0.1, tiles=(2, 2), overlap=0.2)
    assert len(det.shapes) == 1 + 1 + 4
    # Two top tiles see the whole person (in their own coordinates) and the two
    # bottom tiles see the lower half: all four collapse to one box.
    assert found == [((180, 100, 220, 170), 0.7)]
    assert parse_tiles("2x2") == (2, 2) and parse_tiles(None) is None
    with pytest.raises(ValueError):
        parse_tiles("9x9")


def test_recording_time_from_recorder_file_names():
    import datetime as dt
    from src.inference.pipeline import time_from_name
    name = "MUM 192.0.2.102 DJSCE 2_XRN-3010A_CH021(192.0.2.102)_20260914_165019_170000_ID_0100.avi"
    assert time_from_name(name) == dt.datetime(2026, 9, 14, 16, 50, 19)
    assert time_from_name("cam1_2026-10-20T09-15-00.mp4") == dt.datetime(2026, 10, 20, 9, 15)
    assert time_from_name("store entrance.mp4") is None
    assert time_from_name("clip_20261399_250000.mp4") is None          # not a real date


def test_merge_boxes_drops_duplicates_and_tile_fragments():
    kept = merge_boxes([((0, 0, 10, 20), 0.9), ((1, 1, 11, 21), 0.5),     # duplicate
                        ((0, 0, 10, 9), 0.4),                              # upper fragment
                        ((50, 0, 60, 20), 0.3)])
    assert [c for _b, c in kept] == [0.9, 0.3]


# --------------------------------------------------------------- pipeline
def test_pipeline_writes_track_boxes_and_zones(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from src.inference.pipeline import process_video

    path = tmp_path / "clip.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for _ in range(60):
        wr.write(np.full((180, 320, 3), 40, np.uint8))
    wr.release()

    class Still:
        def __call__(self, frame, **kw):
            return [_Res([_Box([100, 40, 140, 140], 0.9)])]

    class Age:
        def predict(self, crops):
            return [{"Age16-30": 1.0, "Age31-45": 0, "Age46-60": 0, "AgeAbove61": 0} for _ in crops]

    zones = {"zones": [{"id": "D1", "kind": "seat",
                        "polygon": [[0.25, 0.5], [0.5, 0.5], [0.5, 0.95], [0.25, 0.95]]}]}
    doc = process_video(path, tmp_path / "run", stride=2, write_video=False, device="cpu",
                        detector=Still(), age_model=Age(), zones=zones,
                        zone_params={"on_sec": 1, "off_sec": 1})
    tb = json.loads((tmp_path / "run" / "track_boxes.json").read_text())
    assert tb["stride"] == 2 and len(tb["tracks"]) == 1
    assert len(next(iter(tb["tracks"].values()))) == 30
    assert doc["zones"]["zones"][0]["metrics"]["utilization"] == pytest.approx(1.0)
    assert json.loads((tmp_path / "run" / "zones.json").read_text())["zones"][0]["id"] == "D1"
    # Without zones, the person who sat still for 6 s is not a seat at 20 s...
    doc2 = process_video(path, tmp_path / "run2", stride=2, write_video=False, device="cpu",
                         detector=Still(), age_model=Age())
    assert "zones" not in doc2


# -------------------------------------------------------------------- API
@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import app.server as server

    runs = tmp_path / "runs"
    run = runs / "class"
    run.mkdir(parents=True)
    (run / "track_boxes.json").write_text(json.dumps(scenario()))
    (run / "analytics.json").write_text(json.dumps({
        "meta": {"source": "c", "site": "DJSCE", "camera": "CR22", "duration_sec": 60},
        "kpis": {"visitors": 5}}))
    monkeypatch.setattr(server, "RUNS", runs)
    monkeypatch.setattr(server, "CAMERA_ZONES", tmp_path / "zones")
    return TestClient(server.app), run, tmp_path


def test_zone_api_preview_save_propose_and_labels(client):
    c, run, root = client
    r = c.get("/api/runs/class/zones")
    assert r.status_code == 200 and r.json()["has_boxes"] and r.json()["zones"] is None

    r = c.put("/api/runs/class/zones", json={"zones": ZONES, "params": {"on_sec": 3}, "save": False})
    assert r.status_code == 200 and r.json()["summary"]["zones"] == 2
    assert not (run / "zones.json").exists()                          # a preview writes nothing

    r = c.put("/api/runs/class/zones", json={"zones": ZONES, "save": True, "camera": True})
    assert r.status_code == 200 and (run / "zones.json").exists()
    assert (root / "zones" / "djsce__cr22.json").exists()
    assert c.get("/api/runs/class/zones").json()["analysis"]["summary"]["zones"] == 2

    assert c.put("/api/runs/class/zones", json={"zones": {"zones": [{"id": "x"}]}}).status_code == 400
    assert c.put("/api/runs/class/zones", json={"params": {"anchor": "nose"}}).status_code == 400
    assert c.put("/api/runs/nope/zones", json={}).status_code == 404
    assert c.put("/api/runs/..%2Fetc/zones", json={}).status_code in (404, 405)   # never reaches a run

    r = c.post("/api/runs/class/zones/propose", json={"min_stay_sec": 10})
    assert r.status_code == 200 and r.json()["source"] == "auto" and r.json()["zones"]

    r = c.post("/api/runs/class/zone-labels",
               json={"annotator": "ann1", "frames": {"10": {"T1": True, "T2": None}}})
    assert r.status_code == 200 and r.json() == {"saved": 1}
    got = c.get("/api/runs/class/zone-labels/ann1").json()
    assert got["frames"] == {"10.00": {"T1": True, "T2": None}}
    assert c.post("/api/runs/class/zone-labels",
                  json={"annotator": "ann1", "frames": {"10": {"T1": "yes"}}}).status_code == 400
    assert c.post("/api/runs/class/zone-labels",
                  json={"annotator": "../x", "frames": {}}).status_code == 400
    assert "ann1" in c.get("/api/runs/class/zones").json()["annotators"]
