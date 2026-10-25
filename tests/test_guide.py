"""The plain-language guide (src/inference/guide.py)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.guide import setting_of, shop_guide  # noqa: E402


def doc(visitors=60, duration=600.0, dwell=12.0, peak=8, avg=4.0, first=None, zones=None, gender=True):
    tracks = []
    for i in range(visitors):
        f = first(i) if first else (i / visitors) * duration
        tracks.append({"id": i, "gender": ("female" if i % 3 == 0 else "male") if gender else None,
                       "age": "Age16-30", "first_sec": f, "last_sec": f + dwell, "dwell_sec": dwell,
                       "zone": "Center"})
    bins = [{"t": t * 60.0, "entries": 6 if t == 2 else 3, "occupancy_peak": peak if t == 2 else 3}
            for t in range(int(duration // 60))]
    d = {"meta": {"duration_sec": duration, "captured_at": "2026-09-05T14:00:00"},
         "kpis": {"visitors": visitors, "peak_occupancy": peak, "avg_occupancy": avg,
                  "female_pct": 33.3, "male_pct": 66.7},
         "timeline": {"bin_sec": 60.0, "bins": bins},
         "distributions": {"zones": {"Left": {"visitors": 5, "avg_dwell": 8.0},
                                     "Center": {"visitors": visitors - 10, "avg_dwell": 12.0},
                                     "Right": {"visitors": 5, "avg_dwell": 9.0}}},
         "tracks": tracks, "model": {}}
    if zones:
        d["zones"] = zones
    return d


def seats(util=0.9, n=4, queue=False):
    zs = [{"id": f"S{i}", "name": f"Seat {i}", "kind": "seat",
           "metrics": {"utilization": util, "turnover_per_hour": 2.0}} for i in range(n)]
    if queue:
        zs.append({"id": "Q", "name": "Till", "kind": "queue", "metrics": {},
                   "queue": {"waits": [], "mean_wait_sec": 90, "median_wait_sec": 80, "max_wait_sec": 300,
                             "max_length": 5}})
    return {"zones": zs, "summary": {"seats_tables": n, "mean_utilization": util, "occupied_at_end": n - 1,
                                     "free_at_end": 1, "median_session_sec": 400.0, "unassigned_share": 0.05}}


def by_id(g):
    return {s["id"]: s for s in g["sections"]}


def test_a_shop_reads_as_walk_through_and_every_section_has_evidence():
    g = shop_guide(doc())
    assert g["setting"]["kind"] == "walk-through"
    s = by_id(g)
    assert {"busy", "arrivals", "stopping", "where", "who", "trust"} <= set(s)
    for sec in g["sections"]:
        assert sec["headline"].endswith(".") and sec["explain"]
        if sec["id"] != "trust":
            assert sec["numbers"] and sec["confidence"] in ("solid", "rough", "unreliable") and sec["why"]
    assert s["busy"]["confidence"] == "solid"
    assert "14:02" in s["busy"]["headline"]                 # the busiest moment as a time of day
    assert g["glossary"]


def test_people_already_there_are_not_arrivals():
    g = shop_guide(doc(first=lambda i: 0.0 if i < 10 else 30.0 + i))
    a = by_id(g)["arrivals"]
    assert a["numbers"]["already_there"] == 10
    assert "10 were already there" in a["headline"]


def test_a_seated_room_with_split_tracks_distrusts_arrivals_and_says_why():
    d = doc(visitors=231, dwell=200.0, peak=45, avg=38.0, zones=seats(util=0.7, n=50))
    g = shop_guide(d)
    assert g["setting"]["kind"] == "seated"
    s = by_id(g)
    assert s["arrivals"]["confidence"] == "unreliable"
    assert "5.1 per person" in s["arrivals"]["why"]
    # no staffing advice built on a number just called unreliable
    assert not any("busiest" in x for x in s["arrivals"]["do"])
    assert any("distrust totals" in x for x in s["trust"]["do"])
    assert g["sections"].index(s["seats"]) < g["sections"].index(s["arrivals"])   # seats first when seated


def test_full_seats_and_long_waits_get_concrete_suggestions():
    g = shop_guide(doc(zones=seats(util=0.92, queue=True)))
    s = by_id(g)
    assert any("close to full" in x for x in s["seats"]["do"])
    assert any("second counter" in x for x in s["queues"]["do"])


def test_unsure_people_are_counted_as_unsure_not_guessed():
    g = shop_guide(doc(gender=False))
    who = by_id(g)["who"]
    assert who["numbers"]["gender_unknown_share"] == 1.0
    assert any("not sure" in x for x in who["do"])


def test_an_empty_run_still_gives_a_guide():
    d = doc(visitors=0)
    g = shop_guide(d)
    assert setting_of(d)["kind"] == "walk-through"
    assert by_id(g)["busy"]


def test_guide_endpoint_writes_one_for_old_runs(tmp_path, monkeypatch):
    import app.server as server
    from fastapi.testclient import TestClient
    run = tmp_path / "old"
    run.mkdir()
    (run / "analytics.json").write_text(json.dumps(doc()), encoding="utf-8")
    monkeypatch.setattr(server, "RUNS", tmp_path)
    r = TestClient(server.app).get("/api/runs/old/guide")
    assert r.status_code == 200 and r.json()["sections"]
    assert TestClient(server.app).get("/api/runs/missing/guide").status_code == 404


def test_tiny_samples_and_cut_off_stays_are_called_out_not_overread():
    d = doc(visitors=4, duration=15.0, dwell=15.0, first=lambda i: 0.0)
    d["distributions"]["zones"] = {"Left": {"visitors": 0, "avg_dwell": 0}, "Center": {"visitors": 3, "avg_dwell": 15},
                                   "Right": {"visitors": 1, "avg_dwell": 15}}
    s = by_id(shop_guide(d))
    assert s["stopping"]["confidence"] == "unreliable" and "Too short to tell" in s["stopping"]["headline"]
    assert s["where"]["confidence"] == "unreliable" and "3 in the middle" in s["where"]["headline"]
    assert "%" not in s["who"]["headline"] and s["who"]["confidence"] == "unreliable"
