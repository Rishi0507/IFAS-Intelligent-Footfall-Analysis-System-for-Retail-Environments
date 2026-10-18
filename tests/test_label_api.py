"""Labelling API and page (task A2/A7 in docs/research-plan.md).

The API tests use FastAPI's test client. The last test drives the real page in
headless Chrome through Playwright, labels two people with the keyboard and
checks the file that lands in the run folder; it is skipped when Playwright or
a Chrome/Chromium build is not available.
"""
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
Image = pytest.importorskip("PIL.Image")

from fastapi.testclient import TestClient  # noqa: E402

import app.server as server  # noqa: E402


def _make_run(runs, name="shop-cam1", tracks=(3, 7, 12)):
    d = runs / name
    lines = []
    for i, track in enumerate(tracks):
        for f in range(2):
            frame = 10 * i + 4 * f
            ext = "jpg" if i == 0 else "png"      # runs exported before PNG crops have JPEGs
            rel = f"crops/{track}/{frame:06d}.{ext}"
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (30, 70), (40 * i, 90, 160)).save(d / rel)
            lines.append({"track": track, "frame": frame, "t": frame / 10, "box": [0, 0, 30, 70],
                          "det_conf": 0.9, "frame_w": 320, "frame_h": 240, "crop": rel, "flow": None,
                          "overlap": 0.0, "cues": {}, "pred": {"gender": {"male": 0.9, "female": 0.1}}})
    (d / "crops.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    (d / "analytics.json").write_text(json.dumps({
        "meta": {"source": "shop.mp4", "duration_sec": 3.0}, "kpis": {"visitors": len(tracks)}}))
    return d


@pytest.fixture
def runs(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(server, "RUNS", root)
    return root


@pytest.fixture
def client(runs):
    return TestClient(server.app)


def test_runs_list_says_which_have_crops(runs, client):
    _make_run(runs)
    (runs / "plain").mkdir()
    (runs / "plain" / "analytics.json").write_text(json.dumps({"meta": {}, "kpis": {"visitors": 0}}))
    by = {r["id"]: r for r in client.get("/api/runs").json()}
    assert by["shop-cam1"]["has_crops"] is True and by["plain"]["has_crops"] is False
    assert by["shop-cam1"]["annotators"] == []


def test_crops_endpoint_groups_by_person_and_hides_predictions(runs, client):
    _make_run(runs)
    doc = client.get("/api/runs/shop-cam1/crops").json()
    assert [p["track"] for p in doc["people"]] == [3, 7, 12]
    assert len(doc["people"][0]["crops"]) == 2
    assert "pred" not in json.dumps(doc) and "probs" not in json.dumps(doc)
    img = client.get(doc["people"][1]["crops"][0]["url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    old = client.get(doc["people"][0]["crops"][0]["url"])
    assert old.status_code == 200 and old.headers["content-type"] == "image/jpeg"


def test_crops_missing_is_a_clear_404(runs, client):
    (runs / "nocrops").mkdir()
    r = client.get("/api/runs/nocrops/crops")
    assert r.status_code == 404 and "crop export" in r.json()["detail"]


def test_save_and_load_labels(runs, client):
    d = _make_run(runs)
    body = {"annotator": "rishi", "labels": {"3": {"gender": "female", "age": "Age31-45", "note": "hat"},
                                             "7": {"gender": "unsure", "age": None}}}
    assert client.post("/api/runs/shop-cam1/labels", json=body).json() == {"saved": 2}
    on_disk = json.loads((d / "labels_rishi.json").read_text())
    assert on_disk == {"run": "shop-cam1", "annotator": "rishi",
                       "labels": {"3": {"gender": "female", "age": "Age31-45", "note": "hat"},
                                  "7": {"gender": "unsure", "age": None, "note": ""}}}
    assert client.get("/api/runs/shop-cam1/labels/rishi").json()["labels"]["3"]["note"] == "hat"
    assert client.get("/api/runs/shop-cam1/labels/yash").json()["labels"] == {}
    assert client.get("/api/runs").json()[0]["annotators"] == ["rishi"]
    assert not list(d.glob(".labels_*"))                       # the temporary file is gone


@pytest.mark.parametrize("body", [
    {"annotator": "../evil", "labels": {}},
    {"annotator": "a b", "labels": {}},
    {"annotator": "rishi", "labels": {"3": {"gender": "other"}}},
    {"annotator": "rishi", "labels": {"3": {"age": "Age99"}}},
    {"annotator": "rishi", "labels": {"x": {"gender": "male"}}},
    {"annotator": "rishi", "labels": []},
])
def test_bad_labels_are_refused(runs, client, body):
    _make_run(runs)
    assert client.post("/api/runs/shop-cam1/labels", json=body).status_code == 400


def test_path_guard(runs, client, tmp_path):
    _make_run(runs)
    sibling = tmp_path / "runs-other"                           # shares the "runs" prefix
    sibling.mkdir()
    (sibling / "analytics.json").write_text("{}")
    assert client.get("/api/runs/..%2Fruns-other/analytics.json").status_code == 404
    assert client.get("/api/runs/shop-cam1/crops/3/..%2F..%2Fcrops.jsonl").status_code == 404
    assert client.get("/api/runs/shop-cam1/crops/3/notajpeg.txt").status_code == 404


# ------------------------------------------------------------ headless Chrome
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(runs):
    uvicorn = pytest.importorskip("uvicorn")
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=5)


def _browser(p):
    for kw in ({"channel": "chrome"}, {"channel": "msedge"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:                                       # noqa: BLE001 - try the next one
            continue
    pytest.skip("no Chrome or Chromium build available to Playwright")


def test_label_page_in_headless_chrome(runs, live_server):
    sync_api = pytest.importorskip("playwright.sync_api")
    d = _make_run(runs)
    with sync_api.sync_playwright() as p:
        browser = _browser(p)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{live_server}/label.html?run=shop-cam1")
        page.wait_for_selector("#strip img")
        assert page.inner_text("#person-title") == "Person 3"

        page.fill("#annotator", "rishi")
        page.press("#annotator", "Enter")
        page.locator("#annotator").blur()
        page.keyboard.press("f")                              # person 3: female
        page.keyboard.press("2")                              # 31 to 45, then it moves on
        page.wait_for_function("document.querySelector('#person-title').textContent === 'Person 7'")
        page.keyboard.press("m")                              # person 7: male
        page.keyboard.press("0")                              # age unsure
        page.wait_for_function("document.querySelector('#person-title').textContent === 'Person 12'")
        page.wait_for_function("document.querySelector('#save-state').textContent === 'Saved'")
        assert page.inner_text("#progress") == "2 of 3 labelled"
        browser.close()
    assert not errors, errors
    labels = json.loads((d / "labels_rishi.json").read_text())["labels"]
    assert labels == {"3": {"gender": "female", "age": "Age31-45", "note": ""},
                      "7": {"gender": "male", "age": "unsure", "note": ""}}
