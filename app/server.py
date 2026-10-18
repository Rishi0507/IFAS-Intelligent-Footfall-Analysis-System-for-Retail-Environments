"""Dashboard server: serves the UI, lists runs, and processes uploaded video.

    pip install fastapi uvicorn python-multipart
    python -m app.server                     # http://127.0.0.1:8000

A "run" is just a directory under outputs/runs/<id>/ holding analytics.json and
(optionally) video.mp4 + poster.jpg. Anything the pipeline writes there shows up
in the dashboard's run picker, so you can also generate runs offline with
`python -m src.inference.pipeline` and never start this server.
"""
from __future__ import annotations
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `src.inference` imports under any launcher
RUNS = ROOT / "outputs" / "runs"
STATIC = Path(__file__).resolve().parent / "static"
RUNS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="IFAS Dashboard")
JOBS: dict[str, dict] = {}
_lock = threading.Lock()


def _run_dir(run_id: str) -> Path:
    # A run is a direct child of RUNS. A prefix test on the path string would
    # also let `../runs-other` through, so compare the parent instead.
    root = RUNS.resolve()
    d = (root / run_id).resolve()
    if d.parent != root or not d.is_dir():
        raise HTTPException(404, f"No such run: {run_id}")
    return d


ANNOTATOR = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
GENDERS = {"female", "male", "unsure"}
AGES = {"Age16-30", "Age31-45", "Age46-60", "AgeAbove61", "unsure"}


def _annotators(d: Path):
    return sorted(p.stem[len("labels_"):] for p in d.glob("labels_*.json"))


@app.get("/api/runs")
def list_runs():
    runs = []
    for d in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        # (sample runs are pushed to the end below, so real footage opens first)
        doc_path = d / "analytics.json"
        if not (d.is_dir() and doc_path.exists()):
            continue
        with open(doc_path, encoding="utf-8") as f:
            doc = json.load(f)
        runs.append({
            "id": d.name,
            "source": doc["meta"].get("source", d.name),
            "generated_at": doc.get("generated_at"),
            "visitors": doc["kpis"]["visitors"],
            "duration_sec": doc["meta"].get("duration_sec"),
            "demo": bool(doc["meta"].get("demo")),
            "site": doc["meta"].get("site"),
            "camera": doc["meta"].get("camera"),
            "has_video": (d / "video.mp4").exists(),
            "has_poster": (d / "poster.jpg").exists(),
            "has_crops": (d / "crops.jsonl").exists(),
            "annotators": _annotators(d),
        })
    runs.sort(key=lambda r: r["demo"])
    return runs


@app.get("/api/runs/{run_id}/analytics.json")
def get_analytics(run_id: str):
    return FileResponse(_run_dir(run_id) / "analytics.json", media_type="application/json")


@app.get("/api/runs/{run_id}/video.mp4")
def get_video(run_id: str, request: Request):
    path = _run_dir(run_id) / "video.mp4"
    if not path.exists():
        raise HTTPException(404, "This run has no annotated video.")
    return _ranged(path, request, "video/mp4")


def _ranged(path: Path, request: Request, media_type: str, chunk=1024 * 512):
    """Serve a file with Range support - without it the player cannot seek."""
    size = path.stat().st_size
    header = request.headers.get("range")
    if not header or not header.startswith("bytes="):
        return FileResponse(path, media_type=media_type,
                            headers={"accept-ranges": "bytes"})
    raw_start, _, raw_end = header[6:].partition("-")
    start = int(raw_start or 0)
    end = int(raw_end) if raw_end else size - 1
    end = min(end, size - 1)
    if start > end:
        return Response(status_code=416, headers={"content-range": f"bytes */{size}"})

    def stream():
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                data = f.read(min(chunk, left))
                if not data:
                    break
                left -= len(data)
                yield data

    return StreamingResponse(stream(), status_code=206, media_type=media_type, headers={
        "content-range": f"bytes {start}-{end}/{size}",
        "content-length": str(end - start + 1),
        "accept-ranges": "bytes",
    })


@app.get("/api/runs/{run_id}/poster.jpg")
def get_poster(run_id: str):
    path = _run_dir(run_id) / "poster.jpg"
    if not path.exists():
        raise HTTPException(404, "No poster frame.")
    return FileResponse(path, media_type="image/jpeg")


# ------------------------------------------------------------------ labelling
@app.get("/api/runs/{run_id}/crops")
def get_crops(run_id: str):
    """The people of a run, each with the crops the pipeline classified.

    Model predictions are left out on purpose: annotators must not see them.
    """
    d = _run_dir(run_id)
    path = d / "crops.jsonl"
    if not path.exists():
        raise HTTPException(404, "This run has no exported crops. Process it with crop export on.")
    people = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            person = people.setdefault(r["track"], {"track": r["track"], "crops": []})
            person["crops"].append({
                "frame": r["frame"], "t": r["t"],
                "url": f"/api/runs/{quote(run_id)}/{r['crop']}",
            })
    ordered = sorted(people.values(), key=lambda p: (p["crops"][0]["t"], p["track"]))
    return {"run": run_id, "people": ordered, "annotators": _annotators(d)}


@app.get("/api/runs/{run_id}/crops/{track}/{name}")
def get_crop_image(run_id: str, track: int, name: str):
    # PNG since crops became lossless; runs exported earlier have JPEG crops.
    if not re.fullmatch(r"\d{6}\.(png|jpg)", name):
        raise HTTPException(404, "No such crop.")
    path = _run_dir(run_id) / "crops" / str(track) / name
    if not path.exists():
        raise HTTPException(404, "No such crop.")
    return FileResponse(path, media_type="image/png" if name.endswith(".png") else "image/jpeg")


@app.get("/api/runs/{run_id}/labels/{annotator}")
def get_labels(run_id: str, annotator: str):
    if not ANNOTATOR.fullmatch(annotator):
        raise HTTPException(400, "Annotator names use letters, digits, _ and - only.")
    path = _run_dir(run_id) / f"labels_{annotator}.json"
    if not path.exists():
        return {"run": run_id, "annotator": annotator, "labels": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/runs/{run_id}/labels")
def save_labels(run_id: str, body: dict = Body(...)):
    """Write labels_<annotator>.json: one file per annotator, so agreement can be measured."""
    d = _run_dir(run_id)
    annotator = str(body.get("annotator", ""))
    if not ANNOTATOR.fullmatch(annotator):
        raise HTTPException(400, "Annotator names use letters, digits, _ and - only.")
    labels = body.get("labels")
    if not isinstance(labels, dict):
        raise HTTPException(400, "labels must be an object keyed by track id.")
    clean = {}
    for track, lab in labels.items():
        if not str(track).isdigit() or not isinstance(lab, dict):
            raise HTTPException(400, f"Bad entry for track {track!r}.")
        gender, age = lab.get("gender"), lab.get("age")
        if gender not in GENDERS and gender is not None:
            raise HTTPException(400, f"Track {track}: gender must be one of {sorted(GENDERS)}.")
        if age not in AGES and age is not None:
            raise HTTPException(400, f"Track {track}: age must be one of {sorted(AGES)}.")
        note = str(lab.get("note") or "")[:500]
        clean[str(int(track))] = {"gender": gender, "age": age, "note": note}
    doc = {"run": run_id, "annotator": annotator, "labels": clean}
    # Write then rename, so a crash never leaves half a file behind.
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".labels_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, d / f"labels_{annotator}.json")
    return {"saved": len(clean)}


@app.get("/api/trends")
def get_trends(include_demo: bool = False, site: str = "", camera: str = ""):
    """Aggregate every run into weekday / hour / season views."""
    from src.inference.trends import build_trends
    runs = []
    for d in sorted(RUNS.iterdir()):
        doc_path = d / "analytics.json"
        if d.is_dir() and doc_path.exists():
            with open(doc_path, encoding="utf-8") as f:
                runs.append((d.name, json.load(f)))
    return build_trends(runs, include_demo=include_demo,
                        site=site or None, camera=camera or None)


def _process(job_id, video_path, run_dir, opts):
    from src.inference.pipeline import process_video

    def progress(**kw):
        with _lock:
            JOBS[job_id].update(kw)

    try:
        process_video(video_path, run_dir, progress=progress, **opts)
        with _lock:
            JOBS[job_id].update(stage="done", pct=100.0, run_id=run_dir.name)
    except BaseException as exc:  # BaseException: a SystemExit here must not vanish
                                 # into a dead thread with the job stuck at "queued"
        with _lock:
            JOBS[job_id].update(stage="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        Path(video_path).unlink(missing_ok=True)
        # A failed job leaves no analytics; do not keep its half-written folder.
        if not (run_dir / "analytics.json").exists():
            shutil.rmtree(run_dir, ignore_errors=True)


@app.post("/api/runs")
async def create_run(background: BackgroundTasks, video: UploadFile = File(...),
                     gender_repo: str = Form(""), gender_ckpt: str = Form(""),
                     age_bundle: str = Form(""), stride: int = Form(2),
                     det_conf: float = Form(0.4), write_video: bool = Form(True),
                     max_frames: int = Form(0), captured_at: str = Form(""),
                     site: str = Form(""), camera: str = Form(""),
                     verifier: str = Form("yolov8m.pt"), export_crops: bool = Form(False)):
    run_id = f"{Path(video.filename).stem[:40] or 'run'}-{uuid.uuid4().hex[:6]}"
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / f"_input{Path(video.filename).suffix or '.mp4'}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(video.file, f)

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"id": job_id, "run_id": run_id, "stage": "queued", "pct": 0.0}
    background.add_task(_process, job_id, str(tmp), run_dir, {
        "gender_repo": gender_repo or None,
        "gender_ckpt": gender_ckpt or None,
        "age_bundle": age_bundle or None,
        "stride": max(1, stride),
        "det_conf": det_conf,
        "write_video": write_video,
        "max_frames": max_frames or None,
        "source_name": video.filename,
        "captured_at": captured_at or None,
        "site": site or None,
        "camera": camera or None,
        "verifier_weights": None if verifier in ("", "none") else verifier,
        "export_crops": export_crops,
    })
    return JSONResponse({"job": job_id, "run_id": run_id}, status_code=202)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "No such job")
    return JOBS[job_id]


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
