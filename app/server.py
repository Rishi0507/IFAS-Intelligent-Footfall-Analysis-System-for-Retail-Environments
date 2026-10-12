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
import shutil
import sys
import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
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
    d = (RUNS / run_id).resolve()
    if not str(d).startswith(str(RUNS.resolve())) or not d.is_dir():
        raise HTTPException(404, f"No such run: {run_id}")
    return d


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
                     verifier: str = Form("yolov8m.pt")):
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
