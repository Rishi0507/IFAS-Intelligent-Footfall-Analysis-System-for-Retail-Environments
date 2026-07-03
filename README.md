# IPD — Retail CCTV Gender Analytics Pipeline

**Version 9.3** · Person detection → gender classification → tracking → footfall counting → reporting, all from a single video file.

> ⚠️ **Age is intentionally NOT classified in this version.** Multi-camera fusion and persistent cross-video ReID are also out of scope (see [Roadmap / What's Not Included](#whats-not-included)).

---

## Table of Contents

1. [What This Is](#what-this-is)
2. [Key Features](#key-features)
3. [What Makes This Different (USP)](#what-makes-this-different-usp)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Repo Layout](#repo-layout)
6. [Installation](#installation)
7. [Ways to Run It](#ways-to-run-it)
   - [1. Command line (local)](#1-command-line-local)
   - [2. Python API (programmatic)](#2-python-api-programmatic)
   - [3. Google Colab / Jupyter notebook](#3-google-colab--jupyter-notebook)
   - [4. Running the test suite](#4-running-the-test-suite)
8. [Configuration Reference](#configuration-reference)
9. [CLI Argument Reference](#cli-argument-reference)
10. [Output Files](#output-files)
11. [Tuning for Speed vs. Accuracy](#tuning-for-speed-vs-accuracy)
12. [How Gender Is Decided (Ensemble Logic)](#how-gender-is-decided-ensemble-logic)
13. [Troubleshooting](#troubleshooting)
14. [What's Not Included](#whats-not-included)
15. [License](#license)

---

## What This Is

IPD is a self-contained Python pipeline that takes a **retail-store CCTV video** and turns it into structured analytics:

- How many people entered / exited, and when
- A male / female / uncertain breakdown of footfall
- A per-person (per-track) record of gender
- Annotated sample frames you can eyeball to sanity-check the model
- CSV, JSON, and PNG reports ready to drop into a dashboard or slide deck

It runs entirely offline after the first model download, on CPU or GPU, from the command line, from Python, or from a Colab notebook.

---

## Key Features

- **Person detection** — YOLOv8n (`ultralytics`), filtered by a minimum box size and aspect-ratio window so shelving, posters, and mannequins don't get counted as people.
- **Dual-signal gender classification**
  - *Face-based*: YuNet face detector (OpenCV) → crop → a HuggingFace ViT gender classifier (`rizvandwiki/gender-classification-2`).
  - *Body-based*: CLIP zero-shot classification (`openai/clip-vit-base-patch32`) against male/female text prompts — works even when the face isn't visible (back-turned, low-res, occluded).
- **Confidence-aware ensemble** — combines face and body signals with an explicit decision tree (see [below](#how-gender-is-decided-ensemble-logic)), and will honestly return `"uncertain"` rather than force a guess.
- **Face landmark validation** — rejects false-positive "faces" detected on clothing patterns, logos, or the backs of heads, by checking eye/nose/mouth geometry is anatomically plausible.
- **Containment-aware deduplication** — a custom NMS that removes not only high-IoU duplicate boxes but also *fully-nested* boxes (e.g., a torso detected inside a full-body box) that standard IoU-only NMS misses.
- **Multi-object tracking** — IoU matching with a **size-scaled centroid fallback**, so tracks survive brief IoU misses (fast motion, partial occlusion) without spawning duplicate IDs.
- **Short-term re-identification (ReID)** — when a track is lost and a *new* detection appears, its CLIP body embedding is checked against a bank of recently-expired tracks (cosine similarity ≥ threshold) so the same shopper walking behind a shelf and back out keeps their original ID instead of getting double-counted.
- **Gender "locking"** — once a track has been confidently classified male/female, that label is locked for the rest of its lifetime, preventing gender from flip-flopping frame to frame as viewing angle changes.
- **Line-crossing footfall counting** — a configurable virtual line (any orientation, not just horizontal) with cross-product side detection determines entries vs. exits.
- **Time-bucketed aggregation** — footfall and gender counts rolled up into configurable buckets (default: 15 minutes) for daily/hourly reporting.
- **Automatic report generation** — CSV (daily bucket report, per-person report), a gender demographics pie chart, an entries-vs-exits-by-hour bar chart, and a full machine-readable JSON summary.
- **Annotated frame sampling** — every N frames, a copy of the frame with tracked boxes + gender labels burned in is saved, so you can visually verify (or feed to a VLM for automated QA) without re-rendering the whole video.
- **Fully config-driven** — every threshold (detection confidence, box size filters, tracker IoU/centroid distances, ReID cosine threshold, uncertain-ratio cutoff, footfall line geometry, aggregation bucket size, frame sub-sampling rate) lives in one YAML-overridable config dict — no hunting through code to tune behavior.
- **Frame sub-sampling** — process every Nth frame for a near-linear speedup with minimal accuracy loss on typical walking speeds.
- **Resumable / chunked processing** — `--start-frame` / `--max-frames` / `--end-frame` let you process a video in segments (useful for very long recordings or distributed processing).

---

## What Makes This Different (USP)

This isn't a generic "run YOLO and call it a day" script — it's a *hardened* v9.3 that specifically fixes a set of real bugs found in earlier iterations of the same pipeline:

| Problem | Naive / v8 behavior | IPD v9.3 fix |
|---|---|---|
| YuNet confidence score | Read from the wrong output column (`row[4]`, actually the right-eye X coordinate) | Reads the correct column (`row[14]`) |
| False face positives | Clothing patterns / logos / back-of-head detected as "faces" | Geometric landmark validation (eye-above-nose, nose-above-mouth, plausible eye/mouth width ratios) rejects them |
| Ambiguous gender signal | Defaulted to `"male"` when CLIP was wishy-washy and no face was visible | Explicitly returns `"uncertain"` — no silent bias toward one label |
| Same person re-detected after occlusion | Assigned a brand-new track ID every time → inflated footfall counts | ReID bank re-links the same ID via CLIP embedding cosine similarity |
| Nested duplicate boxes | Standard IoU-only NMS kept both a full-body box and a smaller box mostly contained inside it | Containment-aware dedup also suppresses "mostly-inside" boxes |
| Gender flicker | Track's gender label could change every frame as the person rotates | Gender is locked once a confident label is reached |

In short: **the interesting engineering here is in the failure modes it closes**, not just the detection/classification models it wraps.

---

## Pipeline Architecture

```
video frame
   |
   v
[1] YOLOv8n person detector (conf >= 0.20, native res)
   |     + aspect-ratio / min-size filter (reject shelves, posters, noise)
   v
[2] dedup_boxes  (IoU NMS + containment-aware suppression)
   v
[3] YuNet face detector (row[14] score bug fixed)
   |     + landmark geometry validation
   v
[4] FaceViT gender classifier  -> face gender + confidence
[5] CLIP zero-shot classifier  -> body gender + confidence ratio + embedding
   v
[6] ensemble_gender()  — CLIP-primary decision tree, explicit 'uncertain' branch
   v
[7] Tracker (IoU + size-scaled centroid matching, persist_frames grace period)
   |     + ReIDBank: unmatched detections queried against recently-expired
   |       tracks by CLIP embedding cosine similarity
   v
[8] LineCounter — entry/exit via cross-product side-of-line detection
[9] TimeBucketAggregator — rolls events into fixed-size time buckets
   v
[10] ReportGenerator -> daily_report.csv, per_person.csv,
                         demographics_pie.png, footfall_by_hour.png,
                         summary.json
```

Every stage is a separate, independently testable module (see [Repo Layout](#repo-layout)) — the orchestration itself lives in `src/pipeline.py`'s `StoreAnalytics` class.

---

## Repo Layout

```
IPD/
├── cctv_gender.py           # CLI entry point
├── gender_pipeline.ipynb    # self-contained Colab/Jupyter notebook (mirrors src/)
├── setup.py                 # pip-installable package (console script: ipd-cctv)
├── requirements.txt         # pinned dependency list
├── README.md
├── docs/
│   └── architecture.md      # pipeline diagram + bug-fix changelog
├── examples/
│   └── config.example.yaml  # copy-and-edit starter config
├── outputs/                 # default output root (empty, .gitkeep)
├── src/
│   ├── __init__.py
│   ├── config.py            # DEFAULT_CONFIG + YAML loader/merger
│   ├── geometry.py          # iou, contains, dedup_boxes, validate_face_landmarks, centroid
│   ├── models.py            # YOLOPersonDetector, YuNetFaceDetector, FaceViTGender, CLIPGender
│   ├── ensemble.py           # ensemble_gender() decision tree
│   ├── tracking.py          # Track (dataclass), Tracker, ReIDBank
│   ├── counting.py          # LineCounter, TimeBucketAggregator
│   ├── reports.py           # ReportGenerator (CSV/PNG/JSON writer)
│   ├── annotate.py          # annotate_frame() — draws boxes + labels
│   └── pipeline.py          # StoreAnalytics — orchestrates all of the above
└── tests/
    ├── __init__.py
    ├── test_geometry.py
    ├── test_ensemble.py
    ├── test_reid.py
    ├── test_tracker.py
    └── test_footfall.py
```

---

## Installation

### Requirements
- Python **3.9+**
- ~2–3 GB free disk (model weights download on first run)
- Internet access on first run (to download YOLOv8n weights, the YuNet ONNX model, and the HuggingFace models)

### CPU install (recommended for most local machines)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# CPU-only torch wheel — much smaller than the default GPU build:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### GPU install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Then set `detector.device: cuda` in your config (see [Configuration Reference](#configuration-reference)).

### Install as a package (optional)

```bash
pip install -e .
```
This registers a console script, so you can invoke `ipd-cctv` from anywhere instead of `python cctv_gender.py`.

---

## Ways to Run It

There are four ways to run this pipeline, depending on your setup.

### 1. Command line (local)

The simplest way — point it at a video and an output folder:

```bash
python cctv_gender.py --video input.mp4 --output outputs/run1/
```

With a custom config and frame-range controls:

```bash
python cctv_gender.py \
  --video input.mp4 \
  --output outputs/run1/ \
  --config my_config.yaml \
  --dump-every 15 \
  --max-frames 5000 \
  --start-frame 1 \
  --end-frame 5000
```

If installed via `pip install -e .`:

```bash
ipd-cctv --video input.mp4 --output outputs/run1/
```

This prints progress to stdout as it processes (frame number, live entry/exit counts, ETA) and ends with a JSON summary block.

### 2. Python API (programmatic)

Use it directly inside another Python script or service — useful for batch jobs, integrating into a larger data pipeline, or building a custom UI on top:

```python
from src.config import load_config
from src.pipeline import StoreAnalytics

cfg = load_config("my_config.yaml")   # or load_config(None) for all defaults
cfg["pipeline"]["process_every_n_frames"] = 3   # override anything in code too

sa = StoreAnalytics(cfg)
summary = sa.process_video(
    video_path="input.mp4",
    out_dir="outputs/run1/",
    dump_annotated_every=30,
    process_every_n=cfg["pipeline"]["process_every_n_frames"],
    max_frames=None,
    start_frame=1,
    end_frame=None,
)

print(summary["entries"], summary["exits"], summary["gender_counts"])
```

`summary` is a plain dict — everything that gets written to `summary.json` is also returned in-memory, so you can act on results without touching the filesystem.

### 3. Google Colab / Jupyter notebook

Open `gender_pipeline.ipynb`:

- **In Colab**: upload the notebook (or open directly from GitHub), run all cells top to bottom. Cell 1 installs all dependencies; a later cell pops the native file-upload widget for your video; the final cells display the pie chart / bar chart inline and zip up the output folder for download.
- **In local Jupyter**: run all cells; skip the Colab-only upload/download cells (they're wrapped in `try/except ImportError` and will fall back to a local file path / local output folder automatically).

The notebook is intentionally **self-contained** — every class from `src/` is redefined inline in its own cell, so it has zero dependency on the rest of the repo and can be shared as a single file.

> Note: because the notebook duplicates `src/`, if you make a fix to the `.py` modules, remember to port it into the notebook manually (or vice versa) if you want both to stay in sync.

### 4. Running the test suite

```bash
pip install pytest
pytest tests/ -v
```

The test suite covers `geometry`, `ensemble`, `tracking`/`ReIDBank`, and `counting` in isolation with synthetic inputs — it does **not** require `torch`, `ultralytics`, or `transformers` to be installed, so it's fast (well under a second) and safe to run in a lightweight CI environment.

---

## Configuration Reference

Copy `examples/config.example.yaml`, edit what you need, and pass it with `--config`. Anything you don't override falls back to `DEFAULT_CONFIG` in `src/config.py`. Full key reference:

| Section | Key | Default | Meaning |
|---|---|---|---|
| `detector` | `yolo_model` | `yolov8n.pt` | Ultralytics YOLO weights to use for person detection |
| | `conf` | `0.20` | Minimum detection confidence |
| | `device` | `cpu` | `cpu` or `cuda` |
| | `imgsz` | `null` | Inference resolution override (`null` = native) |
| | `min_box_w` / `min_box_h` | `20` / `30` | Reject boxes smaller than this (px) |
| | `aspect_min` / `aspect_max` | `0.15` / `6.0` | Reject boxes outside this width/height ratio |
| `face_detector` | `model_path` | `models/face_detection_yunet_2023mar.onnx` | Auto-downloaded if missing |
| | `conf_threshold` | `0.45` | YuNet minimum face score |
| | `nms_threshold` | `0.30` | YuNet internal NMS threshold |
| | `top_k` | `5000` | Max candidate faces per frame before NMS |
| `face_gender` | `model` | `rizvandwiki/gender-classification-2` | HuggingFace face-gender ViT model |
| | `conf_threshold` | `0.70` | Minimum face-classifier confidence to be treated as "confident" |
| `clip` | `model` | `openai/clip-vit-base-patch32` | CLIP model for body-based classification |
| | `gender_male_prompts` / `gender_female_prompts` | see YAML | Text prompts CLIP scores images against |
| | `uncertain_ratio` | `1.8` | Below this male/female score ratio, CLIP is "wishy-washy" |
| | `uncertain_face_threshold` | `0.70` | Face-confidence cutoff used inside the ensemble decision |
| `tracker` | `iou_threshold` | `0.30` | Minimum IoU to match a detection to an existing track |
| | `centroid_max_dist` | `80.0` | Fallback max centroid distance (px) when IoU match fails |
| | `persist_frames` | `3` | Frames a track survives with no matching detection before expiring |
| | `size_scale` | `0.30` | Allowed box-size change (as a fraction) for centroid fallback matching |
| `reid` | `enabled` | `true` | Turn short-term re-identification on/off |
| | `cosine_threshold` | `0.75` | Minimum embedding similarity to re-link an expired track |
| | `expiry_seconds` | `30.0` | How long an expired track stays eligible for ReID matching |
| | `max_bank_size` | `200` | Max number of expired-track embeddings kept in memory |
| `footfall` | `line` | `[[0.0, 0.7], [1.0, 0.7]]` | Normalised `[x, y]` endpoints of the counting line |
| | `direction` | `down` | Which crossing direction counts as an "entry" |
| `aggregator` | `bucket_seconds` | `900` | Time-bucket width for the daily report (900 = 15 min) |
| `pipeline` | `process_every_n_frames` | `1` | Process every Nth frame (speed/accuracy tradeoff) |
| | `max_frames` | `null` | Hard cap on frames processed |
| | `dump_annotated_every` | `30` | Save an annotated frame every N processed frames |
| | `dump_dir` | `annotated_frames` | Subfolder (under `--output`) for annotated frame samples |

---

## CLI Argument Reference

| Flag | Required | Default | Description |
|---|---|---|---|
| `--video` | ✅ | — | Path to the input video file |
| `--output` | ✅ | — | Output directory (created if missing) |
| `--config` | ❌ | `None` | Path to a YAML config file to override defaults |
| `--dump-every` | ❌ | `30` | Save an annotated frame every N processed frames |
| `--max-frames` | ❌ | `None` | Cap total frames processed |
| `--start-frame` | ❌ | `1` | First frame to process (1-indexed) |
| `--end-frame` | ❌ | `None` | Last frame to process |

---

## Output Files

All written under `<output>/`:

| File | Description |
|---|---|
| `daily_report.csv` | One row per time bucket: entries, exits, male/female/uncertain counts, unique tracks seen |
| `per_person.csv` | One row per unique track ID with its final (locked) gender label |
| `demographics_pie.png` | Overall male / female / uncertain split, as a pie chart |
| `footfall_by_hour.png` | Entries vs. exits per hour, as a grouped bar chart |
| `summary.json` | Full machine-readable summary — same data returned in-memory by `process_video()` |
| `annotated_frames/frame_NNNNNN.jpg` | Sampled frames with tracked boxes + gender + face-confidence labels burned in, for visual QA or VLM verification |

---

## Tuning and Fine-Tuning Guide

This section explains how to fine-tune the pipeline's parameters to optimize performance (speed and accuracy) under different retail CCTV scenarios.

### 1. General Speed vs. Accuracy Tuning
* **`pipeline.process_every_n_frames`** (default: `1`): Set to `3` for a ~3x speedup with minimal accuracy loss for typical walking speeds. Raise it further (e.g., `5`) for very slow-moving crowds, and keep it at `1` for fast-moving or dense scenes where people cross the counting line quickly.
* **`detector.device`** (default: `"cpu"`): Change to `"cuda"` if you have an NVIDIA GPU. This is the single largest performance improvement.

### 2. CCTV Camera Placement & Scale Tuning
* **High-Mounted or Wide-Angle Cameras** (people appear very small in the frame):
  * **`detector.min_box_w`** & **`detector.min_box_h`** (default: `20` / `30`): If shoppers at the back of the store are missed, reduce these to `10` and `15`.
  * **`detector.conf`** (default: `0.20`): Lower it slightly to `0.15` to catch faint or distant detections, though this may increase false positives.
* **Close-up / Eye-Level Cameras**:
  * If the camera is close to the entrance and people appear very large, increase **`detector.min_box_w`** & **`detector.min_box_h`** (e.g., `50` / `80`) to filter out background movements and noise.

### 3. Handling Occlusion (Shoppers walking behind displays/shelves)
* **`reid.enabled`** (default: `true`): Keeps track IDs consistent when a shopper is temporarily hidden by columns or display stands.
* **`reid.expiry_seconds`** (default: `30.0`): If shoppers typically disappear behind displays for longer periods, increase this value (e.g., to `60.0`).
* **`reid.cosine_threshold`** (default: `0.75`): 
  * **Lower it** (e.g., `0.70`) to make ReID more lenient (helps re-link IDs if lighting/pose changes drastically, but increases risk of mistakenly merging two different people who look similar).
  * **Raise it** (e.g., `0.82`) to make ReID more strict (reduces false identity merges, but may result in split IDs and slightly higher footfall counts).

### 4. Tracking and Fragmented IDs
* **`tracker.persist_frames`** (default: `3`): The number of frames a track is kept alive without any detections. If tracks are frequently dropping and re-assigning new IDs to the same person, increase this to `5` or `8`.
* **`tracker.centroid_max_dist`** (default: `80.0`): The max pixel distance allowed for the centroid tracking fallback. 
  * If you increase `pipeline.process_every_n_frames` or have fast-moving people, increase this to `120.0` or `150.0` to bridge the larger distance gaps between processed frames.
  * If you have a very dense crowd and tracks are swapping IDs between adjacent people, lower this to `50.0`.
* **`tracker.size_scale`** (default: `0.30`): Restricts centroid matching to boxes of similar sizes (within 30%). Lower this value to prevent mismatching background objects or different-sized people.

### 5. Filtering Out False Positives (Mannequins, Posters, Clothing)
* **`detector.aspect_min`** & **`detector.aspect_max`** (default: `0.15` / `6.0`): Humans are typically vertical rectangles. If horizontal structures (shelving, cash registers) are detected as people, tighten these limits (e.g., `0.2` to `1.5`).
* **`face_detector.conf_threshold`** (default: `0.45`): Face detection on clothing patterns or logos is automatically rejected by the landmark check, but raising this threshold (e.g., `0.55`) will prevent processing poor-quality faces altogether.

### 6. Fine-Tuning Gender Classification
* **`clip.uncertain_ratio`** (default: `1.8`): Controls how confident CLIP must be to assign male/female vs. `"uncertain"`.
  * **Raise it** (e.g., `2.2` or `2.5`) to increase strictness. More ambiguous detections will be labeled `"uncertain"`, leading to cleaner, highly confident male/female metrics.
  * **Lower it** (e.g., `1.3` or `1.5`) to force the model to make a decision on more people, reducing the number of `"uncertain"` outputs.
* **`face_gender.conf_threshold`** (default: `0.70`): The confidence threshold for the FaceViT classifier. If you notice incorrect face gender classification under bad lighting, raise this to `0.85`.

---

## How Gender Is Decided (Ensemble Logic)

`ensemble_gender()` in `src/ensemble.py` combines the face-based and body-based signals with an explicit decision tree — CLIP-primary, with the face model available as an override or a fallback:

1. **CLIP is confident** (`clip_ratio >= uncertain_ratio`):
   - Face agrees with CLIP → use CLIP's label.
   - Face **disagrees** with CLIP **and** the face model is itself confident → use the face label (face gets the tie-break when both models are confident but conflict).
   - Otherwise → use CLIP's label.
2. **CLIP is wishy-washy** (ratio below threshold):
   - Face model is confident → use the face label.
   - Face model is *also* not confident → return `"uncertain"` (this is the deliberate v9.2 fix — earlier versions silently defaulted to `"male"` here).

Once a track has produced a confident (`male`/`female`) label, `StoreAnalytics` **locks** that track's gender for the rest of the video — later frames don't re-run the decision for that track, so it can't flicker as viewing angle or lighting changes.

---

## Troubleshooting

- **First run is slow / hangs on startup** — it's downloading YOLOv8n weights, the YuNet ONNX model, and two HuggingFace models. Subsequent runs use the local cache.
- **`cannot open <video>` error** — check the path and that OpenCV was built with the right codec support (`opencv-python-headless` from `requirements.txt` covers the common formats: mp4/h264, avi, mov).
- **No people detected** — check `detector.conf` isn't set too high, and that `min_box_w`/`min_box_h` aren't larger than people appear in your footage (common on very wide/high-mounted CCTV shots).
- **Everyone is "uncertain"** — check the footage resolution/lighting is sufficient for CLIP/face classification, or loosen `clip.uncertain_ratio` and `face_gender.conf_threshold` slightly.
- **Footfall counts look wrong** — double check `footfall.line` coordinates (normalised `[x, y]` pairs, not pixels) actually cross the walking path in your footage, and that `footfall.direction` matches which way is "in."
- **GPU not being used** — confirm you installed the CUDA torch wheel (not the CPU one) and set `detector.device: cuda` in your config.

---

## What's Not Included

Deliberately out of scope for this version (see `README.md`'s original notes and `docs/architecture.md`):

- **Age classification** — planned for a later version.
- **Multi-camera fusion** — each video is processed independently; there's no cross-camera identity matching.
- **Persistent ReID across separate video files** — the `ReIDBank` only bridges gaps *within* a single video run; it's cleared between runs.
