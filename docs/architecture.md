# IPD Architecture (v9.3, no age)

## Pipeline stages

```
video frame
   |
   v
[1] YOLOv8n person detector (conf >= 0.20, native res)
   |     + aspect-ratio filter (reject shelves / posters)
   v
[2] dedup_boxes  (IoU NMS + containment-aware)
   |     merges small-box-inside-big-box that IoU misses
   v
[3] YuNet face detector (row[14] score bug fixed)
   |     + landmark geometry validation (rejects clothing / back-of-head)
   v
[4] FaceViT (rizvandwiki/gender-classification-2)   ->  face gender + conf
[5] CLIP   (openai/clip-vit-base-patch32)           ->  body gender + ratio + embedding
   v
[6] ensemble_gender  (CLIP-primary, 'uncertain' branch v9.2)
   |     CLIP confident -> CLIP wins (face can override if disagrees & confident)
   |     CLIP wishy-washy + face confident -> face wins
   |     CLIP wishy-washy + face uncertain -> 'uncertain'   (was 'male' in v9)
   v
[7] Tracker (IoU + size-scaled centroid, persist=3 frames)
   |     + ReIDBank: when unmatched, query expired tracks by CLIP cosine >= 0.75
   v
[8] LineCounter (entry/exit cross-product sign)
[9] TimeBucketAggregator (15-min buckets)
   v
[10] ReportGenerator -> daily_report.csv, per_person.csv,
                         demographics_pie.png, footfall_by_hour.png,
                         summary.json
```

## Repo layout

```
src/
  config.py        DEFAULT_CONFIG + load_config
  geometry.py      iou, contains, dedup_boxes, validate_face_landmarks
  models.py        YOLOPersonDetector, YuNetFaceDetector, FaceViTGender, CLIPGender
  ensemble.py      ensemble_gender (with 'uncertain')
  tracking.py      Track, Tracker, ReIDBank
  counting.py      LineCounter, TimeBucketAggregator
  reports.py       ReportGenerator
  annotate.py      annotate_frame
  pipeline.py      StoreAnalytics orchestrator
tests/             unit tests for geometry / ensemble / reid / tracker / footfall
cctv_gender.py     CLI entry
gender_pipeline.ipynb   self-contained Colab notebook
```

## Key bug-fixes vs. v8

| Bug | v8 behaviour | v9.3 fix |
|---|---|---|
| YuNet score read | `row[4]` (right_eye_x) | `row[14]` (actual score) |
| False faces | accepted | landmark geometry validation |
| CLIP wishy-washy + no face | returned 'male' | returns 'uncertain' |
| Same person re-detected | new track ID every time | ReIDBank reuses ID (cosine >= 0.75) |
| Small box inside big box | kept both | containment-aware dedup |

## Performance notes

- CPU-only inference ~1-3 fps. Use `pipeline.process_every_n_frames: 3` for 3x speedup.
- For long videos, run in chunks via `--start-frame` / `--end-frame`.
- GPU: change `detector.device` to `cuda` in config.
