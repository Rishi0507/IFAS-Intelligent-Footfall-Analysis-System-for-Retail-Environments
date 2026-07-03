# IPD — Retail CCTV Gender Analytics

Retail-store CCTV pipeline that detects people, classifies gender (male / female / uncertain), tracks them across frames, and counts footfall entries/exits. **Age is intentionally not classified in this version.**

## Quick start (local)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# torch CPU wheel (much smaller than default GPU build):
pip install torch --index-url https://download.pytorch.org/whl/cpu

python cctv_gender.py --video input.mp4 --output outputs/run1/
```

## Quick start (Colab)

Open `gender_pipeline.ipynb` in Colab, upload a video, run all cells.

## Output files (in `outputs/<run>/`)

| File | Description |
|---|---|
| `daily_report.csv` | per-bucket entries/exits + gender counts |
| `per_person.csv` | one row per unique track ID with its gender |
| `demographics_pie.png` | overall gender mix (male/female/uncertain) |
| `footfall_by_hour.png` | entries vs exits per hour |
| `summary.json` | full machine-readable summary |
| `annotated_frames/frame_NNNNNN.jpg` | sampled annotated frames for VLM verification |

## Repo layout

See `docs/architecture.md` for the full pipeline diagram.

## Config

Copy `examples/config.example.yaml` and edit. Key knobs:

- `clip.uncertain_ratio: 1.8` — below this ratio (and no confident face) → 'uncertain'
- `reid.cosine_threshold: 0.75` — lower = more willing to re-link IDs
- `footfall.line` — normalised `[x,y],[x,y]` of the crossing line
- `pipeline.process_every_n_frames: 3` — 3x speedup, ~no accuracy loss

## Tests

```bash
pytest tests/ -v
```

## What's deliberately NOT in this version

- Age classification — will be added in a later version.
- Multi-camera fusion.
- Persistent ReID across video files.
