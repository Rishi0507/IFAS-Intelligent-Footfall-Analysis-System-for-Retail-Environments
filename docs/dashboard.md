# Dashboard & inference pipeline

The notebooks train and spot-check the models. This part turns them into
something you can look at: a video goes in, an annotated MP4 and an
`analytics.json` come out, and the dashboard reads that JSON.

```mermaid
graph LR
    A[Video] --> B[YOLO person detection]
    B --> C[IoU tracker: boxes -> people]
    C --> D[ViT gender classifier]
    C --> E[SVM age classifier]
    D --> F[Per-track majority vote]
    E --> F
    F --> G[analytics.json]
    C --> H[annotated video.mp4]
    G --> I[Dashboard]
    H --> I
```

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Sample run, so you can see the dashboard without any weights
python scripts/generate_demo_run.py

# 2. Serve it
python -m app.server            # http://127.0.0.1:8000
```

No server? The dashboard also works off plain files:

```bash
python -m http.server 8777      # from the repository root
# then open http://127.0.0.1:8777/app/static/index.html
```

In that mode the run picker only offers `demo` and the upload button is inert -
generate runs from the command line instead.

---

## Processing a real video

```bash
python -m src.inference.pipeline \
    --video store_entrance.mp4 \
    --out outputs/runs/store-2026-09-07 \
    --gender-repo abhshkp/footfall-analysis-vit-stage1 \
    --age-bundle models/age \
    --stride 2
```

| Flag | Default | What it does |
| :--- | :--- | :--- |
| `--gender-repo` / `--gender-ckpt` | – | HuggingFace repo id, or a local checkpoint directory |
| `--age-bundle` | – | Directory holding the pickled age SVMs (see below) |
| `--detector` | `yolov8n.pt` | Any Ultralytics weight; `yolov8s.pt` catches more small people |
| `--det-conf` | `0.4` | Lower it if people are missed, raise it if shelves get boxed |
| `--stride` | `2` | Classify every Nth frame. The tracker fills the gaps, so 2-4 is usually free accuracy-wise and halves the runtime |
| `--min-track-frames` | `3` | Tracks shorter than this are discarded as detector noise |
| `--bin-sec` | `5` | Timeline bin width |
| `--max-frames` | – | Stop early; useful for a smoke test |
| `--no-video` | off | Analytics only, no MP4 re-encode |

Either classifier may be omitted - run gender-only or age-only and the
dashboard simply shows fewer filled fields.

### Running the age model

Unlike the gender checkpoint, there is nothing to download - the age SVMs are
not on the Hub. You need a bundle directory; `models/age/` on this machine is
already set up.

**Whether you need `support.npy` depends on which kernel the grid search
picked.** Check first, it saves a lot of work:

```python
import pickle
print(pickle.load(open("classifiers.pkl", "rb"))["configs"])
```

If every attribute says `rbf` (as the current bundle does), the pickle is
self-contained - copy it into `models/age/` and you are finished. Only an
`intersection` entry needs the support matrix, because a precomputed-kernel SVM
can only score against the exact training matrix it was fitted on.

To produce a bundle from scratch:

**1. Run `age_classifier_v3.ipynb`** (Colab, with PETA) through the training
cells, so `classifiers`, `configs` and `X_train` exist.

**2. Export all three artefacts.** The notebook only pickles the classifiers,
which is not enough: a **precomputed-kernel** SVM can only score against the
exact training matrix it was fitted on, so an intersection-kernel bundle needs
the support matrix as well.

```python
import pickle, numpy as np, os
os.makedirs("age_bundle", exist_ok=True)
with open("age_bundle/classifiers.pkl", "wb") as f:
    pickle.dump({"classifiers": classifiers, "configs": configs}, f)
np.save("age_bundle/support.npy", X_train)     # ~40 MB, required for "intersection" configs
```

If the grid search picked RBF for every attribute, `support.npy` is optional -
the loader tells you which case you are in rather than failing obscurely.

**3. Copy the directory next to the repo** (for example `models/age/`) and run:

```bash
pip install scikit-image        # the LBP texture features need it
python -m src.inference.pipeline --video clip.mp4 --out outputs/runs/clip     --gender-repo abhshkp/footfall-analysis-vit-stage1     --age-bundle models/age
```

In the dashboard's **Process video** dialog, put the same path in the *Age
bundle directory* field. Age is optional: without it the pipeline still runs
and the age panels say so instead of showing empty charts.

The feature vector is 550 numbers per crop (5 body regions x (5 colour
histograms x 16 bins + 3 LBP radii x 10 bins)). If your bundle was trained with
different `N_REGIONS`, `COLOR_BINS`, `LBP_POINTS` or `LBP_RADII`, change the
matching constants at the top of `src/inference/models.py` - the SVMs will
otherwise be fed a feature space they were never fitted on.

---

## What a run directory holds

```
outputs/runs/<run-id>/
├── analytics.json    # the whole record - the dashboard reads only this
├── video.mp4         # annotated output (unless --no-video)
└── poster.jpg        # first frame, used as the video poster
```

Drop any such directory under `outputs/runs/` and it appears in the run picker.

---

## The analytics document

`analytics.json` is the raw per-person record plus pre-computed aggregates. The
dashboard re-aggregates the track list itself so that filters change every chart
consistently; the pre-computed blocks are there for anything else that consumes
the file (a notebook, a BI tool, a report).

| Block | Contents |
| :--- | :--- |
| `meta` | source, resolution, fps, duration, frames, device, throughput |
| `config` | detector, thresholds, stride, model ids - everything needed to reproduce |
| `kpis` | visitors, peak/average occupancy, dwell mean/median/total, footfall rate, gender split and ratio, dominant age bucket, mean confidences, low-confidence rate, busiest window |
| `distributions` | gender, age, age×gender, engagement tiers, path types, entry edges, gender/age confidence histograms, dwell histogram, dwell by age / by gender / by segment, per-zone stats |
| `timeline` | per-bin entries, exits, occupancy (mean and peak), gender and age mix |
| `heatmap` | normalised position density grid over the frame |
| `tracks` | one record per person: id, gender + confidence + vote counts, age + confidence + vote counts, first/last second, dwell, tier, frames, detection confidence, mean box area, entry/exit edge, path type, zone, subsampled trail |
| `model` | model families, checkpoint ids, the age model's reported mA |

Two derived fields worth knowing:

* **Engagement tier** - dwell bucketed as Passing (<3s), Glancing (<10s),
  Browsing (<30s), Engaged (<60s), Dwelling (60s+). Time in frame is the only
  interest proxy this pipeline has; it is not attention.
* **Confidence** - gender confidence is the share of summed softmax probability
  the winning label holds across a person's frames; age confidence is a softmax
  over the four one-vs-rest SVM margins. Both are per-person, after voting.

---

## Store insights and period analysis

The **Store insights** section answers business questions rather than model
questions. It has two halves, and the split matters:

**From this recording** - written into `analytics.json` as `insights`, so any
consumer of the file gets them. Footfall rate, the busiest stretch versus the
quietest, stopping power (who stayed 30s+ versus who walked straight past), who
visits, the busiest area of the frame, and which gender x age segment stays
longest. Each line states what was measured in that clip; clips under two
minutes say so instead of being extrapolated to an hourly rate.

**Across all your recordings** - served by `GET /api/trends`, with a period
switch (**Daily / Weekly / Monthly**) and five metrics: visitors per hour, total
visitors, average visit length, share who stayed 30s+, and share women. Each
period also gets a table row carrying hours watched, visitors, rate, dwell,
engagement, gender split and the biggest age group, plus a per-hour-of-day chart
and a gender mix per period.

### The unit is the observed hour, not the clip

Each run pours its arrivals into `(site, camera, date, hour)` buckets that also
record **how many minutes were actually watched**. Every rate is
`arrivals / minutes observed`, which means:

* a three-hour export outweighs a three-minute phone clip, instead of counting
  the same (this was a real bug in the first version - a short clip could swing
  a weekday average by an order of magnitude);
* two clips from the same afternoon merge into one afternoon rather than voting
  twice;
* a clip spanning 16:50-17:10 contributes ten minutes to each of two hours.

Thin data is stated, not drawn: the daily trend needs **3 days** with at least
five minutes of footage each, and the monthly view needs **two different
months**. A day watched for ten seconds is listed but never counted.

### Sites and cameras

Rates from a doorway camera and an aisle camera are not comparable, so runs
carry `site` and `camera` (`--site` / `--camera`, or the fields in the Process
video dialog). When several cameras are pooled, the coverage line says
*"mixing several cameras"*.

### Real footage versus sample data

Generated sample runs are flagged `demo: true` and **excluded from every
aggregate by default**. The coverage line always states the split - "8 samples
excluded", or a highlighted "includes 8 generated samples" when the *Include
samples* box is ticked - and the recordings table labels every row `Footage` or
`Sample`. Nothing generated ever silently enters a number.

### Capture time

Every period is grouped by when the footage was *recorded*, not processed. The
pipeline stores `meta.captured_at`, taken from:

1. `--captured-at "2026-09-05T14:30"` (or the **Recorded on** dialog field), else
2. the video file's own timestamp, which is what camera exports carry, else
3. the time of processing.

### Building history

Periods only mean something once there are many recordings, so adding a day's
footage is one command:

```bash
python scripts/process_folder.py footage/entrance     --site "MG Road" --camera "Entrance"     --gender-repo abhshkp/footfall-analysis-vit-stage1     --age-bundle models/age --no-video
```

It reads the capture time from each filename (`cam1_2026-09-05_17-30-00.mp4`)
or the file's timestamp, skips videos already processed, and keeps going if one
file fails. `--dry-run` shows what it would do; `--no-video` skips the annotated
MP4, which is much faster for bulk history.

---

## Simple and Detailed views

The header toggle switches the whole page between two audiences, and the choice
is remembered per browser.

**Simple** (the default) is for whoever runs the shop: a plain-language "What
happened" summary, the headline numbers, when people came in, who they were,
and the video. No model vocabulary - no confidence scores, tracks, occupancy or
mA - and the analyst filters are hidden (and cleared, so nothing is silently
filtered out underneath).

**Detailed** adds everything else: the filter row, the segment cross-tab, dwell
and engagement tiers, flow and space, model confidence, and the per-visitor log.

Card titles and captions change with the mode, not just visibility - "Peak
occupancy" becomes "Most at once", "Average dwell" becomes "Typical visit".

---

## Dashboard notes

* **Filters** (gender, age, engagement tier, minimum confidence) sit in one row
  and scope every chart, table and KPI below them, including the video panel.
* **Theme** follows the OS by default; the toggle overrides it and is
  remembered per browser.
* **Charts** are hand-drawn SVG - no chart library, no CDN, works offline. Each
  colour is a CSS variable, so both themes come from one definition. Gender uses
  two categorical hues; age and engagement tiers use ordered single-hue ramps;
  the density map uses a sequential ramp. All were validated for
  colour-vision-deficiency separation and contrast against both surfaces.
* **Table view** - charts with a `Table` button swap to the equivalent numbers,
  and the visitor log carries every raw field (CSV export included).
* The video and the charts share a playhead: scrub the video and the "at the
  playhead" panel follows; click the occupancy chart to seek.

---

## Video codec

Browsers only play H.264. The pipeline asks OpenCV for `avc1` first; if that
build has no H.264 encoder it writes MPEG-4 and then transcodes with `ffmpeg`
when `ffmpeg` is on PATH. With neither available the MP4 is still written (and
plays in VLC), the codec is recorded in `meta.video_codec`, and the dashboard
shows a badge over the player saying so.

---

## Limitations to keep in mind

* One person leaving and re-entering the frame counts twice - there is no
  re-identification across tracks.
* Occupancy is *tracked people in frame*, not a turnstile count.
* Zones are equal thirds of the frame, not real store zones; map them to
  fixtures yourself before quoting them.
* The age model was trained on PETA at PETA's resolution. Small, blurry or
  heavily occluded crops get low-confidence age labels - use the confidence
  filter before drawing conclusions from an age split.
