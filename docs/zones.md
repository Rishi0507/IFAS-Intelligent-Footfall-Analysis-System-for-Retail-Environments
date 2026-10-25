# Zone occupancy (RQ4)

Which seats, tables or areas are taken, for how long, and which will free up
next, from one fixed camera. This is the restaurant-style view: every table
outlined, green when free and red with a running timer when taken, a status
board, and the next table to free up marked with pulsing rings. It is also the
fourth research question (RQ4 in [research-plan.md](research-plan.md)), because
occupancy is a measurement whose reliability can be checked and whose design
choices can be compared.

Occupancy needs no identity and no attribute model: a place is taken while
someone is in it. That makes it robust where per-person counting is not. On the
classroom recording below, the tracker split about 45 people into 231 tracks,
while seat occupancy matched the people visible.

## Using it

**In the workbench.** Open a run and pick the **Zones** tab. Runs processed
since zones existed get seats proposed automatically; **Edit zones** opens the
editor:

| Tool | What it does |
| :--- | :--- |
| Select | Click a zone to select it; drag it to move it, drag a corner to reshape it |
| Box | Drag a rectangle |
| Polygon | Click the corners, then click the first one, double-click or press Enter to close it; Esc cancels |
| Find seats | Propose seats from where people sat still (replaces the drawn zones) |
| Undo | Ctrl+Z, up to 60 steps |
| Save | Recompute the run. With *Reuse for this camera* on, the layout is also saved under `outputs/zones/<site>__<camera>.json`, and later uploads from that camera use it |

Each zone has a name, a kind (seat, table, queue or area) and an optional
capacity. The **Method** card changes the anchor rule and the smoothing and
shows the effect at once as a preview; nothing is written until you save. The
**Validity** card shows the label-free checks, the next-free evaluation and,
once someone has labelled occupancy, agreement with the labels.

The status board, Gantt chart, utilisation bars, survival curve and alert list
follow the playhead, as does the overlay on the video. A switch on the 3D floor
shows the zones there too.

**From the command line.**

```bash
# Process a recording with tiled detection (needed for dense, seated rooms)
python -m src.inference.pipeline --video clip.avi --out outputs/runs/room1 \
    --gender-repo abhshkp/footfall-analysis-vit-stage2 \
    --age-bundle abhshkp/footfall-analysis-age-svm \
    --stride 5 --tiles 2x2 --site DJSCE --camera 2F-B-CR22

# Recompute zones without running any model: propose seats again, or use a file
python -m src.inference.zones outputs/runs/room1 --propose
python -m src.inference.zones outputs/runs/room1 --zones my_zones.json --off-sec 15

# Burn the overlay into a video, with the status panel beside the frame
python scripts/render_zones.py --run outputs/runs/room1 --video clip.avi \
    --out outputs/renders/room1-zones.mp4 [--start 60 --end 120] [--panel overlay] [--boxes]
```

A zones file is `{"units": "norm" | "px", "zones": [{"id", "name", "kind",
"polygon": [[x, y], ...], "capacity"}]}`, with normalised (0 to 1) or pixel
coordinates.

**Labelling occupancy.** `zones-label.html?run=<id>` shows the clean frames the
pipeline saves every 30 s with the zones drawn on them (the Validity card
links to it). Click a zone to cycle taken, free and unsure, or hover it and
press T, F or U; R marks every zone not yet labelled on the frame as free, and
the arrow keys move between frames. Labels autosave to
`zone_labels_<annotator>.json` in the run folder, one file per annotator so
agreement can be measured.

## Method

**Where a person is.** Each detection becomes one point, the anchor:

| Anchor | Point | For |
| :--- | :--- | :--- |
| `lower` (default) | 85% of the way down the box | People seated behind desks or tables, whose feet are hidden |
| `foot` | Bottom centre of the box | People standing or walking |
| `center` | Centre of the box | Overhead cameras |
| `overlap` | At least 25% of the lower half of the box inside the zone | Large zones such as tables |

An anchor inside several zones goes to the smallest, so a seat drawn inside a
table wins.

**Spatial hysteresis (`stick`, default 0.5).** A seated person leans, and their
anchor crosses into the neighbour's seat for a moment. Once placed in a zone, a
person stays there until their anchor leaves the zone scaled by 1 + `stick`
about its centre, or enters another zone's core (scaled by 1 / (1 + `stick`)).

**Smoothing.** Per zone, absences shorter than `off_sec` (8 s) are bridged as
missed detections, then stays shorter than `on_sec` (3 s) are dropped as
passers-by. A track unseen for less than `gap_fill_sec` (2 s) and seen again in
the same zone counts as present in between. Times stay exact; in a live setting
the reported state would lag by `on_sec` and `off_sec`.

**Censoring.** A stay running when the recording starts or ends is censored: its
length is a lower bound, shown with ≥. An absence cut off by the end of the
recording that is shorter than `off_sec` is not counted as leaving, since
nothing can tell it apart from a missed detection. Without this, every seat
whose occupant was missed in the final seconds read as free: on the classroom
run, 21 of 60 seats taken at the end before the fix, 39 after, with about 38
people visible.

**Which place frees up next.** Stays from every seat and table feed one
Kaplan–Meier estimate, with censored stays counted as lasting at least as long
as observed. For a place taken for *e* seconds, the expected remaining time is
the mean residual life, the area under the curve beyond *e* divided by S(*e*).
The board names the place taken longest, as the restaurant demos do, shows its
expected time left, and lists the runners-up. `zone_study.py` checks that rule
and the alternative (least expected time left) causally, using at each moment
only the stays that had ended by then, against chance.

**Automatic seats.** A stay is at least 20 s in which a track's anchor moves
less than a third of a body height. Stays within half a body width merge into
one place, and a place with at least 40 s of stays in total becomes a seat, in
reading order (S01, S02, …). On the classroom run, the 10 places the 40 s
threshold removed were people standing at the front for a while, a doorway and
second boxes on one person. A backpack the detector took for a person for the
whole recording survived; delete such seats in the editor.

**Tiled detection (`--tiles 2x2`).** Besides the full frame, the detector looks
at overlapping tiles (20% overlap), and boxes are merged by NMS that also drops
a box lying mostly inside a stronger one, which is a person cut in half by a
tile edge.

## Results on the classroom recording

One 9.7-minute recording from a ceiling camera in a lecture room
(1920×1080, 25 frames/s), with about forty students seated throughout,
processed with YOLOv8n and 2×2 tiles every 0.2 s, and 50 seats proposed
automatically. Results: `paper/results/rq4_zones.json` and
`rq4_detection.json`; figure: `paper/figures/rq4_zones.pdf`. Seat occupancy
was labelled on the saved frames by an AI annotator
(`zone_labels_claude-ai.json` in the run): good for comparing settings, to be
confirmed by two people before any number is reported.

| Check | Result |
| :--- | :--- |
| Detection | People found per frame at confidence 0.4, on 20 frames: YOLOv8n 5.2, with 2×2 tiles 14.3; YOLOv8m 14.5, with tiles 27.1. Counts, not recall: no box labels yet |
| Flicker | 7.10 raw state changes per zone-minute, 0.71 after smoothing (10×) |
| Spatial hysteresis | Raw flicker 11.26 to 7.10 (37% fewer), mean seats taken 34.8 to 34.4 |
| Anchor | Sightings in no zone: centre 29.4%, foot 9.8%, lower body 6.4%, overlap 4.5% |
| Sampling | Same state as full rate: 98.1% at 0.4 s, 95.9% at 0.8 s, 92.0% at 1.6 s, 88.4% at 3.2 s |
| Identity | 231 tracks (161 re-joins) for at most 45 people in view: 5.1 tracks per person. Seats taken: 41 at the peak, 37 at the end |
| Next to free up | 0.036 (taken longest), 0.036 (Kaplan–Meier), 0.029 (chance) over 56 moments. Nobody left, so every free-up is a detection artefact: this checks the plumbing, not the rule |
| Against seat labels | 860 seat labels on the 20 saved frames (AI annotator, not human ground truth): precision 0.96 [0.88, 1.00], recall 0.71 [0.63, 0.79], kappa 0.10 at a 94% base rate. Bridging gaps up to 30 s lifts recall to 0.78 at the same precision |

Detection is the binding constraint in a room like this. The pipeline does
better than any single frame (peak 45 people in view, 37 seats taken at the
end) because the tracker carries people through missed frames and re-checks
weak detections where someone was just lost.

Reproduce, without running any model:

```bash
python scripts/research/zone_study.py --run outputs/runs/djsce-cr22-2026-09-14 --source-label "<site, channel, time>"
python scripts/research/detector_study.py --video "<the recording>" --source-label "<site, channel, time>"
python paper/figures/make_figures.py
```

Recorder exports put the recorder's network address in the file name, and the
results files record their source, so pass `--source-label` for anything that
goes into git.

## Limitations

* One room, one camera, one recording, labelled by one AI annotator so far.
* Occupancy is only as good as detection. Far rows are small and half hidden;
  tiling helps, and a person missed for longer than `off_sec` makes their seat
  flip to free.
* Automatic seats need people to sit still for a while, so a seat nobody used
  is never proposed, and a still object the detector mistakes for a person is.
* The next-free rule needs footage where people come and go.
* The method sees bodies in places. It does not identify anyone, and should not
  be used to monitor individuals.
