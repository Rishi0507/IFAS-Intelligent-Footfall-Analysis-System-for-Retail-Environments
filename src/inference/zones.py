"""Zone occupancy: which seats, tables or areas are taken, for how long, and
which is likely to free up next.

A *zone* is a polygon drawn on the camera frame (a table, a seat, a queue, an
aisle). Each sampled frame, every tracked person is placed in at most one zone
by an *anchor* rule; a zone is occupied while somebody is in it. Raw presence
flickers - a seated person behind a desk is detected on some frames and not on
others - so presence is smoothed before anything is reported:

* gaps shorter than `off_sec` are closed (a missed detection does not free a
  table), then
* stays shorter than `on_sec` are dropped (someone walking past does not
  occupy it).

That is a morphological closing followed by an opening on the time axis. The
order matters, and the timestamps stay exact: a session starts at the first
frame somebody was present and ends at the last, not when the smoothing made
up its mind. A live system would announce a change `on_sec` / `off_sec` late;
that latency is reported with the results, not hidden in them.

Sessions that were already running when the recording started, or still
running when it stopped, are *censored*: their true length is only known to be
at least what was seen. The "next to free up" estimate respects that through a
Kaplan-Meier fit of session lengths (`survival`), and is checked against what
actually happened later in the same recording, using only sessions that had
ended before each moment (`evaluate_next_free`), so the check never sees the
future.

Everything here works from `track_boxes.json`, which the pipeline writes next
to analytics.json, so zones can be redrawn and settings changed after a run
without running a model again (`analyse_run`, and `PUT /api/runs/{id}/zones`).

    python -m src.inference.zones outputs/runs/<run> [--zones zones.json] [--anchor lower]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import mean, median

import numpy as np

ZONE_KINDS = ("seat", "table", "queue", "area")
ANCHORS = ("foot", "lower", "center", "overlap")
ZONE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,24}$")


@dataclass
class ZoneParams:
    """How people are placed in zones and how presence is smoothed.

    anchor        where a person "is": "foot" (bottom centre of the box, the
                  usual ground contact point), "lower" (`lower_frac` of the way
                  down the box: for people sitting behind a desk, whose feet
                  are hidden), "center", or "overlap" (the share of the lower
                  half of the box inside the zone is at least `min_overlap`)
    on_sec        presence shorter than this is a passer-by, not a session
    off_sec       absence shorter than this is a missed detection, not a leave
    gap_fill_sec  a track unseen for less than this, and seen again in the same
                  zone, counts as present in between
    stick         spatial hysteresis: a person placed in a zone stays in it until
                  their anchor leaves the zone grown by this share of its size
                  (about its centre) or enters another zone's core (shrunk by
                  the same factor), so leaning towards a neighbour's seat does
                  not move them there. 0 turns it off
    long_stay_sec an occupied zone past this raises a "long stay" event
    queue_wait_sec a person waiting in a queue zone past this raises an event
    """
    anchor: str = "lower"
    lower_frac: float = 0.85
    min_overlap: float = 0.25
    on_sec: float = 3.0
    off_sec: float = 8.0
    gap_fill_sec: float = 2.0
    stick: float = 0.5
    long_stay_sec: float = 300.0
    queue_wait_sec: float = 120.0

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        kinds = {f.name: type(f.default) for f in fields(cls)}
        try:
            p = cls(**{k: kinds[k](d[k]) for k in d if k in kinds})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Bad zone setting: {exc}") from None
        if p.anchor not in ANCHORS:
            raise ValueError(f"anchor must be one of {ANCHORS}, not {p.anchor!r}")
        for name in ("on_sec", "off_sec", "gap_fill_sec", "stick", "long_stay_sec", "queue_wait_sec"):
            if getattr(p, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0.0 <= p.lower_frac <= 1.0 or not 0.0 < p.min_overlap <= 1.0:
            raise ValueError("lower_frac must be in [0, 1] and min_overlap in (0, 1]")
        return p


# ---------------------------------------------------------------- geometry
def polygon_area(poly):
    """Shoelace area of a polygon given as [(x, y), ...]."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def polygon_centroid(poly):
    a = 0.0
    cx = cy = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a) < 1e-12:
        xs, ys = zip(*poly)
        return (sum(xs) / n, sum(ys) / n)
    return (cx / (3.0 * a), cy / (3.0 * a))


def points_in_polygon(points, poly):
    """Vectorised even-odd ray casting. points: (n, 2) array -> (n,) bool."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if y1 == y2:
            continue
        crosses = (y1 > y) != (y2 > y)
        xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < xi)
    return inside


def point_in_polygon(x, y, poly):
    return bool(points_in_polygon([[x, y]], poly)[0])


def _clip(poly, edge_inside, intersect):
    out = []
    n = len(poly)
    for i in range(n):
        cur, prev = poly[i], poly[i - 1]
        cin, pin = edge_inside(cur), edge_inside(prev)
        if cin:
            if not pin:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif pin:
            out.append(intersect(prev, cur))
    return out


def rect_polygon_overlap(rect, poly):
    """Exact area of rect (x1, y1, x2, y2) ∩ polygon, by Sutherland-Hodgman
    clipping of the polygon against the rectangle's four edges."""
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1 or len(poly) < 3:
        return 0.0
    out = [tuple(p) for p in poly]

    def ix(a, b, xv):
        t = (xv - a[0]) / (b[0] - a[0])
        return (xv, a[1] + t * (b[1] - a[1]))

    def iy(a, b, yv):
        t = (yv - a[1]) / (b[1] - a[1])
        return (a[0] + t * (b[0] - a[0]), yv)

    for inside, inter in (
        (lambda p: p[0] >= x1, lambda a, b: ix(a, b, x1)),
        (lambda p: p[0] <= x2, lambda a, b: ix(a, b, x2)),
        (lambda p: p[1] >= y1, lambda a, b: iy(a, b, y1)),
        (lambda p: p[1] <= y2, lambda a, b: iy(a, b, y2)),
    ):
        out = _clip(out, inside, inter)
        if not out:
            return 0.0
    return polygon_area(out)


def anchor_point(box, rule="lower", lower_frac=0.85):
    x1, y1, x2, y2 = box[:4]
    cx = (x1 + x2) / 2.0
    if rule == "foot":
        return cx, float(y2)
    if rule == "center":
        return cx, (y1 + y2) / 2.0
    # "lower", and the fallback point for "overlap"
    return cx, y1 + lower_frac * (y2 - y1)


# ------------------------------------------------------------------- zones
def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s or "")).strip("-").lower() or "default"


def normalise_zones(doc, width=None, height=None):
    """Validate a zones document and return it with normalised (0-1) polygons.

    Accepted: {"zones": [{"id", "name"?, "kind"?, "polygon": [[x, y], ...],
    "capacity"?}], "units": "norm" | "px"}. Pixel polygons need width and height.
    """
    if not isinstance(doc, dict) or not isinstance(doc.get("zones"), list):
        raise ValueError("A zones document is an object with a 'zones' list.")
    units = doc.get("units", "norm")
    if units not in ("norm", "px"):
        raise ValueError("units must be 'norm' or 'px'")
    if units == "px" and not (width and height):
        raise ValueError("Pixel polygons need the frame size.")
    seen, out = set(), []
    for i, z in enumerate(doc["zones"]):
        if not isinstance(z, dict):
            raise ValueError(f"Zone {i} is not an object.")
        zid = str(z.get("id") or f"Z{i + 1:02d}")
        if not ZONE_ID.fullmatch(zid):
            raise ValueError(f"Zone id {zid!r}: use up to 24 letters, digits, '.', '_' or '-'.")
        if zid in seen:
            raise ValueError(f"Zone id {zid!r} is used twice.")
        seen.add(zid)
        poly = z.get("polygon")
        if not isinstance(poly, list) or len(poly) < 3 or len(poly) > 64:
            raise ValueError(f"Zone {zid}: a polygon needs 3 to 64 points.")
        pts = []
        for p in poly:
            if not (isinstance(p, (list, tuple)) and len(p) == 2):
                raise ValueError(f"Zone {zid}: each point is [x, y].")
            x, y = float(p[0]), float(p[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"Zone {zid}: coordinates must be finite numbers.")
            if units == "px":
                x, y = x / width, y / height
            pts.append([round(min(1.0, max(0.0, x)), 5), round(min(1.0, max(0.0, y)), 5)])
        if polygon_area(pts) < 1e-6:
            raise ValueError(f"Zone {zid}: the polygon has no area.")
        kind = z.get("kind", "table")
        if kind not in ZONE_KINDS:
            raise ValueError(f"Zone {zid}: kind must be one of {ZONE_KINDS}.")
        cap = z.get("capacity")
        entry = {
            "id": zid,
            "name": str(z.get("name") or zid)[:48],
            "kind": kind,
            "polygon": pts,
            "capacity": int(cap) if cap not in (None, "") else None,
        }
        ev = z.get("evidence_sec")
        if isinstance(ev, (int, float)) and math.isfinite(ev) and ev >= 0:
            entry["evidence_sec"] = round(float(ev), 1)      # seconds of sitting that found it
        out.append(entry)
    return {"schema": 1, "units": "norm", "source": doc.get("source", "user"), "zones": out}


# ---------------------------------------------------------------- presence
def _sample_grid(tb):
    stride = max(1, int(tb.get("stride") or 1))
    total = int(tb.get("frames_total") or 0)
    if not total:
        last = max((obs[-1][0] for obs in tb["tracks"].values() if obs), default=0)
        total = last + 1
    return stride, total, (total + stride - 1) // stride


def assign_observations(tb, zones, params):
    """Place every observation of every track in at most one zone.

    Returns {track_id: [(sample_index, zone_index), ...]} in time order, with
    zone_index -1 for an observation in no zone. When an anchor falls inside
    several zones the smallest wins: a seat drawn inside a table is the more
    specific answer. With `params.stick`, a track keeps its previous zone while
    its anchor stays inside that zone grown about its centre (see ZoneParams).
    """
    W, H = float(tb["width"]), float(tb["height"])
    stride, _total, _n = _sample_grid(tb)
    polys = [[(x * W, y * H) for x, y in z["polygon"]] for z in zones]
    areas = np.array([polygon_area(p) for p in polys]) if polys else np.zeros(0)
    bboxes = [(min(p[0] for p in q), min(p[1] for p in q), max(p[0] for p in q), max(p[1] for p in q))
              for q in polys]

    rows, owners = [], []
    for tid, obs in tb["tracks"].items():
        for o in obs:
            rows.append(o)
            owners.append(tid)
    out = {tid: [] for tid in tb["tracks"]}
    if not rows or not zones:
        return out
    boxes = np.asarray([r[1:5] for r in rows], dtype=float)
    frames = np.asarray([r[0] for r in rows], dtype=int)

    rule = params.anchor
    ax, ay = zip(*(anchor_point(b, "lower" if rule == "overlap" else rule, params.lower_frac)
                   for b in boxes))
    anchors = np.column_stack([ax, ay])
    best = np.full(len(rows), -1, dtype=int)
    best_score = np.full(len(rows), np.inf)
    if rule != "overlap":
        for zi, poly in enumerate(polys):
            inside = points_in_polygon(anchors, poly)
            better = inside & (areas[zi] < best_score)
            best[better] = zi
            best_score[better] = areas[zi]
    else:
        # Share of the lower half of the box inside the zone. The upper body of
        # a seated person leans over the desk in front; the lower half is where
        # they actually sit.
        lower = boxes.copy()
        lower[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
        lower_area = np.maximum(1e-6, (lower[:, 2] - lower[:, 0]) * (lower[:, 3] - lower[:, 1]))
        best_score[:] = 0.0
        for zi, (poly, bb) in enumerate(zip(polys, bboxes)):
            hit = ~((lower[:, 2] < bb[0]) | (lower[:, 0] > bb[2]) |
                    (lower[:, 3] < bb[1]) | (lower[:, 1] > bb[3]))
            for i in np.nonzero(hit)[0]:
                share = rect_polygon_overlap(lower[i], poly) / lower_area[i]
                if share >= params.min_overlap and share > best_score[i]:
                    best[i], best_score[i] = zi, share
    if params.stick > 0:
        # A band either side of each edge: a person leaves their zone only
        # once outside it grown by `stick`, and enters another only once
        # inside its core, shrunk by the same factor.
        def scaled(poly, f):
            cx, cy = polygon_centroid(poly)
            return [(cx + (x - cx) * f, cy + (y - cy) * f) for x, y in poly]
        grown = [scaled(q, 1.0 + params.stick) for q in polys]
        core = [scaled(q, 1.0 / (1.0 + params.stick)) for q in polys]
        by_track = {}
        for i, tid in enumerate(owners):
            by_track.setdefault(tid, []).append(i)
        for rows_of in by_track.values():
            rows_of.sort(key=lambda i: frames[i])
            prev = -1
            for i in rows_of:
                b = int(best[i])
                if prev >= 0 and b != prev and point_in_polygon(anchors[i, 0], anchors[i, 1], grown[prev]):
                    if b < 0 or not point_in_polygon(anchors[i, 0], anchors[i, 1], core[b]):
                        best[i] = prev
                prev = int(best[i])
    for i, tid in enumerate(owners):
        # Off-grid frames (never written by the pipeline, but harmless) are
        # rounded onto the nearest sample. Zone -1 is "seen, but in no zone":
        # kept, because it tells presence() the person was somewhere else.
        out[tid].append((int(round(frames[i] / stride)), int(best[i])))
    for tid in out:
        out[tid].sort()
    return out


def presence(tb, zones, params, assigned=None):
    """Per-zone presence on the sampling grid.

    Returns (counts, occupants): counts is an (n_zones, n_samples) int array of
    people in each zone at each sample; occupants[z] maps a sample index to the
    set of track ids there.
    """
    stride, _total, n = _sample_grid(tb)
    fps = float(tb.get("fps") or 25.0)
    dt = stride / fps
    fill = int(math.floor(params.gap_fill_sec / dt + 1e-9))
    assigned = assigned if assigned is not None else assign_observations(tb, zones, params)
    counts = np.zeros((len(zones), n), dtype=np.int32)
    occupants = [dict() for _ in zones]
    for tid, seq in assigned.items():
        prev = None
        for s, zi in seq:
            if s < 0 or s >= n:
                continue
            if zi < 0:
                prev = (s, zi)
                continue
            span = [s]
            if prev is not None and prev[1] == zi and 1 < s - prev[0] <= fill + 1:
                span = range(prev[0] + 1, s + 1)
            for k in span:
                ids = occupants[zi].setdefault(k, set())
                if tid not in ids:
                    ids.add(tid)
                    counts[zi, k] += 1
            prev = (s, zi)
    return counts, occupants


def _runs(mask):
    """[(start, end_inclusive)] of the True runs in a boolean array."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def smooth(mask, dt, on_sec, off_sec, edges=True):
    """Close gaps shorter than off_sec, then drop runs shorter than on_sec.

    Returns [(start, end_inclusive)] sample runs. Durations count each sample
    as `dt` seconds long, so a single present sample lasts dt.

    With `edges`, a kept run that stops less than off_sec before the recording
    ends is stretched to the end, and one that starts less than off_sec after
    it begins is stretched to the start. Such a gap is too short to count as
    leaving (the recording cut it off), so the stay is treated as running and
    is censored. Without this, every seat whose occupant was missed for the
    last few seconds reads as free at the end.
    """
    runs = _runs(mask)
    merged = []
    for s, e in runs:
        if merged and (s - merged[-1][1] - 1) * dt < off_sec:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    kept = [(s, e) for s, e in merged if (e - s + 1) * dt >= on_sec]
    if edges and kept:
        n = len(mask)
        s, e = kept[0]
        if s * dt < off_sec:
            kept[0] = (0, e)
        s, e = kept[-1]
        if (n - 1 - e) * dt < off_sec:
            kept[-1] = (s, n - 1)
    return kept


def _transitions(mask):
    m = np.asarray(mask, dtype=np.int8)
    return int(np.abs(np.diff(m)).sum()) if len(m) > 1 else 0


# ---------------------------------------------------------------- survival
def kaplan_meier(durations, observed):
    """Kaplan-Meier survival of session lengths.

    durations: seconds; observed: True where the session's end was seen (False
    = right-censored: it lasted at least that long). Returns {"t": [...], "s":
    [...]}, a right-continuous step function starting at (0, 1).
    """
    pairs = sorted(zip(durations, observed))
    t_out, s_out = [0.0], [1.0]
    at_risk = len(pairs)
    s = 1.0
    i = 0
    while i < len(pairs):
        t = pairs[i][0]
        deaths = cens = 0
        while i < len(pairs) and pairs[i][0] == t:
            deaths += 1 if pairs[i][1] else 0
            cens += 0 if pairs[i][1] else 1
            i += 1
        if deaths and at_risk:
            s *= 1.0 - deaths / at_risk
            t_out.append(round(float(t), 3))
            s_out.append(round(s, 5))
        at_risk -= deaths + cens
    return {"t": t_out, "s": s_out, "n": len(pairs), "events": sum(1 for _d, o in pairs if o)}


def km_at(km, t):
    s = 1.0
    for tt, ss in zip(km["t"], km["s"]):
        if tt <= t:
            s = ss
        else:
            break
    return s


def mean_residual(km, elapsed, horizon=None):
    """E[T - e | T > e], restricted to `horizon` (default: the last step).

    None when nothing is known past `elapsed`.
    """
    t, s = km["t"], km["s"]
    if len(t) < 2:
        return None
    horizon = horizon if horizon is not None else t[-1]
    if elapsed >= horizon:
        return None
    se = km_at(km, elapsed)
    if se <= 0:
        return 0.0
    area = 0.0
    pts = [x for x in t if elapsed < x < horizon] + [horizon]
    prev = elapsed
    for x in pts:
        area += km_at(km, prev) * (x - prev)
        prev = x
    return area / se


# ---------------------------------------------------------------- analysis
def _fmt_sessions(runs, counts_row, occupants, dt, n):
    out = []
    for s, e in runs:
        ids = set()
        for k in range(s, e + 1):
            ids |= occupants.get(k, set())
        start, end = s * dt, (e + 1) * dt
        out.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "people": sorted(int(i) if str(i).isdigit() else i for i in ids),
            "peak": int(counts_row[s:e + 1].max()) if e >= s else 0,
            "censored_start": s == 0,
            "censored_end": e == n - 1,
        })
    return out


def _queue_waits(tb, zi, assigned, params, dt, n):
    """Per-person stays inside one queue zone, smoothed like zone presence."""
    waits = []
    for tid, seq in assigned.items():
        mask = np.zeros(n, dtype=bool)
        for s, z in seq:
            if z == zi and 0 <= s < n:
                mask[s] = True
        if not mask.any():
            continue
        for s, e in smooth(mask, dt, params.on_sec, params.off_sec):
            waits.append({
                "track": int(tid) if str(tid).isdigit() else tid,
                "start": round(s * dt, 2), "end": round((e + 1) * dt, 2),
                "wait": round((e - s + 1) * dt, 2),
                "censored": e == n - 1 or s == 0,
            })
    waits.sort(key=lambda w: w["start"])
    return waits


def evaluate_next_free(zone_sessions, duration, step=10.0, min_events=3):
    """How often each rule names the zone that actually frees up first.

    At every `step` seconds, the candidates are the zones occupied at that
    moment. The answer is whichever of them is next seen to empty; moments
    where none empties before the recording stops are skipped. Rules:

    * longest_seated - the zone occupied longest so far
    * km_residual    - the zone with the shortest expected remaining time,
                       from a Kaplan-Meier fit to sessions that had already
                       ended by that moment (causal: no peeking ahead). With
                       fewer than `min_events` ended sessions it falls back to
                       longest_seated, and the fallbacks are counted.
    * random         - the expected score of a uniform guess, 1 / candidates.
    """
    hits = {"longest_seated": 0, "km_residual": 0}
    rnd = 0.0
    n = fallbacks = 0
    t = step
    while t < duration:
        occ = []
        for zi, sess in enumerate(zone_sessions):
            for s in sess:
                if s["start"] <= t < s["end"]:
                    occ.append((zi, s))
                    break
        ended = [s for s in occ if not s[1]["censored_end"]]
        if len(occ) >= 2 and ended:
            truth = min(ended, key=lambda x: x[1]["end"])[0]
            longest = max(occ, key=lambda x: (t - x[1]["start"], -x[0]))[0]
            # What was known at t: sessions that had ended (an event, unless
            # they began before the recording), and sessions still running,
            # censored at their length so far.
            past = [(s["duration"], not s["censored_start"])
                    for sess in zone_sessions for s in sess if s["end"] <= t]
            running = [(t - s["start"], False)
                       for sess in zone_sessions for s in sess if s["start"] < t < s["end"]]
            km = None
            if sum(1 for _d, o in past if o) >= min_events:
                km = kaplan_meier(*zip(*(past + running)))
            if km is not None:
                scored = []
                for zi, s in occ:
                    r = mean_residual(km, t - s["start"])
                    scored.append((r if r is not None else float("inf"), -(t - s["start"]), zi))
                km_pick = min(scored)[2]
            else:
                km_pick = longest
                fallbacks += 1
            hits["longest_seated"] += longest == truth
            hits["km_residual"] += km_pick == truth
            rnd += 1.0 / len(occ)
            n += 1
        t += step
    if not n:
        return {"moments": 0, "step_sec": step}
    return {
        "moments": n,
        "step_sec": step,
        "longest_seated": round(hits["longest_seated"] / n, 4),
        "km_residual": round(hits["km_residual"] / n, 4),
        "random": round(rnd / n, 4),
        "km_fallbacks": fallbacks,
    }


def analyse(tb, zones_doc, params=None, bin_sec=10.0):
    """Zone occupancy for one run. tb: the track_boxes document."""
    params = params if isinstance(params, ZoneParams) else ZoneParams.from_dict(params)
    zdoc = normalise_zones(zones_doc, tb.get("width"), tb.get("height"))
    zones = zdoc["zones"]
    stride, total, n = _sample_grid(tb)
    fps = float(tb.get("fps") or 25.0)
    dt = stride / fps
    duration = total / fps
    assigned = assign_observations(tb, zones, params)
    counts, occupants = presence(tb, zones, params, assigned)

    out_zones, all_sessions, events = [], [], []
    occupied_mask = np.zeros((len(zones), n), dtype=bool)
    for zi, z in enumerate(zones):
        raw = counts[zi] > 0
        runs = smooth(raw, dt, params.on_sec, params.off_sec)
        for s, e in runs:
            occupied_mask[zi, s:e + 1] = True
        sessions = _fmt_sessions(runs, counts[zi], occupants[zi], dt, n)
        all_sessions.append(sessions)
        occ_sec = sum(s["duration"] for s in sessions)
        durs = [s["duration"] for s in sessions]
        gaps = [b["start"] - a["end"] for a, b in zip(sessions, sessions[1:])]
        people = sorted({p for s in sessions for p in s["people"]}, key=str)
        minutes = duration / 60.0 if duration else 0.0
        metrics = {
            "occupied_sec": round(occ_sec, 2),
            "utilization": round(occ_sec / duration, 4) if duration else 0.0,
            "sessions": len(sessions),
            "turnover_per_hour": round(len(sessions) / (duration / 3600.0), 2) if duration else 0.0,
            "mean_session_sec": round(mean(durs), 2) if durs else 0.0,
            "median_session_sec": round(median(durs), 2) if durs else 0.0,
            "max_session_sec": round(max(durs), 2) if durs else 0.0,
            "mean_free_gap_sec": round(mean(gaps), 2) if gaps else None,
            "people": len(people),
            "mean_people_when_occupied": round(float(counts[zi][occupied_mask[zi]].mean()), 2)
            if occupied_mask[zi].any() else 0.0,
            "raw_presence_sec": round(float(raw.sum()) * dt, 2),
            "raw_flips_per_min": round(_transitions(raw) / minutes, 3) if minutes else 0.0,
            "smoothed_flips_per_min": round(_transitions(occupied_mask[zi]) / minutes, 3) if minutes else 0.0,
        }
        entry = dict(z, metrics=metrics, sessions=sessions)
        if z["kind"] == "queue":
            waits = _queue_waits(tb, zi, assigned, params, dt, n)
            w = [x["wait"] for x in waits]
            entry["queue"] = {
                "waits": waits,
                "mean_wait_sec": round(mean(w), 2) if w else 0.0,
                "median_wait_sec": round(median(w), 2) if w else 0.0,
                "max_wait_sec": round(max(w), 2) if w else 0.0,
                "max_length": int(counts[zi].max()) if n else 0,
            }
            for x in waits:
                if x["wait"] >= params.queue_wait_sec:
                    events.append({"t": round(x["start"] + params.queue_wait_sec, 2), "type": "queue_wait",
                                   "zone": z["id"], "track": x["track"], "wait": x["wait"]})
        else:
            for s in sessions:
                if not s["censored_start"]:
                    events.append({"t": s["start"], "type": "occupied", "zone": z["id"],
                                   "people": len(s["people"])})
                if not s["censored_end"]:
                    events.append({"t": s["end"], "type": "freed", "zone": z["id"],
                                   "after": s["duration"]})
                if s["duration"] >= params.long_stay_sec and z["kind"] in ("seat", "table"):
                    events.append({"t": round(s["start"] + params.long_stay_sec, 2), "type": "long_stay",
                                   "zone": z["id"], "lasted": s["duration"],
                                   "censored": s["censored_start"]})
        out_zones.append(entry)
    events.sort(key=lambda e: (e["t"], e["zone"], e["type"]))

    # Seats and tables share one survival curve; queues and areas are not "freed".
    seated = [all_sessions[i] for i, z in enumerate(zones) if z["kind"] in ("seat", "table")]
    flat = [s for sess in seated for s in sess]
    km = kaplan_meier([s["duration"] for s in flat],
                      [not (s["censored_start"] or s["censored_end"]) for s in flat]) if flat else None

    at_end = []
    for zi, z in enumerate(zones):
        if z["kind"] not in ("seat", "table"):
            continue
        last = out_zones[zi]["sessions"][-1] if out_zones[zi]["sessions"] else None
        if last and last["censored_end"]:
            elapsed = duration - last["start"]
            at_end.append({"zone": z["id"], "elapsed": round(elapsed, 2),
                           "lower_bound": last["censored_start"],
                           "expected_left": (round(mean_residual(km, elapsed), 1)
                                             if km and mean_residual(km, elapsed) is not None else None)})
    at_end.sort(key=lambda r: -r["elapsed"])

    n_bins = max(1, int(math.ceil(duration / bin_sec))) if duration else 1
    per_bin = max(1, int(round(bin_sec / dt)))
    occ_count = occupied_mask[[i for i, z in enumerate(zones) if z["kind"] in ("seat", "table")]].sum(axis=0) \
        if seated else np.zeros(n, dtype=int)
    timeline = []
    for b in range(n_bins):
        seg = occ_count[b * per_bin:(b + 1) * per_bin]
        timeline.append(round(float(seg.mean()), 2) if len(seg) else 0.0)

    util = [z["metrics"]["utilization"] for z in out_zones if z["kind"] in ("seat", "table")]
    raw_f = [z["metrics"]["raw_flips_per_min"] for z in out_zones]
    sm_f = [z["metrics"]["smoothed_flips_per_min"] for z in out_zones]
    summary = {
        "zones": len(zones),
        "seats_tables": len(seated),
        "queues": sum(1 for z in zones if z["kind"] == "queue"),
        "mean_utilization": round(mean(util), 4) if util else 0.0,
        "sessions": len(flat),
        "median_session_sec": round(median([s["duration"] for s in flat]), 2) if flat else 0.0,
        "peak_occupied": int(occ_count.max()) if len(occ_count) else 0,
        "mean_occupied": round(float(occ_count.mean()), 2) if len(occ_count) else 0.0,
        "occupied_at_end": int(occ_count[-1]) if len(occ_count) else 0,
        "free_at_end": (len(seated) - int(occ_count[-1])) if len(occ_count) else len(seated),
        "raw_flips_per_min": round(mean(raw_f), 3) if raw_f else 0.0,
        "smoothed_flips_per_min": round(mean(sm_f), 3) if sm_f else 0.0,
        "unassigned_share": _unassigned_share(tb, assigned),
    }
    return {
        "schema": 1,
        "source": zdoc.get("source", "user"),
        "params": asdict(params),
        "sample_sec": round(dt, 4),
        "latency_sec": {"occupied": params.on_sec, "freed": params.off_sec},
        "duration_sec": round(duration, 2),
        "zones": out_zones,
        "summary": summary,
        "timeline": {"bin_sec": bin_sec, "occupied": timeline},
        "events": events[:2000],
        "survival": km,
        "next_free": {"at_end": at_end,
                      "evaluation": evaluate_next_free(seated, duration) if seated else {"moments": 0}},
    }


def _unassigned_share(tb, assigned):
    """Share of observations that fell in no zone: high means the zones miss
    where people actually are (or the anchor rule puts them elsewhere)."""
    seen = [z for seq in assigned.values() for _s, z in seq]
    return round(sum(1 for z in seen if z < 0) / len(seen), 4) if seen else 0.0


# ------------------------------------------------------- automatic proposal
def _stays(obs, params, fps, min_stay_sec):
    """Where one track sat still: [(cx, cy, w, h, seconds)]."""
    out = []
    cur = []
    sx = sy = 0.0                  # running sums of the current stay's anchors
    for o in obs:
        box = o[1:5]
        ax, ay = anchor_point(box, "lower" if params.anchor == "overlap" else params.anchor,
                              params.lower_frac)
        h = box[3] - box[1]
        if cur and math.hypot(ax - sx / len(cur), ay - sy / len(cur)) > 0.35 * max(h, 1):
            if (cur[-1][4] - cur[0][4]) / fps >= min_stay_sec:
                out.append(cur)
            cur = []
            sx = sy = 0.0
        cur.append((ax, ay, box[2] - box[0], h, o[0]))
        sx += ax
        sy += ay
    if cur and (cur[-1][4] - cur[0][4]) / fps >= min_stay_sec:
        out.append(cur)
    return [(mean(c[0] for c in s), mean(c[1] for c in s), median(c[2] for c in s),
             median(c[3] for c in s), (s[-1][4] - s[0][4]) / fps) for s in out]


def propose_zones(tb, params=None, min_stay_sec=20.0, min_evidence_sec=None, kind="seat"):
    """Find seats from where people sat still, without anyone drawing a zone.

    Every stretch of at least `min_stay_sec` in which a track's anchor stayed
    within a third of a body height is a *stay*. Stays from any track whose
    centres lie within half a body width merge into one place, and each place
    becomes a box around its centre, sized from the people who sat there.
    Places with less than `min_evidence_sec` of stays in total (default twice
    `min_stay_sec`) are dropped: on the classroom clip those were people
    standing at the front for a while, a doorway, and second boxes on one
    person. The result is in reading order (top row first) and named S01, ...
    """
    params = params if isinstance(params, ZoneParams) else ZoneParams.from_dict(params)
    W, H = float(tb["width"]), float(tb["height"])
    fps = float(tb.get("fps") or 25.0)
    stays = []
    for obs in tb["tracks"].values():
        stays.extend(_stays(obs, params, fps, min_stay_sec))
    stays.sort(key=lambda s: -s[4])
    places = []                    # [sum_w * x, sum_w * y, w, h, total_sec, sum_w]
    for x, y, w, h, sec in stays:
        best, best_d = None, None
        for p in places:
            px, py = p[0] / p[5], p[1] / p[5]
            d = math.hypot(x - px, (y - py) * 1.6)        # rows are tighter vertically
            if d < 0.5 * max(p[2], w) and (best is None or d < best_d):
                best, best_d = p, d
        if best is None:
            places.append([x * sec, y * sec, w, h, sec, sec])
        else:
            best[0] += x * sec
            best[1] += y * sec
            best[2] = (best[2] * best[5] + w * sec) / (best[5] + sec)
            best[3] = (best[3] * best[5] + h * sec) / (best[5] + sec)
            best[4] += sec
            best[5] += sec
    if min_evidence_sec is None:
        min_evidence_sec = 2.0 * min_stay_sec
    rects = []
    for p in places:
        if p[4] < min_evidence_sec:
            continue
        cx, cy, w, h = p[0] / p[5], p[1] / p[5], p[2], p[3]
        x1, x2 = max(0.0, cx - 0.42 * w), min(W, cx + 0.42 * w)
        y1, y2 = max(0.0, cy - 0.30 * h), min(H, cy + 0.18 * h)
        rects.append((cy, cx, x1, y1, x2, y2, h, p[4]))
    # Reading order: group into rows by centre height, then left to right.
    rects.sort()
    rows, row = [], []
    for r in rects:
        if row and r[0] - mean(q[0] for q in row) > 0.5 * median(q[6] for q in row):
            rows.append(row)
            row = []
        row.append(r)
    if row:
        rows.append(row)
    zones = []
    for row in rows:
        for r in sorted(row, key=lambda q: q[1]):
            _cy, _cx, x1, y1, x2, y2, _h, sec = r
            zones.append({
                "id": f"S{len(zones) + 1:02d}", "name": f"Seat {len(zones) + 1}", "kind": kind,
                "polygon": [[round(x1 / W, 5), round(y1 / H, 5)], [round(x2 / W, 5), round(y1 / H, 5)],
                            [round(x2 / W, 5), round(y2 / H, 5)], [round(x1 / W, 5), round(y2 / H, 5)]],
                "capacity": 1, "evidence_sec": round(sec, 1),
            })
    return {"schema": 1, "units": "norm", "source": "auto",
            "method": {"min_stay_sec": min_stay_sec, "min_evidence_sec": min_evidence_sec,
                       "anchor": params.anchor}, "zones": zones}


# ---------------------------------------------------------------- run files
def write_json_atomic(path, doc, indent=1):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=indent)
    os.replace(tmp, path)


def camera_zone_file(root, site, camera):
    """Where a camera's zones live, so every recording from it reuses them."""
    return Path(root) / f"{_slug(site)}__{_slug(camera)}.json"


def load_track_boxes(run_dir):
    p = Path(run_dir) / "track_boxes.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{run_dir} has no track_boxes.json. Runs processed before zones existed "
            "need processing again.")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def analyse_run(run_dir, zones_doc=None, params=None, save=True, propose_kw=None):
    """Recompute a run's zones from its track_boxes.json.

    zones_doc None: the run's zones.json if there is one, else an automatic
    proposal. With save, writes zones.json and the `zones` block of
    analytics.json. Returns the zones block.
    """
    run_dir = Path(run_dir)
    tb = load_track_boxes(run_dir)
    zfile = run_dir / "zones.json"
    if zones_doc is None and zfile.exists():
        with open(zfile, encoding="utf-8") as f:
            zones_doc = json.load(f)
    if zones_doc is None:
        zones_doc = propose_zones(tb, params, **(propose_kw or {}))
    zones_doc = normalise_zones(zones_doc, tb.get("width"), tb.get("height"))
    block = analyse(tb, zones_doc, params)
    if save:
        write_json_atomic(zfile, zones_doc)
        a = run_dir / "analytics.json"
        if a.exists():
            with open(a, encoding="utf-8") as f:
                doc = json.load(f)
            doc["zones"] = block
            if "insights" in doc:
                from .trends import insights
                doc["insights"] = insights(doc)
            if "guide" in doc or "tracks" in doc:
                from .guide import shop_guide
                doc["guide"] = shop_guide(doc)
            write_json_atomic(a, doc)
    return block


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="a run directory holding track_boxes.json")
    p.add_argument("--zones", default=None, help="zones JSON; default: the run's zones.json, else proposed")
    p.add_argument("--propose", action="store_true", help="ignore saved zones and propose seats again")
    p.add_argument("--min-stay", type=float, default=20.0, help="seconds of sitting still that make a seat")
    p.add_argument("--min-evidence", type=float, default=None,
                   help="seconds of sitting a seat needs in total (default: twice --min-stay)")
    for f in fields(ZoneParams):
        p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    p.add_argument("--dry-run", action="store_true", help="print, do not write")
    a = p.parse_args()
    params = ZoneParams.from_dict({f.name: getattr(a, f.name) for f in fields(ZoneParams)})
    zones_doc = None
    if a.zones:
        with open(a.zones, encoding="utf-8") as f:
            zones_doc = json.load(f)
    elif a.propose:
        zones_doc = propose_zones(load_track_boxes(a.run), params, min_stay_sec=a.min_stay,
                                  min_evidence_sec=a.min_evidence)
    block = analyse_run(a.run, zones_doc, params, save=not a.dry_run,
                        propose_kw={"min_stay_sec": a.min_stay, "min_evidence_sec": a.min_evidence})
    s = block["summary"]
    print(f"{s['zones']} zones ({block['source']}) · mean utilisation {100 * s['mean_utilization']:.1f}% · "
          f"{s['sessions']} sessions, median {s['median_session_sec']:.0f}s · peak {s['peak_occupied']} occupied · "
          f"flicker {s['raw_flips_per_min']:.2f} -> {s['smoothed_flips_per_min']:.2f} changes/min")
    ev = block["next_free"]["evaluation"]
    if ev.get("moments"):
        print(f"next-free top-1 over {ev['moments']} moments: longest-seated {ev['longest_seated']:.2f}, "
              f"KM {ev['km_residual']:.2f}, random {ev['random']:.2f}")


if __name__ == "__main__":
    main()
