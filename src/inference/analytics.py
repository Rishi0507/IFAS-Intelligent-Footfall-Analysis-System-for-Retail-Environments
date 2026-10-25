"""Turn raw tracks into the analytics document the dashboard reads.

Everything here is derived from what the two models actually emit - person
boxes, a gender distribution, an age-bucket distribution - plus time and
position. Nothing is invented: if a value cannot be derived from a track, it is
not in the document.
"""
from __future__ import annotations
import datetime as _dt
from statistics import mean, median

GENDER_LABELS = ["female", "male"]
AGE_LABELS = ["Age16-30", "Age31-45", "Age46-60", "AgeAbove61"]

# Dwell tiers: how retail reads time-in-frame.
DWELL_TIERS = [
    ("Passing", 0.0, 3.0),
    ("Glancing", 3.0, 10.0),
    ("Browsing", 10.0, 30.0),
    ("Engaged", 30.0, 60.0),
    ("Dwelling", 60.0, float("inf")),
]
ZONES = ["Left", "Center", "Right"]
LOW_CONF = 0.60


def _hist(values, n_bins=10, lo=0.0, hi=1.0):
    if hi <= lo:
        hi = lo + 1.0
    counts = [0] * n_bins
    width = (hi - lo) / n_bins
    for v in values:
        i = min(n_bins - 1, max(0, int((v - lo) / width)))
        counts[i] += 1
    edges = [round(lo + i * width, 3) for i in range(n_bins + 1)]
    return {"edges": edges, "counts": counts}


def _pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else 0.0


def _side(cx, cy, w, h, margin=0.12):
    if cx <= w * margin:
        return "left"
    if cx >= w * (1 - margin):
        return "right"
    if cy <= h * margin:
        return "top"
    if cy >= h * (1 - margin):
        return "bottom"
    return "interior"


def _thin(path, n=60):
    """At most n points, evenly spread, always keeping the first and the last.

    (Slicing with a step and then cutting to n drops the end of the trail:
    a path of 119 points lost its whole second half that way.)
    """
    if len(path) <= n:
        return path
    step = (len(path) - 1) / float(n - 1)
    return [path[int(round(i * step))] for i in range(n)]


def _tier(dwell):
    for name, lo, hi in DWELL_TIERS:
        if lo <= dwell < hi:
            return name
    return DWELL_TIERS[-1][0]


def _label(t, which, min_conf):
    """A person's label, or None when it should not be trusted: the votes
    changed their mind halfway (the box most likely moved between two people),
    or the pooled confidence is under `min_conf`. Returns (label, conf, why)."""
    label, conf = t.gender() if which == 0 else t.age()
    if label is None:
        return None, 0.0, None
    if hasattr(t, "consistent") and not t.consistent(which):
        return None, conf, "votes disagree"
    if conf < min_conf:
        return None, conf, "low confidence"
    return label, conf, None


def build_analytics(tracks, meta, config, occupancy_by_frame, model_card=None,
                    bin_sec=5.0, heat_cols=16, heat_rows=9, min_frames=3,
                    min_gender_conf=0.0, min_age_conf=0.0):
    """tracks: list[Track] - occupancy_by_frame: {frame_idx: n_people}.

    A label below `min_gender_conf` / `min_age_conf`, or one whose votes
    disagree between the first and second half of the stay, is left out
    (None) and the reason kept in `gender_unsure` / `age_unsure`: a share of
    "don't know" is better than a confident wrong answer."""
    fps = meta.get("fps") or 25.0
    w, h = meta["width"], meta["height"]
    duration = meta.get("duration_sec") or (meta.get("frames_total", 0) / fps)

    kept = [t for t in tracks if t.frames >= min_frames]
    rejected = len(tracks) - len(kept)

    records = []
    heat = [[0.0] * heat_cols for _ in range(heat_rows)]
    for t in kept:
        g, g_conf, g_why = _label(t, 0, min_gender_conf)
        a, a_conf, a_why = _label(t, 1, min_age_conf)
        first_sec = t.first_frame / fps
        last_sec = t.last_frame / fps
        dwell = max(last_sec - first_sec, 1.0 / fps)
        fx, fy = t.path[0][1], t.path[0][2]
        lx, ly = t.path[-1][1], t.path[-1][2]

        for _f, px, py, _area in t.path:
            c = min(heat_cols - 1, max(0, int(px / w * heat_cols)))
            r = min(heat_rows - 1, max(0, int(py / h * heat_rows)))
            heat[r][c] += 1.0

        entry = _side(fx, fy, w, h)
        exit_ = _side(lx, ly, w, h)
        if entry != "interior" and exit_ == "interior":
            flow = "entered"
        elif entry == "interior" and exit_ != "interior":
            flow = "exited"
        elif entry != "interior" and exit_ != "interior" and entry != exit_:
            flow = "passed through"
        else:
            flow = "in view"

        zone_i = min(len(ZONES) - 1, int(mean(p[1] for p in t.path) / w * len(ZONES)))
        records.append({
            "id": t.id,
            "gender": g,
            "gender_conf": round(g_conf, 3),
            "gender_votes": dict(t.gender_votes),
            "age": a,
            "age_conf": round(a_conf, 3),
            "age_votes": dict(t.age_votes),
            "gender_unsure": g_why,
            "age_unsure": a_why,
            "first_sec": round(first_sec, 2),
            "last_sec": round(last_sec, 2),
            "dwell_sec": round(dwell, 2),
            "tier": _tier(dwell),
            "frames": t.frames,
            "classified_frames": t.n_classified,
            "rejoins": getattr(t, "rejoins", 0),
            "det_conf": round(t.det_conf_sum / max(t.frames, 1), 3),
            "avg_size_pct": round(100 * mean(p[3] for p in t.path), 2),
            "entry": entry,
            "exit": exit_,
            "flow": flow,
            "zone": ZONES[zone_i],
            # Subsampled: the dashboard draws trails, it does not re-track.
            "path": [[round(p[0] / fps, 2), p[1], p[2]] for p in _thin(t.path)],
        })

    peak_cell = max((heat[r][c] for r in range(heat_rows) for c in range(heat_cols)), default=0) or 1.0
    heat = [[round(v / peak_cell, 4) for v in row] for row in heat]

    # ---- distributions -------------------------------------------------
    total = len(records)
    gender_counts = {g: sum(1 for r in records if r["gender"] == g) for g in GENDER_LABELS}
    age_counts = {a: sum(1 for r in records if r["age"] == a) for a in AGE_LABELS}
    cross = {g: {a: sum(1 for r in records if r["gender"] == g and r["age"] == a)
                 for a in AGE_LABELS} for g in GENDER_LABELS}
    tier_counts = {name: sum(1 for r in records if r["tier"] == name) for name, _, _ in DWELL_TIERS}
    flow_counts = {}
    for r in records:
        flow_counts[r["flow"]] = flow_counts.get(r["flow"], 0) + 1

    def seg_stats(pred):
        d = [r["dwell_sec"] for r in records if pred(r)]
        return {"n": len(d),
                "avg": round(mean(d), 2) if d else 0.0,
                "median": round(median(d), 2) if d else 0.0,
                "max": round(max(d), 2) if d else 0.0}

    dwell_by_age = {a: seg_stats(lambda r, a=a: r["age"] == a) for a in AGE_LABELS}
    dwell_by_gender = {g: seg_stats(lambda r, g=g: r["gender"] == g) for g in GENDER_LABELS}
    dwell_by_segment = [
        dict(gender=g, age=a,
             **seg_stats(lambda r, g=g, a=a: r["gender"] == g and r["age"] == a))
        for g in GENDER_LABELS for a in AGE_LABELS
    ]
    zone_stats = {}
    for z in ZONES:
        s = seg_stats(lambda r, z=z: r["zone"] == z)
        zone_stats[z] = {
            "visitors": s["n"],
            "avg_dwell": s["avg"],
            "median_dwell": s["median"],
            "gender": {g: sum(1 for r in records if r["zone"] == z and r["gender"] == g)
                       for g in GENDER_LABELS},
            "age": {a: sum(1 for r in records if r["zone"] == z and r["age"] == a)
                    for a in AGE_LABELS},
        }

    # ---- timeline ------------------------------------------------------
    n_bins = max(1, int(duration / bin_sec) + 1)
    bins = []
    for i in range(n_bins):
        t0, t1 = i * bin_sec, (i + 1) * bin_sec
        entered = [r for r in records if t0 <= r["first_sec"] < t1]
        left = [r for r in records if t0 <= r["last_sec"] < t1]
        occ = [n for f, n in occupancy_by_frame.items() if t0 <= f / fps < t1]
        present = [r for r in records if r["first_sec"] < t1 and r["last_sec"] >= t0]
        bins.append({
            "t": round(t0, 2),
            "entries": len(entered),
            "exits": len(left),
            "occupancy_avg": round(mean(occ), 2) if occ else 0,
            "occupancy_peak": max(occ) if occ else 0,
            "gender": {g: sum(1 for r in entered if r["gender"] == g) for g in GENDER_LABELS},
            "age": {a: sum(1 for r in entered if r["age"] == a) for a in AGE_LABELS},
            "present": len(present),
        })

    busiest = max(bins, key=lambda b: b["entries"]) if bins else None
    g_conf = [r["gender_conf"] for r in records if r["gender"]]
    a_conf = [r["age_conf"] for r in records if r["age"]]
    dwells = [r["dwell_sec"] for r in records]
    occ_values = list(occupancy_by_frame.values()) or [0]

    kpis = {
        "visitors": total,
        "peak_occupancy": max(occ_values),
        "avg_occupancy": round(mean(occ_values), 2),
        "avg_dwell_sec": round(mean(dwells), 2) if dwells else 0.0,
        "median_dwell_sec": round(median(dwells), 2) if dwells else 0.0,
        "total_dwell_sec": round(sum(dwells), 1),
        "footfall_per_min": round(total / (duration / 60.0), 2) if duration else 0.0,
        "male_pct": _pct(gender_counts["male"], total),
        "female_pct": _pct(gender_counts["female"], total),
        "gender_ratio": (round(gender_counts["male"] / gender_counts["female"], 2)
                         if gender_counts["female"] else None),
        "dominant_age": max(age_counts, key=age_counts.get) if total else None,
        "dominant_age_pct": _pct(max(age_counts.values()) if total else 0, total),
        "avg_gender_conf": round(mean(g_conf), 3) if g_conf else 0.0,
        "avg_age_conf": round(mean(a_conf), 3) if a_conf else 0.0,
        "low_conf_pct": _pct(sum(1 for c in g_conf if c < LOW_CONF), len(g_conf)),
        "engaged_pct": _pct(sum(1 for r in records if r["dwell_sec"] >= 30), total),
        "busiest_window": [busiest["t"], round(busiest["t"] + bin_sec, 2)] if busiest else None,
        "busiest_entries": busiest["entries"] if busiest else 0,
        "tracks_dropped": rejected,
    }

    return {
        "schema": 1,
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "meta": dict(meta, duration_sec=round(duration, 2)),
        "config": dict(config, bin_sec=bin_sec, min_track_frames=min_frames,
                       low_conf_threshold=LOW_CONF),
        "labels": {"gender": GENDER_LABELS, "age": AGE_LABELS,
                   "tiers": [t[0] for t in DWELL_TIERS], "zones": ZONES},
        "kpis": kpis,
        "distributions": {
            "gender": gender_counts,
            "age": age_counts,
            "age_by_gender": cross,
            "tiers": tier_counts,
            "flow": flow_counts,
            "entry_side": {s: sum(1 for r in records if r["entry"] == s)
                           for s in ["left", "right", "top", "bottom", "interior"]},
            "gender_conf_hist": _hist(g_conf),
            "age_conf_hist": _hist(a_conf),
            "dwell_hist": _hist(dwells, n_bins=12, lo=0.0, hi=max(dwells) if dwells else 1.0),
            "dwell_by_age": dwell_by_age,
            "dwell_by_gender": dwell_by_gender,
            "dwell_by_segment": dwell_by_segment,
            "zones": zone_stats,
        },
        "timeline": {"bin_sec": bin_sec, "bins": bins},
        "heatmap": {"cols": heat_cols, "rows": heat_rows, "grid": heat},
        "tracks": records,
        "model": model_card or {},
    }
