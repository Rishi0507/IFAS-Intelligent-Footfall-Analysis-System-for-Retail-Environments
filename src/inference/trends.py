"""Aggregate many runs into the trends a shop owner actually asks about.

A single video answers "what happened in this clip". Questions like "are
Saturdays busier?", "when should I staff up?" or "do more women come on
Fridays?" need footage from many days, so this module works over *all* runs.

The unit of aggregation is the **observed hour at one camera**, not the clip.
Each run pours its arrivals into `(site, camera, date, hour)` buckets that also
record how many minutes were actually watched, and every rate is
arrivals / minutes-observed. That is what makes a three-minute phone clip and a
three-hour export comparable instead of equally weighted - and it is why two
recordings from the same afternoon merge into one afternoon rather than voting
twice.

Everything here stays honest about thin data: every block reports the minutes
and distinct days behind it, and the UI refuses to draw a trend that rests on a
single day.
"""
from __future__ import annotations
import datetime as _dt
from statistics import mean

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GENDERS = ["female", "male"]
AGES = ["Age16-30", "Age31-45", "Age46-60", "AgeAbove61"]

DEFAULT_SITE = "Store"
DEFAULT_CAMERA = "Camera 1"

# A trend across one day is one recording wearing a trenchcoat.
MIN_DAYS_FOR_TREND = 3
MIN_MONTHS_FOR_SEASON = 2
# A day watched for seconds says nothing about that day's rate.
MIN_MINUTES_PER_DAY = 5.0
# Multiplying a 10-second clip up to an hourly rate is arithmetic, not evidence.
MIN_MINUTES_TO_EXTRAPOLATE = 2.0


def _captured(doc):
    raw = doc["meta"].get("captured_at") or doc.get("generated_at")
    if not raw:
        return None
    try:
        return _dt.datetime.fromisoformat(raw)
    except ValueError:
        return None


def _rate_per_hour(doc):
    dur = doc["meta"].get("duration_sec") or 0
    return doc["kpis"]["visitors"] / (dur / 3600.0) if dur else 0.0


def site_of(doc):
    return doc["meta"].get("site") or DEFAULT_SITE


def camera_of(doc):
    return doc["meta"].get("camera") or DEFAULT_CAMERA


def is_comparable(doc):
    """Whether this clip is long enough to quote an hourly rate for on its own."""
    return (doc["meta"].get("duration_sec") or 0) / 60.0 >= MIN_MINUTES_TO_EXTRAPOLATE


# ------------------------------------------------------------------ buckets
def hourly_buckets(run_id, doc):
    """Split one run into `(site, camera, date, hour)` buckets.

    `observed_min` is the slice of that wall-clock hour the camera actually
    watched, so a clip spanning 16:50-17:10 contributes 10 minutes to each of
    two hours rather than pretending to cover both in full.
    """
    start = _captured(doc)
    if start is None:
        return []
    duration = doc["meta"].get("duration_sec") or 0.0
    if duration <= 0:
        return []
    end = start + _dt.timedelta(seconds=duration)
    site, camera = site_of(doc), camera_of(doc)

    slots = {}
    slot = start.replace(minute=0, second=0, microsecond=0)
    while slot < end:
        nxt = slot + _dt.timedelta(hours=1)
        observed = (min(end, nxt) - max(start, slot)).total_seconds() / 60.0
        if observed > 0:
            slots[slot] = {
                "site": site, "camera": camera, "run_id": run_id,
                "date": slot.date().isoformat(), "hour": slot.hour,
                "weekday": WEEKDAYS[slot.weekday()], "month": slot.month,
                "season": SEASONS[slot.month],
                "observed_min": round(observed, 3), "arrivals": 0,
                "dwell_sum": 0.0, "engaged": 0,
                "gender": {g: 0 for g in GENDERS},
                "age": {a: 0 for a in AGES},
            }
        slot = nxt

    # Each visitor lands in the hour they were first seen.
    for t in doc.get("tracks", []):
        when = start + _dt.timedelta(seconds=t.get("first_sec", 0.0))
        key = when.replace(minute=0, second=0, microsecond=0)
        b = slots.get(key)
        if b is None:
            continue
        b["arrivals"] += 1
        b["dwell_sum"] += t.get("dwell_sec", 0.0)
        if t.get("dwell_sec", 0.0) >= 30:
            b["engaged"] += 1
        if t.get("gender") in b["gender"]:
            b["gender"][t["gender"]] += 1
        if t.get("age") in b["age"]:
            b["age"][t["age"]] += 1
    return list(slots.values())


def _rate(buckets):
    minutes = sum(b["observed_min"] for b in buckets)
    arrivals = sum(b["arrivals"] for b in buckets)
    return round(arrivals / minutes * 60.0, 1) if minutes else 0.0


def _group(buckets, key, order=None):
    """Group buckets and rate each group by arrivals per observed hour."""
    out = {}
    for b in buckets:
        out.setdefault(b[key], []).append(b)
    keys = [k for k in (order or sorted(out)) if k in out]
    rows = []
    for k in keys:
        group = out[k]
        arrivals = sum(b["arrivals"] for b in group)
        gender = {g: sum(b["gender"][g] for b in group) for g in GENDERS}
        known = gender["female"] + gender["male"]
        minutes = sum(b["observed_min"] for b in group)
        dwell_sum = sum(b["dwell_sum"] for b in group)
        engaged = sum(b["engaged"] for b in group)
        rows.append({
            "key": k,
            "arrivals": arrivals,
            "avg_dwell_sec": round(dwell_sum / arrivals, 1) if arrivals else 0.0,
            "engaged": engaged,
            "engaged_pct": round(100.0 * engaged / arrivals, 1) if arrivals else 0.0,
            "observed_min": round(minutes, 1),
            "observed_hours": round(minutes / 60.0, 2),
            "per_hour": _rate(group),
            "days": len({b["date"] for b in group}),
            "gender": gender,
            "female_pct": round(100.0 * gender["female"] / known, 1) if known else 0.0,
            "age": {a: sum(b["age"][a] for b in group) for a in AGES},
        })
    return rows


# --------------------------------------------------------------- run rows
def summarise_run(run_id, doc):
    """One row per run - the listing view, not the maths."""
    when = _captured(doc)
    k = doc["kpis"]
    tracks = doc.get("tracks", [])
    engaged = sum(1 for t in tracks if t["dwell_sec"] >= 30)
    return {
        "id": run_id,
        "source": doc["meta"].get("source", run_id),
        "site": site_of(doc),
        "camera": camera_of(doc),
        "captured_at": when.isoformat(timespec="minutes") if when else None,
        "date": when.date().isoformat() if when else None,
        "weekday": WEEKDAYS[when.weekday()] if when else None,
        "hour": when.hour if when else None,
        "month": when.month if when else None,
        "season": SEASONS[when.month] if when else None,
        "duration_sec": doc["meta"].get("duration_sec", 0),
        "observed_min": round((doc["meta"].get("duration_sec") or 0) / 60.0, 2),
        "visitors": k["visitors"],
        "per_hour": round(_rate_per_hour(doc), 1),
        "avg_dwell_sec": k["avg_dwell_sec"],
        "engaged_pct": round(100.0 * engaged / len(tracks), 1) if tracks else 0.0,
        "female_pct": k["female_pct"],
        "male_pct": k["male_pct"],
        "age": {a: sum(1 for t in tracks if t["age"] == a) for a in AGES},
        "has_age": any(t["age"] for t in tracks),
        "demo": bool(doc["meta"].get("demo")),
        "comparable": is_comparable(doc),
    }


def build_trends(runs, include_demo=False, site=None, camera=None):
    """runs: iterable of (run_id, analytics_doc).

    `site` / `camera` scope the aggregation - rates from a doorway camera and
    an aisle camera are not comparable, so mixing them is opt-in, not default.
    """
    runs = list(runs)
    all_rows = [summarise_run(rid, doc) for rid, doc in runs]
    sources = {
        "real_runs": sum(1 for r in all_rows if not r["demo"]),
        "sample_runs": sum(1 for r in all_rows if r["demo"]),
        "samples_included": bool(include_demo),
    }
    rows = all_rows
    if not include_demo:
        rows = [r for r in rows if not r["demo"]]
        runs = [(rid, doc) for rid, doc in runs if not doc["meta"].get("demo")]

    sites = sorted({r["site"] for r in rows})
    cameras = sorted({r["camera"] for r in rows})
    if site:
        runs = [(rid, doc) for rid, doc in runs if site_of(doc) == site]
        rows = [r for r in rows if r["site"] == site]
    if camera:
        runs = [(rid, doc) for rid, doc in runs if camera_of(doc) == camera]
        rows = [r for r in rows if r["camera"] == camera]
    rows.sort(key=lambda r: r["captured_at"] or "")

    buckets = [b for rid, doc in runs for b in hourly_buckets(rid, doc)]
    days = sorted({b["date"] for b in buckets})
    months = {b["month"] for b in buckets}
    observed_min = sum(b["observed_min"] for b in buckets)

    # Daily rates drive the trend line; days barely watched are not a data point.
    by_day = [d for d in _group(buckets, "date")
              if d["observed_min"] >= MIN_MINUTES_PER_DAY]

    trend = None
    if len(by_day) >= 2:
        latest, earlier = by_day[-1], by_day[:-1]
        prev = mean(d["per_hour"] for d in earlier)
        trend = {
            "latest_per_hour": latest["per_hour"],
            "latest_date": latest["key"],
            "previous_avg_per_hour": round(prev, 1),
            "change_pct": round(100.0 * (latest["per_hour"] - prev) / prev, 1) if prev else None,
            "from": by_day[0]["key"],
            "to": latest["key"],
        }

    return {
        "runs": rows,
        "sources": sources,
        "sites": sites,
        "cameras": cameras,
        "scope": {"site": site, "camera": camera},
        "coverage": {
            "runs": len(rows),
            "buckets": len(buckets),
            "days": len(days),
            "usable_days": len(by_day),
            "months": len(months),
            "observed_min": round(observed_min, 1),
            "observed_hours": round(observed_min / 60.0, 2),
            "sites": len(sites),
            "cameras": len(cameras),
            "enough_for_trend": len(by_day) >= MIN_DAYS_FOR_TREND,
            "enough_for_season": len(months) >= MIN_MONTHS_FOR_SEASON,
            "min_days_for_trend": MIN_DAYS_FOR_TREND,
            "min_months_for_season": MIN_MONTHS_FOR_SEASON,
            "mixed_cameras": len(cameras) > 1 and not camera,
        },
        "by_weekday": _group(buckets, "weekday", WEEKDAYS),
        "by_hour": _group(buckets, "hour", list(range(24))),
        "by_season": _group(buckets, "season", ["Spring", "Summer", "Autumn", "Winter"]),
        "by_month": _group(buckets, "month", list(range(1, 13))),
        "by_day": by_day,
        "trend": trend,
    }


# ---------------------------------------------------------------- insights
def insights(doc):
    """Plain-language, single-run observations a shop owner can act on.

    Deliberately conservative: each line states what was measured, not a
    prediction, and anything that needs more than one recording is left to the
    trends block above.
    """
    k = doc["kpis"]
    tracks = doc.get("tracks", [])
    out = []
    if not tracks:
        return [{"kind": "none", "text": "No people were detected in this footage."}]

    dur_min = (doc["meta"].get("duration_sec") or 0) / 60.0
    per_hour = _rate_per_hour(doc)
    bin_sec = doc["timeline"]["bin_sec"]
    bins = doc["timeline"]["bins"]
    busiest = max(bins, key=lambda b: b["entries"])
    quietest = min(bins, key=lambda b: b["entries"])

    out.append({
        "kind": "traffic",
        "label": "Footfall",
        "text": (f"{k['visitors']} people in {dur_min:.1f} minutes - about "
                 f"{per_hour:.0f} an hour if this pace held."
                 if dur_min >= MIN_MINUTES_TO_EXTRAPOLATE else
                 f"{k['visitors']} "
                 f"{'person' if k['visitors'] == 1 else 'people'} in {dur_min * 60:.0f} seconds "
                 f"of footage - too short to say anything about an hourly rate."),
    })

    clock = f"{int(busiest['t'] // 60)}:{int(busiest['t'] % 60):02d}"
    if quietest["entries"] == 0:
        peak_text = (f"The busiest {int(bin_sec)} seconds started at {clock} with "
                     f"{busiest['entries']} arrivals, while the quietest stretches had none.")
    elif busiest["entries"] / quietest["entries"] >= 2:
        peak_text = (f"The busiest {int(bin_sec)} seconds started at {clock} with "
                     f"{busiest['entries']} arrivals - "
                     f"{busiest['entries'] / quietest['entries']:.0f}x the quietest stretch.")
    else:
        peak_text = f"Arrivals were fairly even, peaking at {clock}."
    out.append({
        "kind": "staffing",
        "label": "Busiest stretch",
        "text": peak_text,
        "window": [busiest["t"], busiest["t"] + bin_sec],
    })

    passing = sum(1 for t in tracks if t["dwell_sec"] < 3)
    lingering = sum(1 for t in tracks if t["dwell_sec"] >= 30)
    out.append({
        "kind": "engagement",
        "label": "Stopping power",
        "text": f"{100 * lingering / len(tracks):.0f}% stayed more than 30 seconds and "
                f"{100 * passing / len(tracks):.0f}% passed straight through - "
                f"the typical visit was {k['median_dwell_sec']:.0f} seconds.",
    })

    gender = "women" if k["female_pct"] >= k["male_pct"] else "men"
    share = max(k["female_pct"], k["male_pct"])
    line = f"{share:.0f}% of visitors were {gender}"
    if k.get("dominant_age") and any(t["age"] for t in tracks):
        pretty = {"Age16-30": "16-30", "Age31-45": "31-45",
                  "Age46-60": "46-60", "AgeAbove61": "61+"}[k["dominant_age"]]
        line += f", and the biggest age group was {pretty} ({k['dominant_age_pct']:.0f}%)"
    out.append({"kind": "audience", "label": "Who visits", "text": line + "."})

    zones = doc["distributions"]["zones"]
    top = max(zones, key=lambda z: zones[z]["visitors"])
    if zones[top]["visitors"]:
        share = 100.0 * zones[top]["visitors"] / len(tracks)
        out.append({
            "kind": "layout",
            "label": "Busiest area",
            "text": f"The {top.lower()} of the camera view drew {share:.0f}% of visitors, "
                    f"who stayed {zones[top]['avg_dwell']:.0f} seconds on average.",
        })

    longest = max(
        (s for s in doc["distributions"]["dwell_by_segment"] if s["n"] >= max(3, 0.05 * len(tracks))),
        key=lambda s: s["avg"], default=None)
    if longest:
        pretty = {"Age16-30": "16-30", "Age31-45": "31-45",
                  "Age46-60": "46-60", "AgeAbove61": "61+"}[longest["age"]]
        out.append({
            "kind": "segment",
            "label": "Stays longest",
            "text": f"{longest['gender'].capitalize()} visitors aged {pretty} stayed longest "
                    f"({longest['avg']:.0f} seconds on average, {longest['n']} people).",
        })
    return out
