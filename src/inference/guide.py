"""A plain-language guide to one recording, for the person who runs the place.

Every other part of the analytics document is written for someone who reads
charts. This one is written for a shop owner: what happened, what it means,
what to try, and how far to trust it. Each section carries the numbers it is
based on, so nothing here is a claim without evidence, and a confidence level
with the reason for it.

    guide = shop_guide(analytics_doc)
    {"headline": "...", "setting": {...}, "sections": [{"id", "title", "headline", "explain",
     "do": [...], "numbers": {...}, "confidence", "why",
     "stat": {"value", "label"}, "sub", "tip", "brief", "proof": {...}}], "glossary": [...]}

The first page shows only stat, sub, tip and proof; the rest is one click
away. `proof` names the evidence (proof.py fetches it from the recording): a
moment to show with everyone counted then ("frame"), people's pictures
("people"), tracks that ended and began again in the same spot ("handoffs"),
or where people spent time ("heat").

Suggestions are deliberately modest ("try", "consider") and always tied to a
number from this recording. One recording is one day: anything about patterns
over days, weeks or months belongs to the Trends tab.
"""
from __future__ import annotations

import math
from statistics import median

from .proof import occupancy_peak

SCHEMA = 2

PASSING_SEC, BROWSING_SEC, ENGAGED_SEC = 3.0, 30.0, 60.0
MIN_PEOPLE = 20          # fewer than this: counts, not percentages


def _dur(sec):
    """Seconds as words: '45 seconds', '3 minutes', '1 hour 5 minutes'."""
    sec = float(sec or 0)
    if sec < 90:
        n = int(round(sec))
        return f"{n} second{'s' if n != 1 else ''}"
    m = int(round(sec / 60.0))
    if m < 90:
        return f"{m} minute{'s' if m != 1 else ''}"
    h, m = divmod(m, 60)
    return f"{h} hour{'s' if h != 1 else ''}" + (f" {m} minute{'s' if m != 1 else ''}" if m else "")


def _short_dur(sec):
    """Seconds as a figure: '45s', '3m 20s', '12m', '1h 5m'."""
    sec = int(round(float(sec or 0)))
    if sec < 90:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 10:
        return f"{m}m {s}s" if s else f"{m}m"
    if m < 90:
        return f"{m}m"
    h, m = divmod(m, 60)
    return f"{h}h {m}m" if m else f"{h}h"


def _clock(doc, t):
    """A moment in the recording as a time of day when the start is known."""
    start = doc.get("meta", {}).get("captured_at")
    if start:
        try:
            import datetime as dt
            when = dt.datetime.fromisoformat(start) + dt.timedelta(seconds=float(t))
            return when.strftime("%H:%M")
        except ValueError:
            pass
    m, s = divmod(int(t), 60)
    return f"{m}:{s:02d} into the recording"


def _pct(x):
    return f"{100 * x:.0f}%"


def setting_of(doc):
    """Walk-through shop, or a place where people sit (café, restaurant,
    classroom, waiting area)? It changes which numbers matter."""
    duration = doc["meta"].get("duration_sec") or 0.0
    tracks = doc.get("tracks", [])
    dwell = [t["dwell_sec"] for t in tracks]
    med = median(dwell) if dwell else 0.0
    z = doc.get("zones") or {}
    seats = (z.get("summary") or {}).get("seats_tables", 0)
    util = (z.get("summary") or {}).get("mean_utilization", 0.0)
    # Typical stay a large part of the recording and at least a minute, or
    # half of a recording shorter than two minutes (a short clip of a lecture).
    seated = bool(seats and util >= 0.3) or (duration and med >= 0.3 * duration and med >= min(60, 0.5 * duration))
    if seated:
        return {"kind": "seated", "label": "People sat and stayed",
                "text": "People mostly sat and stayed, as in a café, restaurant, classroom or waiting area, "
                        "so seats and how long people stay matter more than how many walked past."}
    return {"kind": "walk-through", "label": "People came and went",
            "text": "People mostly came and went, as in a shop, so how many came in, when, whether they "
                    "stopped, and where they went are the numbers that matter."}


def _identity_inflation(doc):
    """Tracks per person in view at the busiest moment, when that is high for a
    seated room: the tracker split people, so head counts are too high."""
    k = doc["kpis"]
    peak = k.get("peak_occupancy") or 0
    if not peak:
        return None
    ratio = k["visitors"] / peak
    return ratio


def _in_view_series(doc, target=40):
    """People in view over the recording, in at most `target` steps."""
    bins = doc["timeline"]["bins"]
    if not bins:
        return None
    per = max(1, -(-len(bins) // target))
    pts = []
    for i in range(0, len(bins), per):
        chunk = bins[i:i + per]
        pts.append({"t": chunk[0]["t"], "clock": _clock(doc, chunk[0]["t"]),
                    "v": round(sum(b.get("occupancy_avg", 0.0) for b in chunk) / len(chunk), 1)})
    return {"label": "People in view through the recording", "unit": "people", "points": pts}


# ---------------------------------------------------------------- sections
def _busy(doc, setting):
    k = doc["kpis"]
    bins = doc["timeline"]["bins"]
    peak_bin = max(bins, key=lambda b: b.get("occupancy_peak", 0)) if bins else None
    avg, peak = k.get("avg_occupancy", 0.0), k.get("peak_occupancy", 0)
    burst = peak / avg if avg else 0.0
    # The exact moment, from the same first-to-last-sighting count the peak
    # comes from; the 5 s bin when the tracks do not reproduce it.
    t_peak, n_peak = occupancy_peak(doc.get("tracks", []))
    if n_peak != peak or not doc.get("tracks"):
        t_peak = peak_bin["t"] if peak_bin else None
    do = []
    if burst >= 1.5:
        do.append(f"Crowds come in bursts (up to {peak} people against {avg:.0f} on average). Keep one person "
                  f"free to help during the bursts rather than spreading staff evenly.")
    else:
        do.append(f"The room stayed about as full all the time ({avg:.0f} on average, {peak} at most), so steady "
                  f"staffing fits better than extra cover at one moment.")
    if setting["kind"] == "walk-through" and peak:
        do.append("Count how many people one staff member can comfortably serve, and plan staff for the "
                  f"busiest moment ({peak} at once).")
    return {
        "id": "busy", "title": "How busy it was",
        "headline": f"About {avg:.0f} people were in view at any moment, {peak} at the busiest"
                    + (f", around {_clock(doc, t_peak)}." if t_peak is not None else "."),
        "explain": "This counts people visible at the same time, like looking up and counting heads. It does "
                   "not depend on telling people apart, so it is one of the more reliable numbers here.",
        "do": do,
        "numbers": {"average_in_view": avg, "peak_in_view": peak, "peak_at_sec": t_peak},
        "series": _in_view_series(doc),
        "confidence": "solid",
        "why": "Counting heads at one moment only needs the detector to see people, not to follow them.",
        "stat": {"value": f"{peak}", "label": "people at the busiest"},
        "sub": (f"{_clock(doc, t_peak)} · " if t_peak is not None else "") + f"{avg:.0f} on average",
        "tip": ("Crowds come in bursts: keep one person free for them." if burst >= 1.5
                else "About as full all the time: steady staffing fits."),
        "brief": "Counts heads at one moment, so it does not need to tell people apart.",
        "proof": ({"kind": "frame", "t": t_peak, "show": "people",
                   "caption": "The busiest moment, with everyone counted"} if t_peak is not None else None),
    }


def _arrivals(doc, setting, inflation):
    k = doc["kpis"]
    bins = doc["timeline"]["bins"]
    bin_sec = doc["timeline"]["bin_sec"]
    duration = doc["meta"].get("duration_sec") or 0.0
    tracks = doc.get("tracks", [])
    already = sum(1 for t in tracks if t.get("first_sec", 1.0) < 1.0)
    entries = [dict(b) for b in bins]
    if entries and already:
        entries[0]["entries"] = max(0, entries[0]["entries"] - already)
    # 5-minute windows read better than 5-second bins for planning.
    per = max(1, int(round(300.0 / bin_sec))) if duration >= 900 else max(1, int(round(60.0 / bin_sec)))
    windows = []
    for i in range(0, len(entries), per):
        chunk = entries[i:i + per]
        windows.append((chunk[0]["t"], sum(b["entries"] for b in chunk)))
    span = "5-minute" if duration >= 900 else "1-minute"
    arrived = sum(w[1] for w in windows)
    confidence, why = "rough", "Arrivals depend on following each person; the tracker can split or merge people."
    if inflation and inflation > 3 and setting["kind"] == "seated":
        confidence = "unreliable"
        why = (f"In this crowded, seated room the tracker split people: {k['visitors']} tracks for at most "
               f"{k['peak_occupancy']} people in view ({inflation:.1f} per person). Treat arrival counts as "
               f"far too high, and use 'How busy it was' instead.")
    do = []
    if windows and arrived:
        best = max(windows, key=lambda w: w[1])
        quiet = min(windows, key=lambda w: w[1])
        do.append(f"The busiest {span} stretch began at {_clock(doc, best[0])} ({best[1]} arrivals). If this "
                  f"happens at the same time on other days, that is when to have the most staff on the floor.")
        if quiet[1] < best[1]:
            do.append(f"The quietest {span} stretch began at {_clock(doc, quiet[0])} ({quiet[1]} arrivals): "
                      f"a good time for restocking, breaks or a small offer to bring people in.")
    if confidence != "unreliable" and setting["kind"] == "seated" and duration < 300:
        confidence = "unreliable"
        why = ("In a few minutes of a seated room, almost every new track is someone the tracker lost behind "
               "another person and found again, not someone walking in.")
    if confidence == "unreliable":
        # Advice built on a number we just called unreliable would mislead.
        do = ["Plan staff from 'How busy it was', which counts heads and does not depend on following people."]
    if duration < 1800:
        do.append(f"This recording is only {_dur(duration)} long. Process a few full days, and the Trends tab "
                  f"will show which hours and weekdays are reliably busy.")
    headline = (f"{arrived} people arrived during the recording" + (f", and {already} were already there when it "
                f"started." if already else ".")) if windows else "No arrivals were recorded."
    best = max(windows, key=lambda w: w[1]) if windows and arrived else None
    if confidence == "unreliable":
        proof = {"kind": "handoffs", "caption": "A track ended and a new one began in the same spot"}
        tip = "Plan from 'How busy it was' instead."
        brief = (f"{k['visitors']} tracks for at most {k['peak_occupancy']} people in view: people were split."
                 if inflation and inflation > 3 else "Too short to tell arrivals from people found again.")
    else:
        firsts = sorted((t for t in tracks if best and best[0] <= t["first_sec"] < best[0] + per * bin_sec
                         and t["first_sec"] >= 1.0), key=lambda t: t["first_sec"])[:6]
        proof = ({"kind": "people", "tracks": [t["id"] for t in firsts], "show": "arrived",
                  "caption": "The first arrivals of the busiest stretch"} if firsts else None)
        tip = (f"If {_clock(doc, best[0])} is busy on other days too, staff up then." if best
               else "Process longer recordings to see when people come.")
        brief = "Depends on following each person through the recording."
    return {
        "id": "arrivals", "title": "When people arrived",
        "headline": headline,
        "explain": "An arrival is someone the system starts following. People already in view when the "
                   "recording starts are not counted as arrivals, because they came in earlier.",
        "do": do,
        "numbers": {"arrivals": arrived, "already_there": already, "window": span},
        "series": {"label": f"Arrivals per {span.replace('-', ' ')} stretch", "unit": "arrivals",
                   "points": [{"t": t, "clock": _clock(doc, t), "v": n} for t, n in windows][:96]},
        "confidence": confidence, "why": why,
        "stat": {"value": f"{arrived}", "label": "arrived"},
        "sub": (f"Busiest {span} stretch from {_clock(doc, best[0])} ({best[1]})" if best else
                "Nobody arrived") + (f" · {already} already there" if already else ""),
        "tip": tip, "brief": brief, "proof": proof,
    }


def _cut_off(doc, t, slack=1.0):
    """In view when the recording started or still in view when it ended:
    the stay was longer than measured."""
    duration = doc["meta"].get("duration_sec") or 0.0
    return t.get("first_sec", 1e9) < slack or t.get("last_sec", 0.0) > duration - slack


def _stopping(doc, setting, inflation):
    tracks = doc.get("tracks", [])
    if not tracks:
        return None
    n = len(tracks)
    cut = sum(1 for t in tracks if _cut_off(doc, t))
    duration = doc["meta"].get("duration_sec") or 0.0
    if duration < 60 or cut > 0.5 * n:
        # Most stays were cut off by the recording: their length is unknown.
        return {
            "id": "stopping", "title": "Did people stop?",
            "headline": f"Too short to tell: {cut} of {n} {'person was' if n == 1 else 'people were'} already "
                        f"in view when the recording started or still there when it ended.",
            "explain": "A stay is measured from first to last sighting. When the recording starts or stops "
                       "in the middle of someone's visit, the real stay is longer than what was seen.",
            "do": ["Record longer stretches, ideally whole opening hours, to see how long people really stay."],
            "numbers": {"people": n, "cut_off_by_recording": cut},
            "confidence": "unreliable",
            "why": "Most stays here were cut off by the start or end of the recording.",
            "stat": {"value": f"{cut}/{n}", "label": "stays cut off by the recording"},
            "sub": "Too short to tell how long people stay",
            "tip": "Record whole opening hours.",
            "brief": "Most stays were cut off by the start or end of the recording.",
            "proof": None,
        }
    passing = sum(1 for t in tracks if t["dwell_sec"] < PASSING_SEC) / n
    browsing = sum(1 for t in tracks if PASSING_SEC <= t["dwell_sec"] < BROWSING_SEC) / n
    engaged = sum(1 for t in tracks if t["dwell_sec"] >= BROWSING_SEC) / n
    med = median(t["dwell_sec"] for t in tracks)
    do = []
    if setting["kind"] == "walk-through":
        if passing >= 0.4:
            do.append(f"{_pct(passing)} walked straight past. Something eye-catching near the entrance (a "
                      f"display, an offer, better lighting) is the cheapest way to turn passers-by into visitors.")
        if engaged >= 0.3:
            do.append(f"{_pct(engaged)} stayed longer than 30 seconds. These are your most interested visitors: "
                      f"make sure someone is free to help them, as that is when sales are won or lost.")
        if browsing >= 0.4:
            do.append(f"Most people ({_pct(browsing)}) looked around briefly. Clear signs and prices help "
                      f"quick browsers decide without needing staff.")
    else:
        do.append(f"Half of the stays were longer than {_dur(med)}. In a seated place, long stays are normal; "
                  f"the seats section shows how often places changed hands.")
    confidence = "rough"
    why = "Time in view is measured per person, so it inherits the tracker's mistakes."
    brief = "Depends on following each person through the recording."
    if inflation and inflation > 3 and setting["kind"] == "seated":
        confidence = "unreliable"
        why = "The tracker split people in this crowded room, which cuts stays into short pieces."
        brief = "Split tracks cut stays into short pieces."
    if setting["kind"] != "walk-through":
        tip = "Long stays are normal in a seated place."
    elif passing >= 0.4:
        tip = "Many walk past: try something eye-catching at the entrance."
    elif engaged >= 0.3:
        tip = "Many stay 30 s or more: keep someone free to help them."
    elif browsing >= 0.4:
        tip = "Quick browsers: clear signs and prices help them decide."
    else:
        tip = "Compare across days to see what keeps people longer."
    by_stay = sorted((t for t in tracks if not _cut_off(doc, t)), key=lambda t: t["dwell_sec"])
    examples = ([by_stay[0], by_stay[len(by_stay) // 2], by_stay[-1]] if len(by_stay) >= 3 else by_stay)
    return {
        "id": "stopping", "title": "Did people stop?",
        "headline": f"Half of the people stayed longer than {_dur(med)}; {_pct(engaged)} stayed more than "
                    f"30 seconds and {_pct(passing)} passed straight through.",
        "explain": "Time in view is the only sign of interest a camera gives. It is not attention: someone "
                   "waiting for a friend also stays a long time.",
        "do": do,
        "numbers": {"median_stay_sec": med, "passed_through": round(passing, 3), "looked_briefly": round(browsing, 3),
                    "stayed_30s_plus": round(engaged, 3)},
        "confidence": confidence, "why": why,
        "stat": {"value": _short_dur(med), "label": "typical stay"},
        "sub": f"{_pct(engaged)} stayed 30 s or more · {_pct(passing)} passed through",
        "tip": tip, "brief": brief,
        "shares": [{"label": "Passed through", "share": round(passing, 3)},
                   {"label": "Looked briefly", "share": round(browsing, 3)},
                   {"label": "Stayed 30 s+", "share": round(engaged, 3)}],
        "proof": ({"kind": "people", "tracks": [t["id"] for t in examples], "show": "stay",
                   "caption": "The shortest, a typical and the longest stay"}
                  if examples and confidence != "unreliable" else None),
    }


def _where(doc):
    zones = (doc.get("distributions") or {}).get("zones") or {}
    total = sum(v["visitors"] for v in zones.values())
    if not total:
        return None
    if total < MIN_PEOPLE:
        where = {"Left": "on the left", "Center": "in the middle", "Right": "on the right"}
        counts = ", ".join(f"{v['visitors']} {where.get(k, k)}" for k, v in zones.items())
        return {
            "id": "where", "title": "Where people went",
            "headline": f"Too few people to compare areas ({counts}).",
            "explain": "The view is split into three equal strips, left, middle and right. With only a handful "
                       "of people, one person more or less swings the comparison.",
            "do": [f"Compare areas once a recording has at least {MIN_PEOPLE} people in it.",
                   "The Floor tab still shows exactly where these people stood and walked."],
            "numbers": {k: {"people": v["visitors"]} for k, v in zones.items()},
            "confidence": "unreliable", "why": f"Fewer than {MIN_PEOPLE} people.",
            "stat": {"value": f"{total}", "label": "people: too few to compare areas"},
            "sub": counts[:1].upper() + counts[1:], "tip": f"Compare areas with {MIN_PEOPLE} people or more.",
            "brief": f"Fewer than {MIN_PEOPLE} people.",
            "proof": {"kind": "heat", "t": (doc["meta"].get("duration_sec") or 0.0) / 2,
                      "caption": "Where people spent time"},
        }
    ranked = sorted(zones.items(), key=lambda kv: -kv[1]["visitors"])
    hot, cold = ranked[0], ranked[-1]
    names = {"Left": "left side", "Center": "middle", "Right": "right side"}
    spot = {"Left": "on the left", "Center": "in the middle", "Right": "on the right"}
    do = [f"The {names.get(hot[0], hot[0])} of the view drew the most people ({_pct(hot[1]['visitors'] / total)}). "
          f"Put best-sellers and anything you want noticed there."]
    if cold[1]["visitors"] / total < 0.25:
        do.append(f"The {names.get(cold[0], cold[0])} was quiet ({_pct(cold[1]['visitors'] / total)}). Better "
                  f"lighting, a sign, or moving a popular item there can pull people across.")
    do.append("The Floor tab shows exactly where people stood and walked, as a heat map you can replay.")
    return {
        "id": "where", "title": "Where people went",
        "headline": f"Most people were on the {names.get(hot[0], hot[0])} of the view; the "
                    f"{names.get(cold[0], cold[0])} saw the fewest.",
        "explain": "The view is split into three equal strips, left, middle and right, and each person is "
                   "counted in the strip where they spent most of their time.",
        "do": do,
        "numbers": {k: {"people": v["visitors"], "share": round(v["visitors"] / total, 3),
                        "average_stay_sec": v.get("avg_dwell")} for k, v in zones.items()},
        "confidence": "rough",
        "why": "Based on where each tracked person was; the strips are parts of the picture, not of the floor.",
        "stat": {"value": _pct(hot[1]["visitors"] / total), "label": f"of people were {spot.get(hot[0], hot[0])}"},
        "sub": f"The {names.get(cold[0], cold[0])} saw {_pct(cold[1]['visitors'] / total)}",
        "tip": f"Put what you want noticed {spot.get(hot[0], hot[0])}.",
        "brief": "The strips are thirds of the picture, not of the floor.",
        "proof": {"kind": "heat", "t": (doc["meta"].get("duration_sec") or 0.0) / 2,
                  "caption": "Where people spent time, over the whole recording"},
    }


def _seats(doc, setting):
    z = doc.get("zones")
    if not z or not z.get("summary", {}).get("seats_tables"):
        return None
    s = z["summary"]
    kind = ("seats" if all(x["kind"] == "seat" for x in z["zones"]) else
            "tables" if all(x["kind"] == "table" for x in z["zones"]) else "places")
    util = s["mean_utilization"]
    turnover = [x["metrics"]["turnover_per_hour"] for x in z["zones"] if x["kind"] in ("seat", "table")]
    busiest = max((x for x in z["zones"] if x["kind"] in ("seat", "table")),
                  key=lambda x: x["metrics"]["utilization"])
    do = []
    if util >= 0.8:
        do.append(f"{kind.capitalize()} were taken {_pct(util)} of the time: close to full. If people are "
                  f"turned away or wait at busy times, more {kind} or faster turnover at peaks would pay off.")
    elif util <= 0.4:
        do.append(f"{kind.capitalize()} were taken only {_pct(util)} of the time. Consider fewer, better-placed "
                  f"{kind}, and use the space for display.")
    else:
        do.append(f"{kind.capitalize()} were taken {_pct(util)} of the time: busy but not full.")
    if s.get("free_at_end"):
        do.append(f"{s['free_at_end']} of {s['seats_tables']} {kind} were free when the recording ended; the "
                  f"Zones tab shows which, and which is likely to free up next.")
    if s.get("unassigned_share", 0) > 0.15:
        do.append(f"{_pct(s['unassigned_share'])} of sightings fell outside every {kind[:-1]}: some places are "
                  f"missing. Draw them in the Zones tab for a complete picture.")
    lab = z.get("validity", {}).get("labels") if isinstance(z.get("validity"), dict) else None
    # The last sampled moment: "at the end" is judged there.
    step = z.get("sample_sec") or 0.2
    end = z.get("duration_sec") or doc["meta"].get("duration_sec") or 0.0
    t_end = max(0.0, math.floor((end - 1e-6) / step) * step)
    tip = (f"Close to full: more {kind} or faster turnover at peaks would pay off." if util >= 0.8 else
           f"Under-used: try fewer, better-placed {kind}." if util <= 0.4 else "Busy but not full.")
    confidence, why = "rough", ("Taken and free are judged from people the detector sees. In a crowded room it "
                                "misses some, so a few taken seats can show as free.")
    return {
        "id": "seats", "title": f"{kind.capitalize()}",
        "headline": f"{s['occupied_at_end']} of {s['seats_tables']} {kind} were taken at the end; on average each "
                    f"was taken {_pct(util)} of the time, and the busiest ({busiest['name']}) "
                    f"{_pct(busiest['metrics']['utilization'])}.",
        "explain": f"Each {kind[:-1]} is an area drawn on the camera picture (or found automatically from where "
                   f"people sat). It counts as taken while someone is in it; short gaps, like leaning out of "
                   f"view, are bridged so it does not flicker.",
        "do": do,
        "numbers": {"places": s["seats_tables"], "taken_at_end": s["occupied_at_end"], "mean_utilisation": util,
                    "median_stay_sec": s.get("median_session_sec"),
                    "turnover_per_hour_median": median(turnover) if turnover else None,
                    "labels": lab},
        "confidence": confidence, "why": why,
        "stat": {"value": f"{s['occupied_at_end']}/{s['seats_tables']}", "label": f"{kind} taken at the end"},
        "sub": f"Taken {_pct(util)} of the time · busiest {busiest['name']} ({_pct(busiest['metrics']['utilization'])})",
        "tip": tip,
        "brief": "Taken means someone was seen in it; someone hidden can make a taken seat look free.",
        "proof": {"kind": "frame", "t": round(t_end, 2), "show": "zones",
                  "caption": f"The last moment of the recording, {kind} coloured by state"},
    }


def _queues(doc):
    z = doc.get("zones")
    qs = [x for x in (z or {}).get("zones", []) if x["kind"] == "queue" and x.get("queue")]
    if not qs:
        return None
    worst = max(qs, key=lambda q: q["queue"]["max_wait_sec"])
    q = worst["queue"]
    busiest_q = max(worst.get("sessions") or [], key=lambda x: (x.get("peak", 0), x["duration"]), default=None)
    do = []
    if q["max_wait_sec"] >= 180:
        do.append(f"Someone waited {_dur(q['max_wait_sec'])} at {worst['name']}. Opening a second counter, or "
                  f"having someone take orders in the line, at those moments would cut it.")
    do.append(f"At most {q['max_length']} people were waiting at once; the Zones tab shows when.")
    return {
        "id": "queues", "title": "Waiting",
        "headline": f"People waited {_dur(q['median_wait_sec'])} typically at {worst['name']}, "
                    f"{_dur(q['max_wait_sec'])} at worst.",
        "explain": "A wait is the time a person spent inside an area drawn as a queue.",
        "do": do, "numbers": {"queue": worst["name"], **q, "waits": None},
        "confidence": "rough", "why": "Waits follow individual people, so they inherit the tracker's mistakes.",
        "stat": {"value": _short_dur(q["median_wait_sec"]), "label": f"typical wait at {worst['name']}"},
        "sub": f"{_short_dur(q['max_wait_sec'])} at worst · up to {q['max_length']} waiting",
        "tip": ("Open a second counter, or take orders in the line, at peaks." if q["max_wait_sec"] >= 180
                else "Waits stayed short."),
        "brief": "Depends on following each person through the recording.",
        "proof": ({"kind": "frame", "t": round((busiest_q["start"] + busiest_q["end"]) / 2, 1), "show": "zones",
                   "focus": worst["id"], "caption": f"{worst['name']} at its busiest"} if busiest_q else None),
    }


def _who(doc):
    k = doc["kpis"]
    tracks = doc.get("tracks", [])
    if not tracks:
        return None
    known_g = [t for t in tracks if t.get("gender")]
    known_a = [t for t in tracks if t.get("age")]
    unknown_g = 1 - len(known_g) / len(tracks)
    card = doc.get("model") or {}
    g_test = ((card.get("gender") or {}).get("test") or {}).get("balanced_accuracy")
    a_test = ((card.get("age") or {}).get("test") or {}).get("balanced_accuracy")
    parts = []
    small = len(known_g) < MIN_PEOPLE
    if known_g and small:
        w = sum(1 for t in known_g if t["gender"] == "female")
        parts.append(f"{w} {'woman' if w == 1 else 'women'} and {len(known_g) - w} "
                     f"{'man' if len(known_g) - w == 1 else 'men'} among the people the system was sure about")
    elif known_g:
        f = sum(1 for t in known_g if t["gender"] == "female") / len(known_g)
        parts.append(f"{_pct(f)} women and {_pct(1 - f)} men among the {len(known_g)} people the system was "
                     f"sure about")
    groups = {}
    if known_a and not small:
        for t in known_a:
            groups[t["age"]] = groups.get(t["age"], 0) + 1
        top = max(groups, key=groups.get)
        parts.append(f"the largest age group {top.replace('Age', '').replace('Above61', 'over 60')} "
                     f"({_pct(groups[top] / len(known_a))})")
    do = ["Use this as a rough guide to who visits, never to judge an individual."]
    if unknown_g >= 0.2:
        do.append(f"The system said 'not sure' for {_pct(unknown_g)} of people rather than guess. Cameras "
                  f"placed at head height, facing the door, give it much better views.")
    confidence = "unreliable" if small else "rough"
    why = "Gender and age are guessed from small, low-resolution pictures, often from behind or above."
    if known_g and not small:
        f = sum(1 for t in known_g if t["gender"] == "female") / len(known_g)
        stat = {"value": _pct(max(f, 1 - f)),
                "label": f"{'women' if f > 0.5 else 'men'}, of the {len(known_g)} it was sure about"
                         if unknown_g else f"{'women' if f > 0.5 else 'men'}, of {len(known_g)} people"}
    else:
        stat = {"value": f"{len(known_g)}", "label": "people it was sure about"}
    age_bits = []
    if groups:
        top_age = max(groups, key=groups.get)
        age_bits.append(f"Most often {top_age.replace('Age', '').replace('Above61', 'over 60').replace('-', '–')}")
    if unknown_g:
        age_bits.append(f"not sure about {_pct(unknown_g)}")
    sub = " · ".join(age_bits)
    # Examples where the camera had a good view: the most confident detections
    # among people judged several times. (Seen longest would favour anything
    # that never moves, a bag included.)
    clearest = sorted((t for t in tracks if (t.get("classified_frames") or 0) >= 3),
                      key=lambda t: (-(t.get("det_conf") or 0.0), -t["dwell_sec"]))[:10]
    if small:
        why = f"Fewer than {MIN_PEOPLE} people: too few to describe who visits. " + why
    if a_test is not None and a_test < 0.5:
        do.append("Age guesses are weak from CCTV angles; do not base decisions on them.")
    if g_test is not None:
        why += f" On held-out test pictures the gender model was right {_pct(g_test)} of the time (balanced)."
    return {
        "id": "who", "title": "Who visited",
        "headline": ("Roughly " + ", and ".join(parts) + ".") if parts else "No reliable guesses about who visited.",
        "explain": "Each person is judged on several pictures taken through their stay, and the guesses are "
                   "pooled. When the guesses disagree or are unsure, the person is left as 'not sure'.",
        "do": do,
        "numbers": {"female_pct": k.get("female_pct"), "male_pct": k.get("male_pct"),
                    "gender_unknown_share": round(unknown_g, 3), "people": len(tracks),
                    "gender_test_balanced_accuracy": g_test, "age_test_balanced_accuracy": a_test},
        "confidence": confidence, "why": why,
        "stat": stat, "sub": sub[:1].upper() + sub[1:],
        "tip": "A rough guide to who visits; never use it to judge one person.",
        "brief": "Guessed from small pictures, often from behind or above.",
        "proof": {"kind": "people", "tracks": [t["id"] for t in clearest], "show": "labels",
                  "caption": f"The {len(clearest)} people the camera saw most clearly, with the system's guess"},
    }


def _trust(doc, setting, inflation):
    k = doc["kpis"]
    duration = doc["meta"].get("duration_sec") or 0.0
    points = []
    if inflation and inflation > 3 and setting["kind"] == "seated":
        points.append(f"Totals are too high here: {k['visitors']} tracks for at most {k['peak_occupancy']} people "
                      f"in view. Trust 'how busy' and seats, distrust totals and arrivals.")
    elif inflation and inflation > 6:
        points.append(f"{k['visitors']} people in all against {k['peak_occupancy']} at once: plausible for a busy "
                      f"shop, but some may be counted twice.")
    if duration < 1800:
        points.append(f"{_dur(duration).capitalize()} of footage is a snapshot; patterns need several days.")
    z = doc.get("zones") or {}
    if z:
        points.append("Someone hidden behind others can make a taken seat look free.")
    points.append("Nobody is identified: use this to understand the space, not to watch people.")
    return {
        "id": "trust", "title": "How far to trust this",
        "headline": "Some numbers here are solid, some are rough guides. Each section above says which.",
        "explain": "Solid: counting people visible at one moment. Rough: anything that follows one person "
                   "over time, or guesses who they are. Each section's note says why.",
        "do": points, "numbers": {"tracks_per_person_in_view": round(inflation, 2) if inflation else None},
        "confidence": None, "why": None, "proof": None,
    }


GLOSSARY = [
    ("People in view", "How many people the camera sees at the same moment."),
    ("Arrival", "Someone the system starts following during the recording."),
    ("Stay (dwell)", "How long one person stayed in view, from first to last sighting."),
    ("Passed through", "Stayed less than 3 seconds."),
    ("Stayed 30 s+", "Stayed at least 30 seconds: usually someone looking properly."),
    ("Seat or table taken", "Someone was inside the area drawn for it."),
    ("Time taken (utilisation)", "Share of the recording a seat or table was taken."),
    ("Turnover", "How many separate stays a seat or table had per hour."),
    ("≥ before a time", "The stay began before the recording did, so it lasted at least that long."),
    ("Not sure", "The system's guesses disagreed or were weak, so it gave no answer instead of a wrong one."),
    ("Confidence: solid / rough / unreliable", "How far a number can be trusted, and each section says why."),
]


def _headline(doc, setting):
    """One sentence to open on, built from the most trustworthy numbers: in a
    seated room, heads in view and seats taken, never the head count that
    split tracks inflate there (see _arrivals)."""
    k = doc["kpis"]
    avg = k.get("avg_occupancy") or 0
    in_view = (f"about {avg:.0f} {'person was' if round(avg) == 1 else 'people were'} in view at any moment"
               if avg >= 0.5 else "usually nobody was in view")
    if setting["kind"] == "seated":
        h = f"{in_view[0].upper()}{in_view[1:]}, {k.get('peak_occupancy', 0)} at the busiest"
        z = doc.get("zones") or {}
        s = z.get("summary") or {}
        if s.get("seats_tables"):
            places = [x for x in z.get("zones", []) if x["kind"] in ("seat", "table")]
            kind = ("seats" if all(x["kind"] == "seat" for x in places) else
                    "tables" if all(x["kind"] == "table" for x in places) else "places")
            h += f"; {s['occupied_at_end']} of {s['seats_tables']} {kind} were taken at the end"
        return h + "."
    return (f"{k['visitors']} {'person' if k['visitors'] == 1 else 'people'} came by in "
            f"{_dur(doc['meta'].get('duration_sec'))}; {in_view}.")


def shop_guide(doc):
    setting = setting_of(doc)
    inflation = _identity_inflation(doc)
    sections = [s for s in (
        _busy(doc, setting),
        _seats(doc, setting) if setting["kind"] == "seated" else None,
        _arrivals(doc, setting, inflation),
        _stopping(doc, setting, inflation),
        _where(doc),
        _seats(doc, setting) if setting["kind"] != "seated" else None,
        _queues(doc),
        _who(doc),
        _trust(doc, setting, inflation),
    ) if s]
    return {"schema": SCHEMA, "headline": _headline(doc, setting), "setting": setting, "sections": sections,
            "glossary": [{"term": a, "means": b} for a, b in GLOSSARY]}
