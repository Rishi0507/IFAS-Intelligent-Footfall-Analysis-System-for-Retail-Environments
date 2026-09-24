"""Tests for period aggregation (daily / weekly / monthly) and single-run insights.

The aggregation unit is the observed hour at one camera, so most of these are
about weighting: a long recording must count for more than a short one, two
clips from one afternoon must merge into one day, and cameras must not be
pooled blindly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.trends import (  # noqa: E402
    build_trends, hourly_buckets, insights, summarise_run,
)


def _doc(visitors=60, duration=600.0, captured="2026-09-05T14:00", demo=False,
         female=60.0, dwell=12.0, ages=True, site=None, camera=None, spread=True):
    """A minimal analytics doc. `spread` lays arrivals across the whole clip."""
    tracks = []
    for i in range(visitors):
        first = (i / max(visitors, 1)) * duration if spread else 0.0
        tracks.append({
            "id": i, "gender": "female" if i < visitors * female / 100 else "male",
            "gender_conf": 0.9, "age": "Age31-45" if ages else None, "age_conf": 0.6,
            "first_sec": first, "last_sec": first + dwell,
            "dwell_sec": dwell + (20 if i % 10 == 0 else 0), "tier": "Browsing",
            "zone": "Center", "flow": "entered", "frames": 40,
        })
    meta = {"source": "clip.mp4", "duration_sec": duration, "captured_at": captured,
            "demo": demo}
    if site:
        meta["site"] = site
    if camera:
        meta["camera"] = camera
    n_bins = max(1, int(duration / 60))
    return {
        "meta": meta,
        "generated_at": "2026-09-07T10:00:00",
        "kpis": {"visitors": visitors, "avg_dwell_sec": dwell, "median_dwell_sec": dwell,
                 "female_pct": female, "male_pct": round(100 - female, 1),
                 "dominant_age": "Age31-45" if ages else None, "dominant_age_pct": 100.0},
        "timeline": {"bin_sec": 60.0, "bins": [
            {"t": i * 60.0, "entries": 10 if i == 1 else 2} for i in range(n_bins)]},
        "distributions": {
            "zones": {"Left": {"visitors": 10, "avg_dwell": 8.0},
                      "Center": {"visitors": max(visitors - 15, 0), "avg_dwell": 14.0},
                      "Right": {"visitors": 5, "avg_dwell": 9.0}},
            "dwell_by_segment": [
                {"gender": "female", "age": "Age31-45", "n": 30, "avg": 18.0, "median": 15.0, "max": 40.0},
                {"gender": "male", "age": "Age31-45", "n": 20, "avg": 9.0, "median": 8.0, "max": 20.0},
            ],
        },
        "tracks": tracks,
    }


# ------------------------------------------------------------------ buckets
def test_a_clip_inside_one_hour_makes_one_bucket():
    b = hourly_buckets("r", _doc(visitors=12, duration=600.0, captured="2026-09-05T14:05"))
    assert len(b) == 1
    assert b[0]["hour"] == 14 and b[0]["weekday"] == "Sat"
    assert b[0]["observed_min"] == 10.0
    assert b[0]["arrivals"] == 12


def test_a_clip_spanning_two_hours_splits_its_minutes():
    # 16:50 + 20 minutes -> 10 minutes in hour 16, 10 in hour 17
    b = sorted(hourly_buckets("r", _doc(visitors=20, duration=1200.0,
                                        captured="2026-09-05T16:50")),
               key=lambda x: x["hour"])
    assert [x["hour"] for x in b] == [16, 17]
    assert [x["observed_min"] for x in b] == [10.0, 10.0]
    assert sum(x["arrivals"] for x in b) == 20
    assert b[0]["arrivals"] > 0 and b[1]["arrivals"] > 0


def test_buckets_carry_gender_and_age():
    b = hourly_buckets("r", _doc(visitors=10, female=60.0, duration=600.0))[0]
    assert b["gender"]["female"] == 6 and b["gender"]["male"] == 4
    assert b["age"]["Age31-45"] == 10


def test_a_run_without_a_capture_time_makes_no_buckets():
    doc = _doc()
    doc["meta"]["captured_at"] = None
    doc["generated_at"] = None
    assert hourly_buckets("r", doc) == []


# -------------------------------------------------------------- weighting
def test_a_long_recording_outweighs_a_short_one():
    """The bug this rewrite fixes: both clips used to count equally."""
    runs = [
        # 3 hours at 10/hour, then 3 minutes at 200/hour, same weekday
        ("long", _doc(visitors=30, duration=10800.0, captured="2026-09-05T09:00")),
        ("short", _doc(visitors=10, duration=180.0, captured="2026-09-05T13:00")),
    ]
    sat = next(b for b in build_trends(runs)["by_weekday"] if b["key"] == "Sat")
    assert sat["arrivals"] == 40
    assert sat["observed_hours"] == 3.05
    assert sat["per_hour"] == 13.1          # not the ~105 an unweighted mean gave


def test_two_clips_from_one_day_merge_into_one_day():
    runs = [
        ("am", _doc(visitors=30, duration=3600.0, captured="2026-09-05T10:00")),
        ("pm", _doc(visitors=90, duration=3600.0, captured="2026-09-05T17:00")),
    ]
    t = build_trends(runs)
    assert t["coverage"]["days"] == 1
    assert len(t["by_day"]) == 1
    assert t["by_day"][0]["arrivals"] == 120
    assert t["by_day"][0]["per_hour"] == 60.0      # 120 arrivals over 2 observed hours


def test_hour_of_day_uses_the_hour_watched_not_the_clip_start():
    runs = [("r", _doc(visitors=60, duration=7200.0, captured="2026-09-05T15:00"))]
    hours = {b["key"] for b in build_trends(runs)["by_hour"]}
    assert hours == {15, 16}


def test_short_clips_contribute_without_dominating():
    runs = [
        ("long", _doc(visitors=60, duration=3600.0, captured="2026-09-01T10:00")),
        ("blip", _doc(visitors=2, duration=10.0, captured="2026-09-01T12:00")),
    ]
    day = build_trends(runs)["by_day"][0]
    assert day["arrivals"] == 62
    assert 60.0 < day["per_hour"] < 63.0     # the 720/h blip barely moves it


def test_a_day_watched_for_seconds_is_not_a_data_point():
    runs = [
        ("d1", _doc(visitors=60, duration=3600.0, captured="2026-09-01T10:00")),
        ("d2", _doc(visitors=60, duration=3600.0, captured="2026-09-02T10:00")),
        ("blip", _doc(visitors=2, duration=10.0, captured="2026-09-03T10:00")),
    ]
    t = build_trends(runs)
    assert t["coverage"]["days"] == 3
    assert t["coverage"]["usable_days"] == 2      # the 10-second day is dropped
    assert t["trend"]["latest_per_hour"] == 60.0


# ---------------------------------------------------------- daily/monthly
def test_days_months_and_weekdays_are_all_available():
    runs = [
        ("a", _doc(visitors=30, duration=3600.0, captured="2026-09-05T10:00")),
        ("b", _doc(visitors=60, duration=3600.0, captured="2026-10-06T10:00")),
    ]
    t = build_trends(runs)
    assert [d["key"] for d in t["by_day"]] == ["2026-09-05", "2026-10-06"]
    assert [m["key"] for m in t["by_month"]] == [9, 10]
    assert {w["key"] for w in t["by_weekday"]} == {"Sat", "Tue"}
    assert t["by_month"][1]["per_hour"] == 60.0


def test_every_period_row_carries_the_retail_metrics():
    runs = [("r", _doc(visitors=100, female=70.0, duration=3600.0,
                       captured="2026-09-05T10:00"))]
    row = build_trends(runs)["by_weekday"][0]
    for field in ("arrivals", "observed_hours", "per_hour", "days", "gender",
                  "female_pct", "age", "avg_dwell_sec", "engaged_pct"):
        assert field in row
    assert row["gender"] == {"female": 70, "male": 30}
    assert row["female_pct"] == 70.0
    assert row["age"]["Age31-45"] == 100


# ------------------------------------------------------------ site/camera
def test_period_rows_carry_dwell_and_engagement():
    # every 10th visitor gets +20s dwell, so 10 of 100 clear the 30s bar
    row = build_trends([("r", _doc(visitors=100, dwell=12.0, duration=3600.0,
                                   captured="2026-09-05T10:00"))])["by_weekday"][0]
    assert row["engaged"] == 10
    assert row["engaged_pct"] == 10.0
    assert 13.0 < row["avg_dwell_sec"] < 15.0


def test_sources_report_real_versus_sample_counts():
    runs = [("real", _doc()), ("s1", _doc(demo=True)), ("s2", _doc(demo=True))]
    src = build_trends(runs)["sources"]
    assert src == {"real_runs": 1, "sample_runs": 2, "samples_included": False}
    assert build_trends(runs, include_demo=True)["sources"]["samples_included"] is True


def test_cameras_are_listed_and_flagged_when_mixed():
    runs = [
        ("door", _doc(camera="Entrance", captured="2026-09-01T10:00", duration=3600.0)),
        ("till", _doc(camera="Till", captured="2026-09-01T10:00", duration=3600.0)),
    ]
    t = build_trends(runs)
    assert t["cameras"] == ["Entrance", "Till"]
    assert t["coverage"]["mixed_cameras"] is True


def test_scoping_to_one_camera_excludes_the_others():
    runs = [
        ("door", _doc(visitors=90, camera="Entrance", duration=3600.0,
                      captured="2026-09-01T10:00")),
        ("till", _doc(visitors=10, camera="Till", duration=3600.0,
                      captured="2026-09-01T10:00")),
    ]
    t = build_trends(runs, camera="Entrance")
    assert t["coverage"]["runs"] == 1
    assert t["by_day"][0]["arrivals"] == 90
    assert t["coverage"]["mixed_cameras"] is False


def test_runs_default_to_one_site_and_camera():
    row = summarise_run("r", _doc())
    assert row["site"] == "Store" and row["camera"] == "Camera 1"


# ---------------------------------------------------------------- gating
def test_demo_runs_are_excluded_by_default():
    runs = [("real", _doc()), ("demo", _doc(demo=True))]
    assert build_trends(runs)["coverage"]["runs"] == 1
    assert build_trends(runs, include_demo=True)["coverage"]["runs"] == 2


def test_sample_runs_never_leak_into_the_buckets():
    runs = [("real", _doc(visitors=10, duration=3600.0, captured="2026-09-01T10:00")),
            ("demo", _doc(visitors=999, duration=3600.0, captured="2026-09-01T10:00", demo=True))]
    assert build_trends(runs)["by_day"][0]["arrivals"] == 10


def test_one_day_is_not_a_trend():
    t = build_trends([("a", _doc(duration=3600.0))])
    assert t["coverage"]["enough_for_trend"] is False
    assert t["trend"] is None


def test_three_days_are():
    runs = [("d%d" % i, _doc(visitors=30, duration=3600.0,
                             captured=f"2026-09-0{i}T10:00")) for i in (1, 2, 3)]
    assert build_trends(runs)["coverage"]["enough_for_trend"] is True


def test_monthly_comparison_needs_two_months():
    same = [("a", _doc(duration=3600.0, captured="2026-09-05T10:00")),
            ("b", _doc(duration=3600.0, captured="2026-09-12T10:00"))]
    assert build_trends(same)["coverage"]["enough_for_season"] is False
    spread = same + [("c", _doc(duration=3600.0, captured="2026-12-07T10:00"))]
    assert build_trends(spread)["coverage"]["enough_for_season"] is True


def test_trend_compares_latest_day_against_earlier_days():
    runs = [
        ("a", _doc(visitors=30, duration=3600.0, captured="2026-09-01T10:00")),
        ("b", _doc(visitors=30, duration=3600.0, captured="2026-09-02T10:00")),
        ("c", _doc(visitors=60, duration=3600.0, captured="2026-09-03T10:00")),
    ]
    t = build_trends(runs)["trend"]
    assert t["latest_per_hour"] == 60.0
    assert t["previous_avg_per_hour"] == 30.0
    assert t["change_pct"] == 100.0


def test_no_runs_is_not_an_error():
    t = build_trends([])
    assert t["coverage"]["runs"] == 0 and t["by_weekday"] == [] and t["trend"] is None


# -------------------------------------------------------------- insights
def test_short_clips_are_not_extrapolated_to_an_hourly_rate():
    text = next(i["text"] for i in insights(_doc(visitors=2, duration=10.0)) if i["kind"] == "traffic")
    assert "if this pace held" not in text     # no extrapolated rate
    assert "too short" in text


def test_long_enough_clips_are_extrapolated():
    text = next(i["text"] for i in insights(_doc(visitors=60, duration=600.0)) if i["kind"] == "traffic")
    assert "360 an hour" in text


def test_insights_read_as_sentences():
    out = insights(_doc())
    kinds = {i["kind"] for i in out}
    assert {"traffic", "staffing", "engagement", "audience", "layout"} <= kinds
    assert all(i["text"].endswith(".") for i in out)


def test_insights_skip_age_when_no_age_model_ran():
    text = " ".join(i["text"] for i in insights(_doc(ages=False)))
    assert "age group" not in text


def test_insights_handle_an_empty_run():
    doc = _doc(visitors=0)
    doc["tracks"] = []
    out = insights(doc)
    assert len(out) == 1 and out[0]["kind"] == "none"


def test_quiet_footage_does_not_divide_by_zero():
    doc = _doc()
    doc["timeline"]["bins"] = [{"t": 0.0, "entries": 5}, {"t": 60.0, "entries": 0}]
    text = next(i["text"] for i in insights(doc) if i["kind"] == "staffing")
    assert "quietest stretches had none" in text
