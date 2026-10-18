"""Tests for the script that records what the gender loader fix changed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))

from summarize_gender_runs import changed, per_person, summarize  # noqa: E402


def _doc(labels):
    return {
        "config": {"gender_model": "repo/x"},
        "meta": {"source": "clip.mp4"},
        "tracks": [{"id": i, "gender": g, "gender_conf": c, "gender_votes": {g: 1}}
                   for i, (g, c) in enumerate(labels, start=1)],
    }


def test_summarize_counts_labels_and_low_confidence():
    s = summarize(_doc([("female", 0.9), ("male", 0.6), ("male", 0.8)]))
    assert (s["people"], s["female"], s["male"], s["low_conf"]) == (3, 1, 2, 1)
    assert s["mean_conf"] == round((0.9 + 0.6 + 0.8) / 3, 3)
    assert s["gender_model"] == "repo/x"


def test_people_line_up_across_runs_and_changes_are_counted():
    before = _doc([("male", 0.66), ("male", 0.9)])
    after = _doc([("female", 1.0), ("male", 1.0)])
    people = per_person([("before", before), ("after", after)])
    assert people["1"]["before"][0] == "male" and people["1"]["after"][0] == "female"
    assert changed(people, "before", "after") == 1


def test_missing_confidence_is_skipped_not_counted_as_low():
    s = summarize(_doc([("female", None), ("male", 0.95)]))
    assert s["low_conf"] == 0 and s["mean_conf"] == 0.95
