"""Tests for the PETA records and splits (src/research/peta.py) on a fake PETA tree."""
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.peta import (  # noqa: E402
    identity_split, image_split, parse_peta, read_split_csv, write_split_csv,
)


@pytest.fixture
def peta(tmp_path):
    root = tmp_path / "PETA dataset"
    for subset, n_people in (("VIPeR", 60), ("CUHK", 40)):
        d = root / subset / "archive"
        d.mkdir(parents=True)
        lines = []
        for pid in range(1, n_people + 1):
            gender = "personalMale" if pid % 2 else "personalFemale"
            age = ["personalLess30", "personalLess45", "personalLess60", "personalLarger60",
                   "personalLess15"][pid % 5]
            lines.append(f"{pid} {gender} {age} upperBodyBlack")
            for k in range(1 + pid % 4):                    # 1 to 4 images per person
                (d / f"{pid}_{k}.bmp").write_bytes(b"")
        lines.append("999 personalMale personalLess30")      # person without images
        (d / "Label.txt").write_text("\n".join(lines), encoding="utf-8")
    return root


def test_parse_reads_gender_age_and_images(peta):
    recs = parse_peta(peta)
    assert {r["subset"] for r in recs} == {"VIPeR", "CUHK"}
    one = [r for r in recs if r["subset"] == "VIPeR" and r["pid"] == "1"]
    assert len(one) == 2 and one[0]["gender"] == "male" and one[0]["age"] == "Age31-45"
    four = [r for r in recs if r["subset"] == "VIPeR" and r["pid"] == "4"]
    assert four[0]["gender"] == "female" and four[0]["age"] is None   # personalLess15 dropped
    assert not any(r["pid"] == "999" for r in recs)
    assert all(r["image"].endswith(".bmp") and "/archive/" in r["image"] for r in recs)


def test_identity_split_never_shares_a_person(peta):
    recs = parse_peta(peta)
    split = identity_split(recs, seed=1)
    owner = {}
    for r in recs:
        key = (r["subset"], r["pid"])
        assert owner.setdefault(key, split[r["image"]]) == split[r["image"]]
    counts = Counter(split.values())
    n = len(recs)
    # close to the PETA proportions 9500 / 1900 / 7600 (people are 1-4 images, so not exact)
    assert abs(counts["train"] / n - 0.5) < 0.06 and abs(counts["test"] / n - 0.4) < 0.06


def test_identity_split_keeps_gender_balance(peta):
    recs = parse_peta(peta)
    split = identity_split(recs, seed=1)
    for s in ("train", "test"):
        g = Counter(r["gender"] for r in recs if split[r["image"]] == s)
        assert 0.35 < g["male"] / (g["male"] + g["female"]) < 0.65


def test_image_split_has_exact_proportions_and_leaks_people(peta):
    recs = parse_peta(peta)
    split = image_split(recs, seed=1)
    people = {}
    for r in recs:
        people.setdefault((r["subset"], r["pid"]), set()).add(split[r["image"]])
    assert any(len(s) > 1 for s in people.values())     # the leak it exists to measure


def test_splits_are_deterministic(peta):
    recs = parse_peta(peta)
    assert identity_split(recs, 3) == identity_split(recs, 3)
    assert identity_split(recs, 3) != identity_split(recs, 4)


def test_split_csv_round_trip(peta, tmp_path):
    recs = parse_peta(peta)
    out = tmp_path / "split.csv"
    write_split_csv(recs, out, seed=2)
    test_rows = read_split_csv(out, "test")
    assert test_rows and all(r["split_identity"] == "test" and r["gender"] for r in test_rows)
    assert len(read_split_csv(out, "train", column="split_image")) > 0
