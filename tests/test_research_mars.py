"""Tests for the MARS index (src/research/mars.py) on a tiny fake MARS tree."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.mars import build_index, load_labels, parse_name  # noqa: E402

CSV_HEADER = ("person-id,camera,tracklet,Tops.c,Bottoms.c,Age,Tops.l,Bottoms.l,"
              "shoulder bag,backpack,hat,handbag,hair,gender,Bottoms.t\n")


def test_parse_name():
    assert parse_name("0001C1T0058F003.jpg") == (1, 1, 58, 3)
    assert parse_name("0065C1T0002F0016.jpg") == (65, 1, 2, 16)   # 4-digit frame
    assert parse_name("00-1C1T0001F001.jpg") is None       # distractor id
    assert parse_name("readme.txt") is None


@pytest.fixture
def mars(tmp_path):
    def touch(split, name):
        p = tmp_path / f"bbox_{split}" / name[:4] / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    # labelled train tracklet, frames written out of order
    for f in (3, 1, 2):
        touch("train", f"0001C2T0058F{f:03d}.jpg")
    # labelled test tracklet
    touch("test", "0007C1T0001F001.jpg")
    # junk id 0000 and an unlabelled tracklet: both must be dropped
    touch("test", "0000C1T0001F001.jpg")
    touch("train", "0001C3T0009F001.jpg")
    (tmp_path / "labels.csv").write_text(
        CSV_HEADER
        + "1,2,58,8,0,1,1,1,0,0,0,0,1,0,0\n"     # gender code 0, age teenager
        + "7,1,1,8,0,3,1,1,0,0,0,0,1,1,0\n",     # gender code 1, age old
        encoding="utf-8")
    return tmp_path


def test_labels_are_decoded(mars):
    lab = load_labels(mars / "labels.csv")
    assert lab[(1, 2, 58)]["gender"] == "male" and lab[(1, 2, 58)]["age"] == "teenager"
    assert lab[(7, 1, 1)]["gender"] == "female" and lab[(7, 1, 1)]["age"] == "old"


def test_index_joins_frames_and_labels(mars):
    index = build_index(mars, mars / "labels.csv")
    assert [(t["split"], t["pid"], t["cam"], t["tracklet"]) for t in index] == [
        ("test", 7, 1, 1), ("train", 1, 2, 58)]
    train = index[1]
    assert train["frames"] == [f"bbox_train/0001/0001C2T0058F{f:03d}.jpg" for f in (1, 2, 3)]
    assert train["gender"] == "male" and train["age"] == "teenager"


def test_min_frames_filters_short_tracklets(mars):
    index = build_index(mars, mars / "labels.csv", min_frames=2)
    assert [(t["pid"], t["tracklet"]) for t in index] == [(1, 58)]
