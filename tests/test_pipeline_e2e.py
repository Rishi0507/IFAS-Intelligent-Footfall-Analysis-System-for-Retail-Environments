"""End-to-end pipeline test on a synthetic video with stubbed models.

Real weights are not available in CI, so the detector and both classifiers are
stubs. Everything else - decoding, tracking, voting, annotation, MP4 writing and
analytics - is the production code path.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from src.inference.pipeline import process_video  # noqa: E402

W, H, FPS, N_FRAMES = 320, 180, 10, 40


class _Box:
    def __init__(self, xyxy, conf):
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.conf = [conf]


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class StubDetector:
    """Two people walking in opposite directions, one entering halfway through."""

    def __init__(self):
        self.calls = 0

    def __call__(self, frame, **kwargs):
        i = self.calls
        self.calls += 1
        boxes = [_Box([10 + 4 * i, 40, 40 + 4 * i, 130], 0.9)]
        if i >= 10:
            boxes.append(_Box([290 - 4 * (i - 10), 50, 318 - 4 * (i - 10), 140], 0.8))
        return [_Result(boxes)]


class StubGender:
    def predict(self, crops):
        return [{"female": 0.2, "male": 0.8} for _ in crops]


class StubAge:
    def predict(self, crops):
        return [{"Age16-30": 0.55, "Age31-45": 0.25, "Age46-60": 0.15, "AgeAbove61": 0.05}
                for _ in crops]


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert writer.isOpened(), "no MP4 encoder available"
    for i in range(N_FRAMES):
        frame = np.full((H, W, 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, (10 + 4 * i, 40), (40 + 4 * i, 130), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return path


def test_process_video_writes_video_and_analytics(video, tmp_path):
    out = tmp_path / "run"
    doc = process_video(video, out, stride=1, write_video=True, bin_sec=1.0,
                        min_track_frames=3, device="cpu",
                        detector=StubDetector(), gender_model=StubGender(), age_model=StubAge())

    assert (out / "analytics.json").exists()
    assert (out / "video.mp4").exists() and (out / "video.mp4").stat().st_size > 0
    assert (out / "poster.jpg").exists()

    on_disk = json.loads((out / "analytics.json").read_text(encoding="utf-8"))
    assert on_disk["kpis"] == doc["kpis"]

    assert doc["meta"]["width"] == W and doc["meta"]["height"] == H
    assert doc["meta"]["frames_total"] == N_FRAMES
    assert doc["meta"]["has_video"] is True
    assert doc["kpis"]["visitors"] == 2
    assert doc["kpis"]["peak_occupancy"] == 2
    # Every crop got both classifiers' votes.
    assert all(t["gender"] == "male" and t["age"] == "Age16-30" for t in doc["tracks"])
    assert all(t["classified_frames"] > 0 for t in doc["tracks"])
    # One walks left-to-right, the other right-to-left.
    assert {t["entry"] for t in doc["tracks"]} == {"left", "right"}


def test_stride_samples_fewer_frames(video, tmp_path):
    doc = process_video(video, tmp_path / "run2", stride=4, write_video=False,
                        device="cpu", detector=StubDetector(),
                        gender_model=StubGender(), age_model=None)
    assert doc["meta"]["frames_sampled"] == N_FRAMES // 4
    assert doc["meta"]["has_video"] is False
    assert not (tmp_path / "run2" / "video.mp4").exists()
    # No age model: age stays unset, gender still resolves.
    assert all(t["age"] is None and t["gender"] == "male" for t in doc["tracks"])


def test_requires_at_least_one_classifier(video, tmp_path):
    # ValueError, not SystemExit: the server catches it and reports it as a
    # failed job instead of losing it in a background thread.
    with pytest.raises(ValueError, match="Nothing to classify"):
        process_video(video, tmp_path / "run3", detector=StubDetector())


def test_missing_video_is_reported_as_an_error(tmp_path):
    with pytest.raises(ValueError, match="Could not open video"):
        process_video(tmp_path / "nope.mp4", tmp_path / "run4",
                      detector=StubDetector(), gender_model=StubGender())

class FaintDetector:
    """One clear walker, plus two faint boxes that never get above 0.15:
    a person sitting behind glass (top right) and a poster (top left)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, frame, **kwargs):
        i = self.calls
        self.calls += 1
        return [_Result([_Box([10 + 4 * i, 40, 40 + 4 * i, 130], 0.9),
                         _Box([250, 20, 280, 90], 0.12),
                         _Box([100, 10, 130, 80], 0.14)])]


class StubVerifier:
    """The larger model: it sees the seated person and not the poster."""

    def __init__(self):
        self.calls = 0

    def __call__(self, frame, **kwargs):
        self.calls += 1
        return [_Result([_Box([248, 18, 282, 92], 0.5)])]


def test_faint_person_is_counted_only_when_the_second_detector_agrees(video, tmp_path):
    verifier = StubVerifier()
    doc = process_video(video, tmp_path / "run5", stride=1, write_video=False, device="cpu",
                        detector=FaintDetector(), gender_model=StubGender(), verifier=verifier)
    assert verifier.calls > 0
    assert doc["kpis"]["visitors"] == 2                       # the walker and the seated person
    seated = [t for t in doc["tracks"] if t["first_sec"] == 0.0 and t["zone"] == "Right"]
    assert len(seated) == 1 and seated[0]["dwell_sec"] > 3.0  # counted from the first faint sighting
    assert doc["kpis"]["hunches_dropped"] >= 1                # the poster
    # Asked a bounded number of times, not on every frame.
    assert verifier.calls <= 8


def test_without_a_second_detector_faint_boxes_count_for_nothing(video, tmp_path):
    doc = process_video(video, tmp_path / "run6", stride=1, write_video=False, device="cpu",
                        detector=FaintDetector(), gender_model=StubGender(), verifier_weights=None)
    assert doc["kpis"]["visitors"] == 1
