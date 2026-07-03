import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from src.geometry import iou, contains, dedup_boxes, validate_face_landmarks, centroid


def test_iou_overlap():
    a = np.array([0, 0, 100, 100], dtype=np.float32)
    b = np.array([50, 0, 150, 100], dtype=np.float32)
    assert iou(a, b) == pytest.approx(1 / 3, rel=0.05)


def test_iou_disjoint():
    a = np.array([0, 0, 50, 50], dtype=np.float32)
    b = np.array([200, 200, 250, 250], dtype=np.float32)
    assert iou(a, b) == 0.0


def test_contains_full():
    outer = np.array([0, 0, 200, 200], dtype=np.float32)
    inner = np.array([50, 50, 100, 100], dtype=np.float32)
    assert contains(outer, inner, 0.85)


def test_dedup_iou_suppresses():
    a = np.array([[0, 0, 100, 100], [50, 0, 150, 100]], dtype=np.float32)
    s = np.array([0.9, 0.5], dtype=np.float32)
    b, _ = dedup_boxes(a, s, iou_thresh=0.30)
    assert len(b) == 1


def test_dedup_containment_suppresses():
    big = np.array([0, 0, 200, 200], dtype=np.float32)
    small = np.array([80, 80, 110, 110], dtype=np.float32)
    b, _ = dedup_boxes(np.stack([big, small]), np.array([0.7, 0.95], dtype=np.float32))
    assert len(b) == 1


def test_landmarks_valid():
    lm = np.array([[40, 40], [60, 40], [50, 55], [42, 65], [58, 65]], dtype=np.float32)
    bbox = np.array([20, 20, 80, 100], dtype=np.float32)
    assert validate_face_landmarks(lm, bbox)


def test_landmarks_eyes_below_nose_rejected():
    lm = np.array([[40, 60], [60, 60], [50, 55], [42, 65], [58, 65]], dtype=np.float32)
    bbox = np.array([20, 20, 80, 100], dtype=np.float32)
    assert not validate_face_landmarks(lm, bbox)
