import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ensemble import ensemble_gender


def test_clip_confident_face_agrees():
    assert ensemble_gender("male", 0.9, "male", 3.0) == "male"
    assert ensemble_gender("female", 0.9, "female", 3.0) == "female"


def test_clip_confident_face_disagrees_but_face_confident():
    assert ensemble_gender("female", 0.85, "male", 3.0) == "female"
    assert ensemble_gender("male", 0.85, "female", 3.0) == "male"


def test_clip_confident_face_low_conf_uses_clip():
    assert ensemble_gender("female", 0.50, "male", 3.0) == "male"


def test_uncertain_branch():
    assert ensemble_gender("male", 0.40, "male", 1.5) == "uncertain"
    assert ensemble_gender("female", 0.40, "female", 1.5) == "uncertain"


def test_clip_wishywashy_face_confident_uses_face():
    assert ensemble_gender("female", 0.90, "male", 1.5) == "female"
    assert ensemble_gender("male", 0.90, "female", 1.5) == "male"


def test_boundary_ratio():
    assert ensemble_gender("male", 0.50, "female", 1.8) == "female"
