"""Tests for the degradation suite (src/research/degrade.py)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
from PIL import Image  # noqa: E402

from src.research.degrade import SUITE, apply, iter_suite  # noqa: E402


@pytest.fixture
def person():
    """A textured 64x160 crop: stripes give the blur tests high frequencies to remove."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (160, 64, 3), dtype=np.uint8)
    arr[::4] = 255
    return Image.fromarray(arr)


def _sharpness(img):
    """Variance of a discrete Laplacian: falls as fine detail is lost."""
    g = np.asarray(img.convert("L"), dtype=float)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return lap.var()


@pytest.mark.parametrize("kind,level", list(iter_suite()))
def test_every_cell_is_deterministic_and_valid(person, kind, level):
    a = apply(person, kind, level, seed=7)
    b = apply(person, kind, level, seed=7)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert a.mode == "RGB"
    if kind == "truncation":
        assert a.size == (64, round(160 * (1 - level)))
    else:
        assert a.size == person.size


@pytest.mark.parametrize("kind", ["resolution", "gaussian", "motion"])
def test_blur_like_degradations_lose_detail_monotonically(person, kind):
    s = [_sharpness(apply(person, kind, lv)) for lv in SUITE[kind]]
    assert _sharpness(person) > s[0]
    assert all(x > y for x, y in zip(s, s[1:])), s


def test_jpeg_lower_quality_moves_further_from_the_original(person):
    ref = np.asarray(person, dtype=float)
    err = [np.abs(np.asarray(apply(person, "jpeg", q), dtype=float) - ref).mean() for q in SUITE["jpeg"]]
    assert all(x < y for x, y in zip(err, err[1:])), err


@pytest.mark.parametrize("frac", SUITE["occlusion"])
def test_occlusion_hides_the_requested_share_of_columns(person, frac):
    out = np.asarray(apply(person, "occlusion", frac, seed=3))
    flat = [np.all(out[:, c] == out[0, c]) for c in range(out.shape[1])]
    assert sum(flat) == round(64 * frac)
    # the band touches one edge
    assert flat[0] or flat[-1]


def test_seed_changes_the_random_degradations(person):
    sides = {tuple(np.asarray(apply(person, "occlusion", 0.25, seed=s))[0, 0]) for s in range(10)}
    assert len(sides) > 1


def test_unknown_kind_is_rejected(person):
    with pytest.raises(ValueError, match="unknown degradation"):
        apply(person, "fog", 1)
