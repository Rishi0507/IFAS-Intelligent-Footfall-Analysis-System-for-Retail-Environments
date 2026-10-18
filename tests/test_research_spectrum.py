"""Tests for the Fourier tools (src/research/spectrum.py) and the RQ2 training
transforms built on them (src/training/transforms.py)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")
from PIL import Image  # noqa: E402

from src.research.degrade import apply  # noqa: E402
from src.research.spectrum import (  # noqa: E402
    amplitude, fda_swap, high_freq_ratio, low_freq_band, radial_power_spectrum,
)


@pytest.fixture
def src():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 120, (64, 48, 3))           # dark, textured


@pytest.fixture
def tgt():
    rng = np.random.default_rng(1)
    return rng.uniform(120, 255, (64, 48, 3))         # bright


# ---------------------------------------------------------------- FDA
def test_beta_zero_returns_the_source(src, tgt):
    assert np.array_equal(fda_swap(src, amplitude(tgt), 0.0), src)


def test_fda_keeps_the_phase_spectrum(src, tgt):
    out = fda_swap(src, amplitude(tgt), 0.1)
    f_src = np.fft.fft2(src, axes=(0, 1))
    f_out = np.fft.fft2(out, axes=(0, 1))
    keep = (np.abs(f_src) > 1e-6) & (np.abs(f_out) > 1e-6)
    # compare unit phasors so a 2*pi wrap does not count as a difference
    diff = np.abs(f_src[keep] / np.abs(f_src[keep]) - f_out[keep] / np.abs(f_out[keep]))
    assert diff.max() < 1e-6


def test_only_the_low_frequency_band_takes_the_target_amplitude(src, tgt):
    beta = 0.1
    out = fda_swap(src, amplitude(tgt), beta)
    a_out, a_src, a_tgt = amplitude(out), amplitude(src), amplitude(tgt)
    h, w = src.shape[:2]
    b = low_freq_band(h, w, beta)
    band = np.zeros((h, w), bool)
    band[h // 2 - b:h // 2 + b + 1, w // 2 - b:w // 2 + b + 1] = True
    assert np.allclose(a_out[band], a_tgt[band])
    assert np.allclose(a_out[~band], a_src[~band])


def test_fda_moves_the_mean_colour_to_the_target(src, tgt):
    out = fda_swap(src, amplitude(tgt), 0.05)
    # the zero-frequency term is in the band, and it carries the mean
    assert np.allclose(out.mean(axis=(0, 1)), tgt.mean(axis=(0, 1)))


def test_fda_with_itself_as_target_is_identity(src):
    assert np.allclose(fda_swap(src, amplitude(src), 0.2), src)


def test_shape_mismatch_is_rejected(src):
    with pytest.raises(ValueError):
        fda_swap(src, amplitude(src[:32]), 0.1)


# ---------------------------------------------------------------- spectra
@pytest.fixture
def person():
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 255, (160, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_radial_spectrum_covers_zero_to_nyquist(person):
    f, p = radial_power_spectrum(person, n_bins=16)
    assert len(f) == len(p) == 16
    assert 0 < f[0] < f[-1] < 0.5
    assert np.all(p >= 0)


@pytest.fixture
def natural():
    """1/f ("pink") noise, which has the power spectrum of natural images.
    White noise would mislead: JPEG blocking adds high frequencies to it."""
    rng = np.random.default_rng(3)
    h, w = 160, 64
    r = np.sqrt(np.fft.fftfreq(w)[None, :] ** 2 + np.fft.fftfreq(h)[:, None] ** 2)
    r[0, 0] = 1
    chans = []
    for _ in range(3):
        x = np.real(np.fft.ifft2((rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))) / r))
        chans.append((x - x.min()) / (x.max() - x.min()) * 255)
    return Image.fromarray(np.stack(chans, -1).astype(np.uint8))


# Below this the ratio is 8-bit rounding noise, so further blur cannot lower it.
QUANT_FLOOR = 1e-4


@pytest.mark.parametrize("kind,levels", [("gaussian", [1.0, 2.0, 3.0]),
                                         ("resolution", [48, 32, 24, 16]),
                                         ("jpeg", [50, 30, 10])])
def test_high_freq_ratio_falls_as_detail_is_lost(natural, kind, levels):
    r = [high_freq_ratio(natural)] + [high_freq_ratio(apply(natural, kind, lv)) for lv in levels]
    assert all(y < x or y < QUANT_FLOOR for x, y in zip(r, r[1:])), r
    assert r[1] < r[0]


# ---------------------------------------------------------------- transforms
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_random_degrade_is_reproducible_and_respects_p(person):
    from src.training.transforms import RandomDegrade
    assert RandomDegrade(p=0.0)(person) is person
    torch.manual_seed(5)
    a = np.asarray(RandomDegrade(p=1.0)(person))
    torch.manual_seed(5)
    b = np.asarray(RandomDegrade(p=1.0)(person))
    assert np.array_equal(a, b)


def test_fourier_style_outputs_vit_size_and_target_look(person):
    from src.training.transforms import FDA_SIZE, FourierStyle
    bright = Image.fromarray(np.full((90, 40, 3), 230, np.uint8))
    out = FourierStyle([bright], beta=0.05)(person)
    assert out.size == (FDA_SIZE, FDA_SIZE)
    assert np.asarray(out, dtype=float).mean() > np.asarray(person, dtype=float).mean() + 50


def test_build_transforms_rejects_bad_conditions():
    from src.training.transforms import build_transforms
    with pytest.raises(ValueError, match="unknown condition"):
        build_transforms(condition="mixup")
    with pytest.raises(ValueError, match="fda_targets"):
        build_transforms(condition="fda")
