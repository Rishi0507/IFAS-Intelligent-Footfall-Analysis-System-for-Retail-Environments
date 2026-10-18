"""Fourier-domain tools: describing the domain gap (RQ1) and closing it (RQ2).

* `radial_power_spectrum` / `high_freq_ratio` measure how much fine detail a
  crop carries. CCTV crops lose high frequencies to distance, blur and
  compression, and this is the number that shows it.
* `fda_swap` is Fourier Domain Adaptation (Yang & Soatto, CVPR 2020): the
  low-frequency amplitude of a source image is replaced by that of a target
  image while the phase is kept. Amplitude at low frequencies carries global
  colour, contrast and illumination; phase carries the structure. A PETA
  training image therefore takes on the look of the store footage while
  staying the same person, with no target labels needed.
"""
from __future__ import annotations
import math

import numpy as np


def _gray(img):
    a = np.asarray(img, dtype=np.float64)
    return a if a.ndim == 2 else a[..., :3] @ np.array([0.299, 0.587, 0.114])


def radial_power_spectrum(img, n_bins=32):
    """Radially averaged power spectrum of a greyscale version of `img`.

    Returns (frequencies, power): `n_bins` rings from 0 to the Nyquist radius,
    with frequency in cycles per pixel (0 to 0.5).
    """
    g = _gray(img)
    g = g - g.mean()
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w))           # reduce edge leakage
    p = np.abs(np.fft.fftshift(np.fft.fft2(g * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    edges = np.linspace(0, 0.5, n_bins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
    inside = r <= 0.5
    power = np.bincount(idx[inside], weights=p[inside], minlength=n_bins)
    count = np.bincount(idx[inside], minlength=n_bins)
    power = np.divide(power, count, out=np.zeros(n_bins), where=count > 0)
    return (edges[:-1] + edges[1:]) / 2, power


def high_freq_ratio(img, cutoff=0.25):
    """Share of spectral power above `cutoff` cycles per pixel.

    A scale-free sharpness measure: it falls as a crop is blurred, downscaled
    or compressed, whatever its size.
    """
    f, p = radial_power_spectrum(img)
    total = p.sum()
    return float(p[f >= cutoff].sum() / total) if total > 0 else 0.0


def low_freq_band(h, w, beta):
    """Half-width of the centred low-frequency square FDA swaps: floor(min(h, w) * beta)."""
    return int(math.floor(min(h, w) * beta))


def amplitude(arr):
    """Centred amplitude spectrum per channel of an HxWxC array."""
    return np.abs(np.fft.fftshift(np.fft.fft2(arr, axes=(0, 1)), axes=(0, 1)))


def fda_swap(src, tgt_amp, beta):
    """Replace the low-frequency amplitude of `src` with `tgt_amp`, keep the phase.

    src: HxWxC float array. tgt_amp: the target's centred amplitude spectrum of
    the same shape (see `amplitude`), precomputed once per target image.
    Returns an unclipped float array; clip to [0, 255] before use as an image.
    beta = 0 returns `src` unchanged.
    """
    src = np.asarray(src, dtype=np.float64)
    if tgt_amp.shape != src.shape:
        raise ValueError(f"target amplitude {tgt_amp.shape} != source {src.shape}")
    h, w = src.shape[:2]
    b = low_freq_band(h, w, beta)
    if b == 0:
        return src.copy()
    f = np.fft.fftshift(np.fft.fft2(src, axes=(0, 1)), axes=(0, 1))
    amp, pha = np.abs(f), np.angle(f)
    cy, cx = h // 2, w // 2
    amp[cy - b:cy + b + 1, cx - b:cx + b + 1] = tgt_amp[cy - b:cy + b + 1, cx - b:cx + b + 1]
    out = np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha), axes=(0, 1)), axes=(0, 1))
    return np.real(out)


# ------------------------------------------------- measured as the model sees it
# The quality cues (RQ3) and eval_frames.py (RQ1) compare crops of very different
# native sizes. They measure each crop resized to one fixed square, the way the
# classifier sees it, so a tiny crop that was blown up scores as having little
# detail, which is what it has. `high_freq_ratio` above works at native size and
# takes the mean power per ring; `hf_ratio` takes the share of total energy.
SIZE = 128          # side of the square these are measured on
HF_CUTOFF = 0.25    # cycles/pixel; above this counts as high frequency
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def to_gray(img, size=SIZE):
    """PIL image or array -> float32 grey square in [0, 1], resized to `size`.

    Grey conversion and resizing are done in floating point. Rounding the
    resized image to 8 bits would add white noise at every frequency, and on a
    blurred crop that noise outweighs what detail is left.
    """
    from PIL import Image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    g = (np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0) @ LUMA
    return np.asarray(Image.fromarray(g).resize((size, size), Image.BILINEAR), dtype=np.float32)


def _power(gray):
    g = gray - gray.mean()
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w))        # cuts the leakage from the crop's edges
    return np.abs(np.fft.fftshift(np.fft.fft2(g * win))) ** 2


def _radius(shape):
    h, w = shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    return np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def hf_ratio(img_or_gray, cutoff=HF_CUTOFF, size=SIZE):
    """Share of the crop's spectral energy at or above `cutoff` cycles/pixel,
    measured on the crop resized to `size`.

    0 for a flat crop. Falls as the crop is blurred or downscaled. Once almost
    no detail is left (a few thousandths) it levels off at the floor set by the
    crop's own 8-bit encoding, so tiny differences there mean nothing.
    """
    gray = img_or_gray if (isinstance(img_or_gray, np.ndarray) and img_or_gray.ndim == 2
                           and img_or_gray.dtype.kind == "f") else to_gray(img_or_gray, size)
    if np.ptp(gray) < 1e-6:                             # flat: only rounding residue is left
        return 0.0
    p = _power(gray)
    r = _radius(p.shape)
    total = p[r > 0].sum()
    if total <= 0:
        return 0.0
    return float(p[(r >= cutoff) & (r <= 0.5)].sum() / total)


def mean_log_spectrum(images, n_bins=32, size=SIZE):
    """Average of log10 radial power over many crops resized to `size`: one
    domain's curve in the RQ1 figure."""
    curves = []
    freqs = None
    for img in images:
        freqs, power = radial_power_spectrum(to_gray(img, size), n_bins)
        curves.append(np.log10(power + 1e-12))
    if not curves:
        return None, None
    return freqs, np.mean(curves, axis=0)
