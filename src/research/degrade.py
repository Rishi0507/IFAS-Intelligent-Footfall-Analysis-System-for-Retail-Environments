"""Controlled degradations that imitate what CCTV does to a pedestrian crop.

RQ1 asks *why* accuracy drops from PETA to store footage. Real footage mixes
every cause at once, so the PETA test split is degraded one cause at a time,
at increasing severity, and accuracy is read off per cause:

    resolution   the person is far away: downscale, then scale back up
    gaussian     out of focus
    motion       the person moves during the exposure
    jpeg         stream compression
    truncation   the frame edge cuts off the lower body
    occlusion    a shelf, a display or another person hides part of the body

Every function takes and returns a PIL RGB image. `apply` is deterministic for a
given seed, so a degraded test set can be rebuilt exactly. Truncation returns a
shorter image, as a truncated detector box would; everything else keeps the
input size, so the classifier sees the same input geometry as on clean data.
"""
from __future__ import annotations
import io

import numpy as np
from PIL import Image, ImageFilter

# Severity levels used in the paper, mildest first.
SUITE = {
    "resolution": [48, 32, 24, 16],         # target width in pixels (PETA crops are ~ 60-100)
    "gaussian": [1.0, 2.0, 3.0],            # blur sigma in pixels
    "motion": [5, 9, 15],                   # blur length in pixels
    "jpeg": [50, 30, 10],                   # JPEG quality
    "truncation": [0.25, 0.5],              # share of the height cut from the bottom
    "occlusion": [0.25, 0.5],               # share of the width hidden
}


def resolution(img, width):
    """Downscale to `width` pixels wide (area averaging), then back up (bicubic)."""
    w, h = img.size
    if width >= w:
        return img.copy()
    small = img.resize((width, max(1, round(h * width / w))), Image.BOX)
    return small.resize((w, h), Image.BICUBIC)


def gaussian(img, sigma):
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def motion(img, length, angle_deg=0.0):
    """Linear motion blur: average along a line of `length` pixels at `angle_deg`
    (0 = horizontal, the usual direction of a person walking past a camera)."""
    import cv2

    length = int(length)
    if length <= 1:
        return img.copy()
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= k.sum()
    arr = cv2.filter2D(np.asarray(img), -1, k, borderType=cv2.BORDER_REFLECT)
    return Image.fromarray(arr)


def jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def truncation(img, frac):
    """Keep the top (1 - frac) of the image, as when the frame edge cuts the legs."""
    w, h = img.size
    keep = max(1, round(h * (1 - frac)))
    return img.crop((0, 0, w, keep))


def occlusion(img, frac, rng):
    """Hide a full-height band covering `frac` of the width, from the left or the
    right edge (people are mostly hidden from the side by shelves or others).
    The band is filled with a flat colour drawn from the image itself, so it
    does not add an out-of-distribution colour the classifier could key on."""
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    band = max(1, round(w * frac))
    colour = arr[rng.integers(0, h), rng.integers(0, w)]
    if rng.random() < 0.5:
        arr[:, :band] = colour
    else:
        arr[:, w - band:] = colour
    return Image.fromarray(arr)


_FUNCS = {"resolution": resolution, "gaussian": gaussian, "motion": motion,
          "jpeg": jpeg, "truncation": truncation, "occlusion": occlusion}


def apply(img, kind, level, seed=0):
    """Apply one degradation. `seed` only matters for the random ones
    (occlusion side and colour, motion direction); fix it per image, e.g. the
    image's index, so the degraded set is reproducible."""
    if kind not in _FUNCS:
        raise ValueError(f"unknown degradation {kind!r}; choose from {sorted(_FUNCS)}")
    img = img.convert("RGB")
    rng = np.random.default_rng(seed)
    if kind == "occlusion":
        return occlusion(img, level, rng)
    if kind == "motion":
        # Mostly horizontal, as people walk across the view; a little vertical spread.
        return motion(img, level, angle_deg=float(rng.uniform(-20, 20)))
    return _FUNCS[kind](img, level)


def iter_suite(kinds=None):
    """Yield (kind, level) for every cell of the suite, mildest first per kind."""
    for kind, levels in SUITE.items():
        if kinds is None or kind in kinds:
            for level in levels:
                yield kind, level
