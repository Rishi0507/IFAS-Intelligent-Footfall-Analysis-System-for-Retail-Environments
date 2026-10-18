"""Cues that say how trustworthy one crop of a person is (RQ3).

A person is classified on up to 12 frames. Some of those crops are sharp and
whole; others are blurred by motion, cut off by the frame edge or half hidden
behind someone else. These cues describe each crop so that pooling can trust the
good ones more:

    hf_ratio     FFT high-frequency energy ratio (src/research/spectrum.py)
    lap_var      variance of the Laplacian, the usual sharpness check
    motion       Farneback flow inside the box minus the background median, in
                 pixels per frame: the person's own movement, so motion blur
    truncated    1 if the box touches the frame border
    occlusion    the box's largest IoU with any other box in the frame
    size         box area as a share of the frame
    det_conf     the detector's confidence in this box

Motion needs the previous video frame, which only the pipeline has, so the
pipeline measures it at export time (`frame_flow` + `flow_stats`) and stores it
in crops.jsonl. The rest can be recomputed from a saved crop.
"""
from __future__ import annotations

import numpy as np

from ..inference.tracker import iou
from .spectrum import hf_ratio, to_gray

CUE_NAMES = ["hf_ratio", "lap_var", "motion", "truncated", "occlusion", "size", "det_conf"]


def fft_sharpness(crop):
    return hf_ratio(crop)


def laplacian_variance(crop):
    import cv2
    g = to_gray(crop)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def frame_flow(prev_gray, gray, max_width=480):
    """Dense Farneback flow magnitude between two frames.

    Computed on frames scaled down to at most `max_width` pixels wide, which is
    plenty for a mean over a person's box and keeps it fast on a CPU. Returns
    (magnitude, scale); divide magnitudes by `scale` for original pixels.
    """
    import cv2
    h, w = gray.shape[:2]
    scale = min(1.0, max_width / float(w))
    if scale < 1.0:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        prev_gray = cv2.resize(prev_gray, size, interpolation=cv2.INTER_AREA)
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return np.hypot(flow[..., 0], flow[..., 1]), scale


def flow_stats(mag, scale, box, all_boxes=()):
    """Mean flow inside `box` and the median flow outside every box.

    The background term removes camera shake and global flicker, so `box_mag -
    bg_mag` is the person's own movement. Both are in original pixels per frame.
    """
    h, w = mag.shape
    x1, y1, x2, y2 = (int(round(v * scale)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x2, x1 + 1)), min(h, max(y2, y1 + 1))
    inside = mag[y1:y2, x1:x2]
    box_mag = float(inside.mean()) if inside.size else 0.0
    bg = np.ones_like(mag, dtype=bool)
    for b in list(all_boxes) + [box]:
        bx1, by1, bx2, by2 = (int(round(v * scale)) for v in b)
        bg[max(0, by1):max(0, by2), max(0, bx1):max(0, bx2)] = False
    bg_mag = float(np.median(mag[bg])) if bg.any() else 0.0
    return {"box_mag": round(box_mag / scale, 4), "bg_mag": round(bg_mag / scale, 4)}


def truncation(box, frame_w, frame_h, margin=2):
    x1, y1, x2, y2 = box
    return float(x1 <= margin or y1 <= margin or x2 >= frame_w - margin or y2 >= frame_h - margin)


def occlusion(box, other_boxes):
    return max((iou(box, o) for o in other_boxes), default=0.0)


def size_frac(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1) * (y2 - y1)) / float(max(1, frame_w * frame_h))


def crop_cues(crop, box, frame_w, frame_h, det_conf=None, flow=None, overlap=None):
    """Every cue for one crop. `flow` is the dict from `flow_stats`; leave it None
    when no previous frame was available, and `motion` is then None too."""
    motion = None
    if flow is not None:
        motion = max(0.0, float(flow["box_mag"]) - float(flow["bg_mag"]))
    return {
        "hf_ratio": round(fft_sharpness(crop), 5),
        "lap_var": round(laplacian_variance(crop), 6),
        "motion": None if motion is None else round(motion, 4),
        "truncated": truncation(box, frame_w, frame_h),
        "occlusion": round(float(overlap or 0.0), 4),
        "size": round(size_frac(box, frame_w, frame_h), 6),
        "det_conf": None if det_conf is None else round(float(det_conf), 4),
    }
