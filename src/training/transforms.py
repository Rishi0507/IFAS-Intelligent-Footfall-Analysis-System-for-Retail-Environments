"""Image transforms for training / validation (pure ViT, no CLIP).

`build_transforms(condition=...)` selects the training condition compared in
RQ2 of the research plan:

    baseline       flips and mild colour jitter (the original stage-1 recipe)
    degrade        + a random CCTV degradation (src/research/degrade.py)
    fda            + Fourier Domain Adaptation towards unlabelled target frames
    fda+degrade    both

Randomness comes from torch, so DataLoader worker seeding (and torch.manual_seed)
makes a run reproducible. The validation transform never changes.
"""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor

_PROC = None
CONDITIONS = ("baseline", "degrade", "fda", "fda+degrade")
FDA_SIZE = 224          # FDA runs at the ViT input size, which is what the network sees


def _get_proc(model_name="google/vit-base-patch16-224"):
    global _PROC
    if _PROC is None:
        _PROC = ViTImageProcessor.from_pretrained(model_name)
    return _PROC


class RandomDegrade:
    """With probability `p`, apply one CCTV degradation drawn uniformly from the
    suite (any kind, any severity)."""

    def __init__(self, p=0.5, kinds=None):
        from src.research.degrade import SUITE
        self.p = p
        self.cells = [(k, lv) for k, lvs in SUITE.items() if kinds is None or k in kinds for lv in lvs]

    def __call__(self, pil_img):
        from src.research.degrade import apply
        if torch.rand(1).item() >= self.p:
            return pil_img
        kind, level = self.cells[torch.randint(len(self.cells), (1,)).item()]
        return apply(pil_img, kind, level, seed=torch.randint(2 ** 31 - 1, (1,)).item())


class FourierStyle:
    """Fourier Domain Adaptation towards a pool of unlabelled target images.

    Each call resizes the image to FDA_SIZE x FDA_SIZE and swaps its
    low-frequency amplitude with that of a random target. Target amplitudes
    are computed once, here. `beta` sets the size of the swapped band;
    Yang & Soatto use 0.01 to 0.09.
    """

    def __init__(self, targets, beta=0.01, p=1.0):
        from src.research.spectrum import amplitude
        if not targets:
            raise ValueError("FDA needs at least one target image")
        self.beta, self.p = beta, p
        self.amps = []
        for t in targets:
            img = t if isinstance(t, Image.Image) else Image.open(t)
            arr = np.asarray(img.convert("RGB").resize((FDA_SIZE, FDA_SIZE), Image.BICUBIC), dtype=np.float64)
            self.amps.append(amplitude(arr))

    def __call__(self, pil_img):
        from src.research.spectrum import fda_swap
        if torch.rand(1).item() >= self.p:
            return pil_img
        src = np.asarray(pil_img.convert("RGB").resize((FDA_SIZE, FDA_SIZE), Image.BICUBIC), dtype=np.float64)
        amp = self.amps[torch.randint(len(self.amps), (1,)).item()]
        out = fda_swap(src, amp, self.beta)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def target_frames(folder, max_frames=500):
    """Paths of the unlabelled target frames in `folder` (e.g. the output of
    scripts/research/fda_frames.py), at most `max_frames`, spread evenly over
    the sorted list so every clip contributes, not only the first few by name."""
    from pathlib import Path
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(paths) > max_frames:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, max_frames).round().astype(int)]
    return paths


def build_transforms(image_size=224, augment=True,
                     model_name="google/vit-base-patch16-224",
                     condition="baseline", fda_targets=None, fda_beta=0.01, degrade_p=0.5):
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")
    if "fda" in condition and not fda_targets:
        raise ValueError(f"condition {condition!r} needs fda_targets (unlabelled target frames)")
    proc = _get_proc(model_name)
    degrade = RandomDegrade(degrade_p) if "degrade" in condition else None
    fda = FourierStyle(fda_targets, fda_beta) if "fda" in condition else None
    if augment:
        def train_transform(pil_img):
            if degrade is not None:
                pil_img = degrade(pil_img)
            if torch.rand(1).item() < 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            from PIL import ImageEnhance
            pil_img = ImageEnhance.Brightness(pil_img).enhance(0.9 + 0.1*torch.rand(1).item())
            pil_img = ImageEnhance.Contrast(pil_img).enhance(0.9 + 0.1*torch.rand(1).item())
            pil_img = ImageEnhance.Color(pil_img).enhance(0.9 + 0.1*torch.rand(1).item())
            if fda is not None:
                pil_img = fda(pil_img)
            return _to_tensor(proc, pil_img)
    else:
        def train_transform(pil_img):
            return _to_tensor(proc, pil_img)
    def val_transform(pil_img):
        return _to_tensor(proc, pil_img)
    return train_transform, val_transform

def _to_tensor(proc, pil_img):
    out = proc(images=pil_img, return_tensors="pt")
    return out["pixel_values"][0]
