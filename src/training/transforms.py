"""Image transforms for training / validation (pure ViT, no CLIP)."""
from __future__ import annotations
import torch
from PIL import Image
from transformers import ViTImageProcessor

_PROC = None

def _get_proc(model_name="google/vit-base-patch16-224"):
    global _PROC
    if _PROC is None:
        _PROC = ViTImageProcessor.from_pretrained(model_name)
    return _PROC

def build_transforms(image_size=224, augment=True,
                     model_name="google/vit-base-patch16-224"):
    proc = _get_proc(model_name)
    if augment:
        def train_transform(pil_img):
            if torch.rand(1).item() < 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            from PIL import ImageEnhance
            pil_img = ImageEnhance.Brightness(pil_img).enhance(0.9 + 0.1*torch.rand(1).item())
            pil_img = ImageEnhance.Contrast(pil_img).enhance(0.9 + 0.1*torch.rand(1).item())
            pil_img = ImageEnhance.Color(pil_img).enhance(0.9 + 0.1*torch.rand(1).item())
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
