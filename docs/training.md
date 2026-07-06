# Two-stage fine-tuning guide

This document explains the **what** and **why** of two-stage fine-tuning. The **how** is in `training_pipeline.ipynb` — open it in Colab and run the cells.

## Overview

```
Stage 1: PETA + RAP + SCface  →  checkpoints/stage1/best/  (saved to Google Drive)
Stage 2: your data, lower LR  →  checkpoints/stage2/best/   (used at inference)
```

## Why ViT, not CLIP?

The backbone is `google/vit-base-patch16-224` — a **pure vision transformer** pretrained on ImageNet-21k.

CLIP is a vision-language model (VLM). Fine-tuning it for a binary gender head throws away the text encoder and reduces a powerful multi-modal model to a simple classifier. ViT is a pure vision backbone — better suited for fine-tuning on a visual attribute task.

CLIP is used once (in the SCface auto-labeling cell) as a one-time shortcut to generate initial gender labels for the 130 SCface subjects. The trained model never touches CLIP.

## Stage 1 — PETA + RAP + SCface

- **Phase A** (3 epochs, linear probe): backbone frozen, head only @ LR 1e-3.
- **Phase B** (10 epochs, full fine-tune): backbone unfrozen @ LR 1e-5 with cosine decay + 10% warmup.
- Best checkpoint saved to `checkpoints/stage1/best/` and copied to Google Drive.

## Stage 2 — Your data, lower LR

- **Phase A** (2 epochs, linear probe @ 1e-4): 1/10 of stage 1.
- **Phase B** (5 epochs, full fine-tune @ 1e-6): 1/10 of stage 1.
- Best checkpoint saved to `checkpoints/stage2/best/` and copied to Google Drive.

## If Colab runtime disconnects

You do NOT need to re-run training. The checkpoint is saved to Google Drive.

1. Reopen `training_pipeline.ipynb` in Colab.
2. Run cells 0-1 (install deps + mount Drive).
3. Skip to the section you need:
   - To test stage 1 on a video → jump to cell 27 (video test cell)
   - To train stage 2 → jump to Section 5; the notebook auto-loads stage 1 from Drive
4. Load the checkpoint:
   ```python
   model = GenderClassifier.load('/content/drive/MyDrive/IPD_checkpoints/stage1', device=DEVICE)
   ```

## How to test on a video

### Option A: Quick test (in training_pipeline.ipynb, cell 27)

After stage 1 (or stage 2) finishes, run the video test cell. It:
1. Uploads a video
2. Loads YOLOv8n + your trained ViT
3. For each frame: detects people → classifies gender → draws colored boxes
4. Writes an annotated MP4 and downloads it

This is the fastest way to see if your model works. No tracking, no ReID, no footfall — just raw detection + classification.

### Option B: Full pipeline (gender_pipeline.ipynb)

For the complete pipeline (tracking + ReID + footfall counting + reports):
1. Open `gender_pipeline.ipynb`
2. In the CONFIG cell, set:
   ```python
   CONFIG['face_gender']['model'] = '/content/checkpoints/stage1/best'
   # or from Drive:
   # CONFIG['face_gender']['model'] = '/content/drive/MyDrive/IPD_checkpoints/stage1'
   ```
3. Upload your video (Section 10)
4. Run the pipeline (Section 11)
5. View results (Section 12): demographics pie, footfall chart, annotated frames

## Datasets

See `scripts/download_datasets.md` for where to download PETA, RAP, and SCface.

| Dataset  | Images  | Gender source |
|----------|---------|---------------|
| PETA     | ~19,000 | `personalMale` attribute or `Label.txt` files |
| RAP      | ~42,000 | gender column in `RAP_annotation.mat` |
| SCface   | ~4,160  | no public labels — auto-labeled via CLIP, you correct |

## Tuning tips

### If stage 1 overfits (val acc plateaus or drops in phase B)
- Reduce `phase_b_epochs` from 10 → 5.
- Reduce `phase_b_backbone_lr` from `1e-5` → `5e-6`.
- Increase `weight_decay` from `0.01` → `0.05`.

### If stage 2 underfits (val acc < stage 1 val acc)
- Increase `phase_b_epochs` from 5 → 10.
- Increase `phase_b_head_lr` from `1e-6` → `5e-6`.

### If your stage 2 dataset is very small (<1000 images)
- Skip phase B entirely (set `phase_b_epochs: 0` in the CONFIG cell).
- Only do linear probe at `1e-4` for 3-5 epochs.

## Sharing checkpoints with your team

Checkpoints are too large for GitHub (340 MB > 100 MB limit). Use one of:

1. **HuggingFace Hub** (recommended):
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload YOUR_USERNAME/ipd-gender-vit-stage1 checkpoints/stage1/best --repo-type=model
   ```
   Teammates download:
   ```bash
   huggingface-cli download YOUR_USERNAME/ipd-gender-vit-stage1 --local-dir checkpoints/stage1/best
   ```

2. **Google Drive**: share the `IPD_checkpoints/` folder with "Anyone with the link".

3. **WeTransfer**: one-time transfer, no account needed.

## What's deliberately NOT here

- Age classification — per your earlier directive.
- CLIP fine-tuning — ViT only.
- CLI entry points — notebooks only.
- Distributed training — single-GPU only.
