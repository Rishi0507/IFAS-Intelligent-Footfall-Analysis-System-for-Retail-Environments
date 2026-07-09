# Two-stage fine-tuning guide (Gender) and Age Attribute Model Guide

This document explains the **what** and **why** of two-stage fine-tuning. The **how** is in `training_pipeline.ipynb` - open it in Colab and run the cells.

## Overview

```
Stage 1: PETA  →  checkpoints/stage1/best/  (saved to Google Drive)
Stage 2: your data, lower LR  →  checkpoints/stage2/best/   (used at inference)
```

The age model follows a single-stage classical pipeline instead:

```
Age model: PETA (age attributes)  →  region-based color + texture histograms  →  per-attribute SVMs  →  classifiers.pkl  (saved locally / Google Drive)
```

## Why ViT, not CLIP?

The backbone is `google/vit-base-patch16-224` - a **pure vision transformer** pretrained on ImageNet-21k.

CLIP is a vision-language model (VLM). Fine-tuning it for a binary gender head throws away the text encoder and reduces a powerful multi-modal model to a simple classifier. ViT is a pure vision backbone - better suited for fine-tuning on a visual attribute task.

## Why classical CV, not deep learning, for age?

The age model replicates the methodology of the original PETA benchmark (Deng et al., 2014): region-based color and texture histograms as features, with one independent histogram-intersection-kernel SVM per attribute (one vs rest), evaluated with per-attribute mean Accuracy (mA). No CNNs, no deep learning, no ensembling, classical computer vision only.

This keeps the age model directly comparable to the published baseline, rather than trading that comparability for the deep ViT backbone used for gender.

## Stage 1 - PETA

- **Phase A** (3 epochs, linear probe): backbone frozen, head only @ LR 1e-3.
- **Phase B** (10 epochs, full fine-tune): backbone unfrozen @ LR 1e-5 with cosine decay + 10% warmup.
- Best checkpoint saved to `checkpoints/stage1/best/` and copied to Google Drive.

## Stage 2 - Your data, lower LR

- **Phase A** (2 epochs, linear probe @ 1e-4): 1/10 of stage 1.
- **Phase B** (5 epochs, full fine-tune @ 1e-6): 1/10 of stage 1.
- Best checkpoint saved to `checkpoints/stage2/best/` and copied to Google Drive.

## Age model - PETA (single stage)

- **Feature extraction**: each image is split into horizontal body regions. Per region, 16-bin RGB and HSV color histograms plus a uniform LBP texture histogram (radii 1, 2, 3), all L1-normalized, concatenated across regions.
- **Per-attribute SVMs**: one SVM per age attribute (one vs rest), grid searched over a histogram-intersection kernel and an RBF kernel (`C` and `gamma`), best configuration per attribute chosen by validation mA.
- No backbone to freeze or unfreeze and no linear probe / full fine-tune phases, it is a single training pass per attribute.
- Best classifiers saved to `classifiers.pkl` (not pushed to Hugging Face Hub).

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

For the age classifier, you also do not need to re-run feature extraction and SVM training as long as `classifiers.pkl` has been saved.

1. Reopen `age_classifier_v3.ipynb` in Colab.
2. Run the setup cell and, if needed, mount Drive.
3. Load the saved classifiers directly instead of retraining:
   ```python
   import pickle
   with open('classifiers.pkl', 'rb') as f:
       saved = pickle.load(f)
   classifiers, configs = saved['classifiers'], saved['configs']
   ```

## How to test on a video

### Option A: Quick test (in training_pipeline.ipynb, cell 27)

After stage 1 (or stage 2) finishes, run the video test cell. It:
1. Uploads a video
2. Loads YOLOv8n + your trained ViT
3. For each frame: detects people → classifies gender → draws colored boxes
4. Writes an annotated MP4 and downloads it

This is the fastest way to see if your model works. No tracking, no ReID, no footfall, just raw detection + classification.

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

### Option C: Age model video test (in age_classifier_v3.ipynb)

After the SVMs finish training, run the video test cells. They:
1. Upload a video
2. Load a Faster R-CNN (ResNet-50 FPN) person detector and your trained per-attribute SVMs
3. For each frame: detect people → crop each person → classify age bucket → draw boxes and labels
4. Write an annotated MP4 and download it

## Datasets

See `scripts/download_datasets.md` for where to download the PETA dataset.

| Dataset  | Images  | Gender source |
|----------|---------|---------------|
| PETA     | ~19,000 | `personalMale` attribute or `Label.txt` files |

| Dataset  | Images  | Age source |
|----------|---------|------------|
| PETA     | ~14,000 | `personalLess30` / `personalLess45` / `personalLess60` / `personalLarger60` attributes in `Label.txt` (`personalLess15` dropped) |

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

### If the age model underperforms on a specific bucket
- Widen the `C_GRID` and `GAMMA_GRID` search ranges for that attribute.
- Check class balance; `class_weight="balanced"` is already used, but very small buckets (such as `AgeAbove61`) may still need more PETA samples or targeted local data.
- Confirm the region split and histogram bin counts (`N_REGIONS`, `COLOR_BINS`, `LBP_POINTS`, `LBP_RADII`) still match the configuration used during validation.

## Sharing checkpoints with your team

Checkpoints are too large for GitHub (340 MB > 100 MB limit). Use one of:

1. **HuggingFace Hub** (recommended):
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload abhshkp/footfall-analysis-vit-stage1 checkpoints/stage1/best --repo-type=model
   ```
   Teammates download:
   ```bash
   huggingface-cli download abhshkp/footfall-analysis-vit-stage1 --local-dir checkpoints/stage1/best
   ```

2. **Google Drive**: share the `IPD_checkpoints/` folder with "Anyone with the link".

3. **WeTransfer**: one-time transfer, no account needed.

The age model's `classifiers.pkl` (roughly 79 MB, about 35 MB zipped) is well under the GitHub limit but is still kept out of Git for consistency; share it the same way, via Hugging Face Hub, Google Drive, or WeTransfer.

## What's deliberately NOT here

- CLIP fine-tuning - ViT only (for the gender model).
- CLI entry points - notebooks only.
- Distributed training - single-GPU only.
