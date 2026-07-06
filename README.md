# Footfall Analysis — Gender Model Training

Fine-tune a ViT-B/16 backbone for gender classification. Two-stage training + testing.

## Notebooks

| Notebook | Purpose | When to run |
|---|---|---|
| `train_stage1.ipynb` | Train on PETA + RAP + SCface → push to HF Hub | Once (or when retraining) |
| `train_stage2.ipynb` | Pull stage 1 from HF Hub, fine-tune on your data → push to HF Hub | When you have new labeled data |
| `test_model.ipynb` | Pull checkpoint from HF Hub, test on image + video | Every time you want to verify |

## Quick start

### Test the model (you already have a checkpoint on HF Hub)
1. Open `test_model.ipynb` in Colab
2. Set `HF_REPO = 'YOUR_USERNAME/ipd-gender-vit-stage1'`
3. Run all cells — upload a video, get annotated MP4

### Train stage 1 (PETA + RAP + SCface)
1. Open `train_stage1.ipynb` in Colab (Runtime → T4 GPU)
2. Follow the cells: upload datasets, train, push to HF Hub
3. Expected time: ~4-6 hours on T4

### Train stage 2 (your data, lower LR)
1. Open `train_stage2.ipynb` in Colab
2. Set `hf_repo` to your stage 1 HF repo
3. Upload your labeled data (`data/user/{male,female}/`)
4. Run all cells — pulls stage 1 from HF, trains stage 2, pushes to HF

## Trained checkpoints

- Stage 1: https://huggingface.co/YOUR_USERNAME/ipd-gender-vit-stage1
- Stage 2: https://huggingface.co/YOUR_USERNAME/ipd-gender-vit-stage2 (after you train it)

## Repo layout

```
train_stage1.ipynb            train on PETA+RAP+SCface
train_stage2.ipynb            continue training on your data (pulls stage 1 from HF)
test_model.ipynb              test on image + video (pulls from HF)
src/training/                 model, datasets, trainer, transforms (reference code)
tests/test_training.py        unit tests
docs/training.md              training guide
scripts/download_datasets.md  PETA/RAP/SCface download links
```

## What's NOT in this version
- Old inference pipeline (gender_pipeline.ipynb) — removed
- Age classification — will add later
- CLI entry points — notebooks only
- CLIP fine-tuning — ViT only (CLIP is a VLM, less valuable for this task)
