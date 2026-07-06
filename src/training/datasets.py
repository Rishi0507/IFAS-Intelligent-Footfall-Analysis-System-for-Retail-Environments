"""Dataset classes for PETA, RAP, SCface, and stage-2 user folder."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import torch
from PIL import Image
from torch.utils.data import Dataset

GENDER_TO_IDX = {"female": 0, "male": 1}
IDX_TO_GENDER = {0: "female", 1: "male"}

class CSVDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.image_root = Path(image_root)
        self.transform = transform
        self.rows = []
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                g = r["gender"].strip().lower()
                if g not in GENDER_TO_IDX: continue
                p = Path(r["image"])
                if not p.is_absolute(): p = self.image_root / p
                if not p.exists(): continue
                self.rows.append((p, GENDER_TO_IDX[g]))
        if not self.rows: raise RuntimeError(f"No valid rows in {csv_path}")
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        p, lab = self.rows[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None: img = self.transform(img)
        return img, lab

class FolderPerClassDataset(Dataset):
    def __init__(self, root, transform=None, exts=(".jpg",".jpeg",".png")):
        self.root = Path(root); self.transform = transform; self.rows = []
        for name, idx in GENDER_TO_IDX.items():
            d = self.root / name
            if not d.exists(): continue
            for ext in exts:
                for img in d.glob(f"*{ext}"): self.rows.append((img, idx))
                for img in d.glob(f"*{ext.upper()}"): self.rows.append((img, idx))
        if not self.rows: raise RuntimeError(f"No images in {self.root}/{{male,female}}/")
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        p, lab = self.rows[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None: img = self.transform(img)
        return img, lab

class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets; self.cum = [0]
        for d in datasets: self.cum.append(self.cum[-1] + len(d))
    def __len__(self): return self.cum[-1]
    def __getitem__(self, idx):
        for i, c in enumerate(self.cum[1:]):
            if idx < c: return self.datasets[i][idx - self.cum[i]]
        raise IndexError(idx)

def stratified_split(rows, val_ratio=0.1, seed=42):
    g = torch.Generator().manual_seed(seed)
    by_class = {}
    for i, (_, lab) in enumerate(rows): by_class.setdefault(lab, []).append(i)
    train_idx, val_idx = [], []
    for lab, idxs in by_class.items():
        perm = torch.randperm(len(idxs), generator=g).tolist()
        n_val = int(len(idxs) * val_ratio)
        for j, p in enumerate(perm):
            (val_idx if j < n_val else train_idx).append(idxs[p])
    return train_idx, val_idx
