import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import numpy as np
from PIL import Image

def test_dataset_csv(tmp_path):
    from src.training.datasets import CSVDataset, GENDER_TO_IDX
    img_dir = tmp_path / "imgs"; img_dir.mkdir()
    rows = []
    for i, g in enumerate(["male", "female", "male", "female"]):
        Image.fromarray(np.zeros((32,32,3), dtype=np.uint8)).save(img_dir / f"{i}.jpg")
        rows.append({"image": f"{i}.jpg", "gender": g})
    import csv
    with open(tmp_path / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image","gender"]); w.writeheader(); w.writerows(rows)
    ds = CSVDataset(str(tmp_path/"labels.csv"), str(img_dir), transform=None)
    assert len(ds) == 4

def test_folder_per_class_dataset(tmp_path):
    from src.training.datasets import FolderPerClassDataset
    (tmp_path / "male").mkdir(); (tmp_path / "female").mkdir()
    for i in range(3): Image.fromarray(np.zeros((32,32,3),dtype=np.uint8)).save(tmp_path/"male"/f"{i}.jpg")
    for i in range(2): Image.fromarray(np.zeros((32,32,3),dtype=np.uint8)).save(tmp_path/"female"/f"{i}.jpg")
    ds = FolderPerClassDataset(str(tmp_path), transform=None)
    assert len(ds) == 5

def test_stratified_split():
    from src.training.datasets import stratified_split
    rows = [(None, 0)] * 100 + [(None, 1)] * 100
    tr, va = stratified_split(rows, val_ratio=0.1, seed=42)
    assert len(tr) == 180 and len(va) == 20
