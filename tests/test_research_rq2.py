"""Tests for the RQ2 runner (src/research/rq2.py) without downloading a ViT."""
import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
from PIL import Image  # noqa: E402

from src.research.rq2 import (  # noqa: E402
    LabelledImages, aggregate, cell_name, cells, predict, run_name, score,
)


def to_tensor(img):
    return torch.tensor(np.asarray(img.resize((8, 16)), dtype=np.float32) / 255).permute(2, 0, 1)


class Brightness(torch.nn.Module):
    """Stand-in classifier: 'male' when the image is bright."""
    def forward(self, x):
        m = x.mean(dim=(1, 2, 3))
        return torch.stack([0.5 - m, m - 0.5], dim=1) * 20


@pytest.fixture
def images(tmp_path):
    items = []
    for i in range(12):
        level = 200 if i % 2 else 40
        p = tmp_path / f"{i}.png"
        Image.fromarray(np.full((64, 32, 3), level, np.uint8)).save(p)
        items.append((str(p), i % 2))
    return items


def test_cells_start_clean_and_cover_the_suite():
    cs = cells()
    assert cs[0] == ("clean", None) and len(cs) == 1 + 17
    assert cell_name("gaussian", 2.0) == "gaussian:2.0" and cell_name("clean", None) == "clean"


def test_degraded_items_are_reproducible(images):
    ds = LabelledImages(images, to_tensor, degrade=("occlusion", 0.5))
    a, la = ds[3]
    b, lb = ds[3]
    assert torch.equal(a, b) and la == lb == 1


def test_predict_and_score_with_a_stand_in_model(images):
    p = predict(Brightness(), LabelledImages(images, to_tensor), "cpu", batch_size=5, num_workers=0)
    assert p.shape == (12,)
    s = score([lab for _, lab in images], p)
    assert s == {"accuracy": 1.0, "balanced_accuracy": 1.0, "n": 12}


def _fake_run(d, cond, seed, clean_correct):
    labels = [0, 1] * 10
    p = [float(l) if i < clean_correct else 1.0 - l for i, l in enumerate(labels)]
    bal = score(labels, p)
    (d / f"{run_name(cond, seed)}.json").write_text(json.dumps({
        "condition": cond, "seed": seed, "test": {"clean": bal, "gaussian:2.0": bal},
        "packs": {}}), encoding="utf-8")
    with gzip.open(d / f"{run_name(cond, seed)}.preds.gz", "wt", encoding="utf-8") as f:
        json.dump({"test_labels": labels, "test": {"clean": p}}, f)


def test_aggregate_means_over_seeds_and_pairs_with_baseline(tmp_path):
    _fake_run(tmp_path, "baseline", 0, 14)
    _fake_run(tmp_path, "baseline", 1, 16)
    _fake_run(tmp_path, "fda+degrade", 0, 18)
    _fake_run(tmp_path, "fda+degrade", 1, 20)
    out = aggregate(tmp_path)
    rows = {r["condition"]: r for r in out["rows"]}
    assert out["meta"]["runs"] == 4
    assert rows["baseline"]["test"]["clean"]["balanced_accuracy_mean"] == pytest.approx(0.75)
    assert rows["fda+degrade"]["test"]["clean"]["balanced_accuracy_mean"] == pytest.approx(0.95)
    vs = rows["fda+degrade"]["vs_baseline"]
    assert [v["seed"] for v in vs] == [0, 1]
    assert vs[0]["clean_diff"] == pytest.approx(0.2) and vs[0]["mcnemar"]["c"] == 0
    assert "vs_baseline" not in rows["baseline"]


def test_train_run_skips_a_finished_run(tmp_path):
    pytest.importorskip("transformers")
    from src.research.rq2 import train_run
    (tmp_path / "degrade_seed3.json").write_text('{"condition": "degrade", "seed": 3}', encoding="utf-8")
    assert train_run("degrade", 3, "unused.csv", "unused", tmp_path) == {"condition": "degrade", "seed": 3}
