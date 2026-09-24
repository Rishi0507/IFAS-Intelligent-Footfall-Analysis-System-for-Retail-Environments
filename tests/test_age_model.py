"""Tests for the age model wrapper, built against a real (tiny) SVM bundle.

The PETA-trained bundle is far too large to ship, so these tests train
throwaway SVMs on random features - the point is the loading contract and the
feature layout, not the accuracy.
"""
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")
pytest.importorskip("skimage")
pytest.importorskip("sklearn")
from PIL import Image  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from src.inference.models import (  # noqa: E402
    AGE_CLASSES, AgeModel, extract_features, intersection_kernel,
)

# 5 regions x (3 RGB + 2 HSV channels x 16 bins + 3 LBP radii x 10 bins)
EXPECTED_DIM = 5 * (5 * 16 + 3 * 10)


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    return [Image.fromarray(rng.integers(0, 255, (120, 60, 3), dtype=np.uint8)) for _ in range(24)]


def _bundle(tmp_path, images, kernels=("intersection", "rbf")):
    X = np.vstack([extract_features(i) for i in images])
    rng = np.random.default_rng(1)
    y = rng.integers(0, len(AGE_CLASSES), len(X))
    K = intersection_kernel(X, X)
    classifiers, configs = {}, {}
    for idx, cls in enumerate(AGE_CLASSES):
        yb = (y == idx).astype(int)
        yb[0], yb[1] = 1, 0                      # both classes must be present
        kernel = kernels[idx % len(kernels)]
        if kernel == "intersection":
            classifiers[cls] = SVC(kernel="precomputed", C=1).fit(K, yb)
            configs[cls] = ("intersection", 1, None)
        else:
            classifiers[cls] = SVC(kernel="rbf", C=1, gamma=0.01).fit(X, yb)
            configs[cls] = ("rbf", 1, 0.01)
    out = tmp_path / "age"
    out.mkdir()
    with open(out / "classifiers.pkl", "wb") as f:
        pickle.dump({"classifiers": classifiers, "configs": configs}, f)
    np.save(out / "support.npy", X)
    return out


def test_feature_vector_matches_the_notebook_layout(images):
    assert extract_features(images[0]).shape == (EXPECTED_DIM,)


def test_predict_returns_a_distribution_over_all_four_buckets(tmp_path, images):
    model = AgeModel(_bundle(tmp_path, images))
    out = model.predict(images[:4])
    assert len(out) == 4
    for dist in out:
        assert set(dist) == set(AGE_CLASSES)
        assert abs(sum(dist.values()) - 1.0) < 1e-9
        assert all(0.0 <= p <= 1.0 for p in dist.values())


def test_predict_handles_an_empty_batch(tmp_path, images):
    assert AgeModel(_bundle(tmp_path, images)).predict([]) == []


def test_intersection_kernel_bundle_requires_the_support_matrix(tmp_path, images):
    bundle = _bundle(tmp_path, images)
    (bundle / "support.npy").unlink()
    with pytest.raises(ValueError, match="support.npy"):
        AgeModel(bundle)


def test_rbf_only_bundle_needs_no_support_matrix(tmp_path, images):
    bundle = _bundle(tmp_path, images, kernels=("rbf",))
    (bundle / "support.npy").unlink()
    model = AgeModel(bundle)
    assert len(model.predict(images[:2])) == 2


def test_a_bundle_without_configs_says_so(tmp_path, images):
    bundle = _bundle(tmp_path, images)
    with open(bundle / "classifiers.pkl", "rb") as f:
        blob = pickle.load(f)
    with open(bundle / "classifiers.pkl", "wb") as f:
        pickle.dump(blob["classifiers"], f)          # classifiers only, no configs
    with pytest.raises(ValueError, match="configs"):
        AgeModel(bundle)
