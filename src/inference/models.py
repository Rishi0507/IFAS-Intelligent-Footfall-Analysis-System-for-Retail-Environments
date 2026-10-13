"""Model wrappers used by the inference pipeline.

Two very different models sit behind one interface:

* `GenderModel` - the fine-tuned ViT checkpoint (HF Hub or a local directory).
* `AgeModel`    - the classical PETA baseline: region colour + LBP texture
                  histograms, one one-vs-rest SVM per age bucket.

Both return a *distribution* rather than a bare label, because the dashboard's
confidence analytics need per-class scores.
"""
from __future__ import annotations
import json
import math
import pickle
from pathlib import Path

import numpy as np

AGE_CLASSES = ["Age16-30", "Age31-45", "Age46-60", "AgeAbove61"]

# Feature-extraction defaults. A bundle that records its own `hyperparams` (the
# Hub bundle does) overrides these; they only apply to older bundles that do not,
# and must then match the notebook run that trained them, otherwise the
# serialized SVMs see a different feature space.
N_REGIONS = 5
COLOR_BINS = 16
LBP_POINTS = 8
LBP_RADII = [1, 2, 3]


def _softmax(scores, temperature=1.0):
    m = max(scores)
    exp = [math.exp((s - m) / temperature) for s in scores]
    total = sum(exp)
    return [e / total for e in exp]


# --------------------------------------------------------------------------
# Gender: ViT
# --------------------------------------------------------------------------
class GenderModel:
    def __init__(self, ckpt_dir, device="cpu", batch_size=16):
        import torch
        import torch.nn as nn
        from transformers import ViTModel, ViTImageProcessor

        ckpt_dir = Path(ckpt_dir)
        with open(ckpt_dir / "config.json") as f:
            cfg = json.load(f)
        self.name = cfg.get("vit_model_name", "google/vit-base-patch16-224")
        self.id2label = {int(k): v for k, v in cfg.get("id2label", {0: "female", 1: "male"}).items()}

        class _Net(nn.Module):
            def __init__(self, backbone, num_labels):
                super().__init__()
                self.vit = ViTModel.from_pretrained(backbone)
                self.head = nn.Linear(self.vit.config.hidden_size, num_labels)

            def forward(self, pixel_values):
                return self.head(self.vit(pixel_values=pixel_values).pooler_output)

        self._torch = torch
        self.model = _Net(self.name, cfg.get("num_labels", 2))
        state = torch.load(ckpt_dir / "pytorch_model.bin", map_location=device)
        self.model.load_state_dict(state, strict=False)
        self.model.to(device).eval()
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.device = device
        self.batch_size = batch_size

    @classmethod
    def from_hub(cls, repo_id, cache_dir, device="cpu", **kw):
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=repo_id, local_dir=str(cache_dir), repo_type="model")
        m = cls(local, device=device, **kw)
        m.source = repo_id
        return m

    def predict(self, crops):
        """crops: list[PIL.Image] -> list[{label: prob}]"""
        out = []
        for i in range(0, len(crops), self.batch_size):
            chunk = crops[i:i + self.batch_size]
            pv = self.processor(images=chunk, return_tensors="pt")["pixel_values"].to(self.device)
            with self._torch.no_grad():
                probs = self._torch.softmax(self.model(pv), dim=-1).cpu().numpy()
            for row in probs:
                out.append({self.id2label[j]: float(p) for j, p in enumerate(row)})
        return out


# --------------------------------------------------------------------------
# Age: region histograms + per-attribute SVMs
# --------------------------------------------------------------------------
def region_split(img, n_regions=N_REGIONS):
    w, h = img.size
    step = max(h // n_regions, 1)
    regions = []
    for i in range(n_regions):
        top = i * step
        bottom = (i + 1) * step if i < n_regions - 1 else h
        regions.append(img.crop((0, top, w, max(bottom, top + 1))))
    return regions


def color_histogram(region, bins=COLOR_BINS):
    rgb = np.array(region.convert("RGB")).astype(np.float32)
    hsv = np.array(region.convert("HSV")).astype(np.float32)
    hist = []
    for c in range(3):
        h, _ = np.histogram(rgb[:, :, c], bins=bins, range=(0, 255))
        hist.append(h / (h.sum() + 1e-8))
    for c in range(2):
        h, _ = np.histogram(hsv[:, :, c], bins=bins, range=(0, 255))
        hist.append(h / (h.sum() + 1e-8))
    return np.concatenate(hist)


def texture_histogram(region, points=LBP_POINTS, radii=None):
    from skimage.feature import local_binary_pattern
    radii = radii or LBP_RADII
    gray = np.array(region.convert("L"))
    hists = []
    for r in radii:
        lbp = local_binary_pattern(gray, points, r, method="uniform")
        n_bins = points + 2
        h, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
        hists.append(h / (h.sum() + 1e-8))
    return np.concatenate(hists)


def extract_features(pil_img, n_regions=N_REGIONS, bins=COLOR_BINS, points=LBP_POINTS, radii=None):
    feats = []
    for region in region_split(pil_img.convert("RGB"), n_regions):
        feats.append(color_histogram(region, bins))
        feats.append(texture_histogram(region, points, radii))
    return np.concatenate(feats)


def intersection_kernel(X, Y):
    from scipy.spatial.distance import cdist
    sums_x, sums_y = X.sum(axis=1), Y.sum(axis=1)
    l1 = cdist(X, Y, metric="cityblock")
    return 0.5 * (sums_x[:, None] + sums_y[None, :]) - 0.5 * l1


class AgeModel:
    """Loads the pickled per-attribute SVMs produced by age_classifier_v3.ipynb.

    `bundle` is the Hub file `age_classifier_bundle.pkl` (see `from_hub`), a
    directory holding it, or a directory in the older hand-exported layout:

        classifiers.pkl   dict[label] -> SVC, or {"classifiers": ..., "configs": ...}
        configs.pkl       dict[label] -> (kernel, C, gamma)         [optional if bundled]
        support.npy       the X_train matrix                        [required for
                          any attribute whose selected kernel is "intersection",
                          because a precomputed-kernel SVM can only score against
                          the training features it was fitted on]

    The Hub bundle carries `configs`, `X_train` (the support matrix) and the
    feature `hyperparams` it was trained with, so it needs nothing alongside it.
    """

    HUB_FILE = "age_classifier_bundle.pkl"

    def __init__(self, bundle):
        bundle = Path(bundle)
        if bundle.is_file():
            bundle_dir, pkl = bundle.parent, bundle
        else:
            bundle_dir = bundle
            pkl = next((bundle / n for n in ("classifiers.pkl", self.HUB_FILE)
                        if (bundle / n).exists()), bundle / "classifiers.pkl")
        with open(pkl, "rb") as f:
            blob = pickle.load(f)
        if isinstance(blob, dict) and "classifiers" in blob:
            self.classifiers = blob["classifiers"]
            self.configs = blob.get("configs", {})
        else:
            self.classifiers, blob = blob, {}
            self.configs = {}
        cfg_path = bundle_dir / "configs.pkl"
        if not self.configs and cfg_path.exists():
            with open(cfg_path, "rb") as f:
                self.configs = pickle.load(f)
        if not self.configs:
            raise ValueError(
                f"No SVM configs found in {bundle_dir}. Export configs.pkl alongside "
                "classifiers.pkl (see docs/dashboard.md)."
            )

        hp = blob.get("hyperparams") or {}
        self.feature_kw = {
            "n_regions": int(hp.get("N_REGIONS", N_REGIONS)),
            "bins": int(hp.get("COLOR_BINS", COLOR_BINS)),
            "points": int(hp.get("LBP_POINTS", LBP_POINTS)),
            "radii": list(hp.get("LBP_RADII", LBP_RADII)),
        }

        support = bundle_dir / "support.npy"
        if support.exists():
            self.support = np.load(support)
        else:
            self.support = np.asarray(blob["X_train"]) if blob.get("X_train") is not None else None
        needs_support = any(str(c[0]) == "intersection" for c in self.configs.values())
        if needs_support and self.support is None:
            raise ValueError(
                f"{bundle_dir} selects an intersection-kernel SVM but has no support.npy. "
                "Re-export from the notebook with: np.save('support.npy', X_train)"
            )
        self.labels = [c for c in AGE_CLASSES if c in self.classifiers]
        self.source = str(bundle)

    @classmethod
    def from_hub(cls, repo_id, cache_dir):
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(repo_id=repo_id, filename=cls.HUB_FILE,
                                local_dir=str(cache_dir), repo_type="model")
        m = cls(local)
        m.source = repo_id
        return m

    def predict(self, crops):
        """crops: list[PIL.Image] -> list[{age_label: prob}]

        Each bucket's one-vs-rest SVM gives a signed margin; the margins are
        turned into a distribution with a softmax so the dashboard can report a
        confidence. The argmax is identical to the notebook's `max(score)` rule.
        """
        if not crops:
            return []
        X = np.vstack([extract_features(c, **self.feature_kw) for c in crops])
        K = intersection_kernel(X, self.support) if self.support is not None else None
        margins = {}
        for label in self.labels:
            kernel = str(self.configs[label][0])
            clf = self.classifiers[label]
            margins[label] = clf.decision_function(K if kernel == "intersection" else X)
        out = []
        for i in range(len(crops)):
            scores = [float(margins[l][i]) for l in self.labels]
            probs = _softmax(scores)
            out.append({l: p for l, p in zip(self.labels, probs)})
        return out
