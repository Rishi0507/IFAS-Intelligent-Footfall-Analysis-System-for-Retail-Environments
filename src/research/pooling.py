"""Ways to turn several per-frame predictions of one person into one label (RQ3).

Every strategy is a pure function over a list of votes, one per classified crop:

    votes = [(probs, cues), ...]      probs: {label: probability}
                                      cues:  {cue: value} (src/research/quality.py)

and returns (label, confidence), where confidence is the weighted mean
probability of the chosen label. `majority` is exactly what the product did
before this module existed, tie-break included.

The quality-weighted strategy needs a `QualityModel`: a logistic regression from
a crop's cues to the chance the model classifies that crop correctly. It is
fitted on labelled crops and saved as JSON, so the pipeline can load it.
"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np

from .quality import CUE_NAMES


def _labels(votes):
    seen = []
    for probs, _ in votes:
        for k in probs:
            if k not in seen:
                seen.append(k)
    return seen


def _weighted(votes, weights):
    labels = _labels(votes)
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(votes))
    totals = {c: float(sum(wi * p.get(c, 0.0) for wi, (p, _) in zip(w, votes))) for c in labels}
    label = max(labels, key=lambda c: totals[c])
    return label, totals[label] / float(w.sum())


def single_frame(votes, rng=None):
    """No pooling: the label of one crop picked at random."""
    if not votes:
        return None, 0.0
    rng = rng or np.random.default_rng(0)
    probs, _ = votes[int(rng.integers(len(votes)))]
    label = max(probs, key=probs.get)
    return label, float(probs[label])


def majority(votes):
    """Most votes wins; a tie goes to the higher summed probability (the product's rule)."""
    if not votes:
        return None, 0.0
    counts, sums = {}, {}
    for probs, _ in votes:
        top = max(probs, key=probs.get)
        counts[top] = counts.get(top, 0) + 1
        for k, v in probs.items():
            sums[k] = sums.get(k, 0.0) + v
    label = max(counts, key=lambda k: (counts[k], sums.get(k, 0.0)))
    total = sum(sums.values()) or 1.0
    return label, sums.get(label, 0.0) / total


def mean_prob(votes):
    """Argmax of the mean probability vector."""
    if not votes:
        return None, 0.0
    return _weighted(votes, np.ones(len(votes)))


def confidence_weighted(votes):
    """Mean probability, each crop weighted by its own top probability."""
    if not votes:
        return None, 0.0
    return _weighted(votes, [max(p.values()) for p, _ in votes])


def quality_weighted(votes, model):
    """Mean probability, each crop weighted by P(correct | cues) from `model`."""
    if not votes:
        return None, 0.0
    if model is None:
        raise ValueError("quality-weighted pooling needs a fitted QualityModel")
    return _weighted(votes, model.predict([c or {} for _, c in votes]))


STRATEGIES = ["single", "majority", "mean", "confidence", "quality"]


def pool(votes, strategy="majority", model=None, rng=None):
    if strategy == "single":
        return single_frame(votes, rng)
    if strategy == "majority":
        return majority(votes)
    if strategy == "mean":
        return mean_prob(votes)
    if strategy == "confidence":
        return confidence_weighted(votes)
    if strategy == "quality":
        return quality_weighted(votes, model)
    raise ValueError(f"unknown pooling strategy: {strategy}")


# ----------------------------------------------------------------- quality model
def cue_features(cues):
    """Cue dict -> feature vector, with None for anything not measured.

    Laplacian variance and size are spread over orders of magnitude, so they go
    in as logs.
    """
    def get(k):
        v = cues.get(k)
        return None if v is None else float(v)
    lap, size, motion = get("lap_var"), get("size"), get("motion")
    return [
        get("hf_ratio"),
        None if lap is None else math.log10(lap + 1e-6),
        None if motion is None else math.log1p(max(0.0, motion)),
        get("truncated"),
        get("occlusion"),
        None if size is None else math.log10(size + 1e-6),
        get("det_conf"),
    ]


FEATURE_NAMES = ["hf_ratio", "log_lap_var", "log1p_motion", "truncated", "occlusion",
                 "log_size", "det_conf"]
assert len(FEATURE_NAMES) == len(CUE_NAMES)


class QualityModel:
    """L2-regularised logistic regression, fitted by Newton's method.

    Features are standardised with the training mean and standard deviation. A
    cue that was not measured for a crop is filled with the training mean, so it
    adds nothing either way.
    """

    def __init__(self, coef=None, intercept=0.0, mean=None, std=None, l2=1.0, meta=None):
        self.coef = None if coef is None else np.asarray(coef, float)
        self.intercept = float(intercept)
        self.mean = None if mean is None else np.asarray(mean, float)
        self.std = None if std is None else np.asarray(std, float)
        self.l2 = float(l2)
        self.meta = meta or {}

    def _matrix(self, cue_dicts, fit=False):
        raw = np.array([[np.nan if v is None else v for v in cue_features(c)] for c in cue_dicts],
                       dtype=float).reshape(len(cue_dicts), len(FEATURE_NAMES))
        if fit:
            with warnings.catch_warnings():               # a cue never measured is all-NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(raw, axis=0)
            mean = np.where(np.isnan(mean), 0.0, mean)
            filled = np.where(np.isnan(raw), mean, raw)
            std = filled.std(axis=0)
            self.mean, self.std = mean, np.where(std > 1e-9, std, 1.0)
        filled = np.where(np.isnan(raw), self.mean, raw)
        return (filled - self.mean) / self.std

    def fit(self, cue_dicts, correct, iters=50):
        X = self._matrix(cue_dicts, fit=True)
        y = np.asarray(correct, float)
        n, d = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])
        w = np.zeros(d + 1)
        reg = np.full(d + 1, self.l2)
        reg[0] = 0.0                                  # the intercept is not shrunk
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
            grad = Xb.T @ (p - y) + reg * w
            hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(reg) + 1e-9 * np.eye(d + 1)
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.intercept, self.coef = float(w[0]), w[1:]
        self.meta.update({"n": int(n), "positives": int(y.sum())})
        return self

    def predict(self, cue_dicts):
        """P(the crop is classified correctly) for each cue dict."""
        X = self._matrix(cue_dicts)
        return 1.0 / (1.0 + np.exp(-np.clip(X @ self.coef + self.intercept, -30, 30)))

    def to_dict(self):
        return {"features": FEATURE_NAMES, "coef": [round(float(c), 6) for c in self.coef],
                "intercept": round(self.intercept, 6), "mean": self.mean.tolist(),
                "std": self.std.tolist(), "l2": self.l2, "meta": self.meta}

    @classmethod
    def from_dict(cls, d):
        if d.get("features") != FEATURE_NAMES:
            raise ValueError("quality model was fitted on a different feature set")
        return cls(d["coef"], d["intercept"], d["mean"], d["std"], d.get("l2", 1.0), d.get("meta"))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
