"""Metrics and significance tests shared by every experiment in the paper.

Accuracy alone hides two things the paper has to report: class imbalance (most
attribute datasets have more of one gender or age group) and uncertainty (one
split, a few hundred people). So every headline number goes through one of:

* `balanced_accuracy` / `mean_accuracy` (mA, the standard PAR metric),
* `off_by_one_accuracy` / `mean_absolute_group_error` for ordered age groups,
* `bootstrap_ci` / `paired_bootstrap_diff` for confidence intervals,
* `mcnemar` for two models scored on the same people,
* `cohen_kappa` for agreement between two annotators,
* `mean_sd` for results over several training seeds.

Bootstrap resampling is over *units* (people, not frames): frames of one person
are not independent, so resampling them would understate the uncertainty.
"""
from __future__ import annotations
import math

import numpy as np


# --------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------
def balanced_accuracy(y_true, y_pred, labels=None):
    """Mean recall over classes. Equals mA for a single binary attribute.

    `labels` fixes the classes to average over; by default, those in y_true.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred differ in shape")
    labels = np.unique(y_true) if labels is None else np.asarray(labels)
    recalls = [np.mean(y_pred[y_true == c] == c) for c in labels if np.any(y_true == c)]
    if not recalls:
        raise ValueError("no samples for any requested label")
    return float(np.mean(recalls))


def mean_accuracy(y_true, y_pred):
    """mA over binary attributes: per attribute, the mean of the true-positive and
    true-negative rates, then averaged over attributes (Deng et al., 2014).

    y_true, y_pred: (n_samples, n_attributes) arrays of 0/1. Returns
    (mean mA, per-attribute mA array). An attribute with no positives (or no
    negatives) in y_true gets NaN and is left out of the mean.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if y_true.ndim == 1:
        y_true, y_pred = y_true[:, None], y_pred[:, None]
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred differ in shape")
    pos, neg = y_true.sum(0), (~y_true).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = (y_true & y_pred).sum(0) / pos
        tnr = (~y_true & ~y_pred).sum(0) / neg
    per = np.where((pos > 0) & (neg > 0), (tpr + tnr) / 2, np.nan)
    if np.all(np.isnan(per)):
        raise ValueError("no attribute has both positives and negatives")
    return float(np.nanmean(per)), per


# --------------------------------------------------------------------------
# Ordered age groups
# --------------------------------------------------------------------------
def off_by_one_accuracy(true_idx, pred_idx):
    """Share of predictions at most one age group away from the truth."""
    d = np.abs(np.asarray(true_idx) - np.asarray(pred_idx))
    return float(np.mean(d <= 1))


def mean_absolute_group_error(true_idx, pred_idx):
    """Mean distance in age groups between prediction and truth."""
    return float(np.mean(np.abs(np.asarray(true_idx) - np.asarray(pred_idx))))


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------
def bootstrap_ci(metric, *arrays, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of `metric(*arrays)`, resampling rows together.

    Each array's first axis is the unit to resample (one row per person).
    Returns (point estimate, lower, upper).
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if n == 0 or any(len(a) != n for a in arrays):
        raise ValueError("arrays must be non-empty and the same length")
    rng = np.random.default_rng(seed)
    point = float(metric(*arrays))
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            stats[i] = metric(*(a[idx] for a in arrays))
        except ValueError:          # e.g. a resample without one of the classes
            stats[i] = np.nan
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        raise ValueError("metric failed on every bootstrap resample")
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def paired_bootstrap_diff(metric, y_true, pred_a, pred_b, **kw):
    """CI of metric(y_true, pred_a) - metric(y_true, pred_b) on the same units.

    An interval that excludes 0 means A and B differ at level `alpha`.
    """
    return bootstrap_ci(lambda t, a, b: metric(t, a) - metric(t, b),
                        y_true, pred_a, pred_b, **kw)


def mcnemar(correct_a, correct_b, exact=None):
    """McNemar's test for two classifiers scored on the same items.

    correct_a, correct_b: boolean arrays, True where each model was right.
    Only discordant items count: b = A right and B wrong, c = A wrong and B
    right. Uses the exact binomial test when b + c < 25 (or exact=True),
    otherwise chi-square with continuity correction. Returns a dict with
    b, c, statistic (None for the exact test) and the two-sided p-value.
    """
    from scipy.stats import binomtest, chi2

    a, b_ = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    if a.shape != b_.shape:
        raise ValueError("correct_a and correct_b differ in shape")
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    if b + c == 0:
        return {"b": 0, "c": 0, "statistic": None, "p": 1.0}
    if exact or (exact is None and b + c < 25):
        return {"b": b, "c": c, "statistic": None,
                "p": float(binomtest(b, b + c, 0.5).pvalue)}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(stat), "p": float(chi2.sf(stat, 1))}


def cohen_kappa(labels_a, labels_b):
    """Cohen's kappa between two annotators over the same items.

    Drop items either annotator marked unsure before calling, and report how
    many were dropped.
    """
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    if a.shape != b.shape or len(a) == 0:
        raise ValueError("labels must be non-empty and the same length")
    cats = np.union1d(a, b)
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == k) * np.mean(b == k) for k in cats))
    if math.isclose(pe, 1.0):
        return 1.0                  # both annotators used one identical label throughout
    return (po - pe) / (1 - pe)


def mean_sd(values):
    """Mean and sample standard deviation (ddof=1) over seeds."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        raise ValueError("no values")
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return float(np.mean(v)), sd
