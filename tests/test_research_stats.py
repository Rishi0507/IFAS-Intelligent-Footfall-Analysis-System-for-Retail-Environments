"""Tests for src/research/stats.py against hand-computed values."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from src.research.stats import (  # noqa: E402
    balanced_accuracy, bootstrap_ci, cohen_kappa, mcnemar, mean_absolute_group_error,
    mean_accuracy, mean_sd, off_by_one_accuracy, paired_bootstrap_diff,
)


def test_balanced_accuracy_averages_recall_per_class():
    # class 0: 2 of 3 right, class 1: 1 of 1 right
    assert balanced_accuracy([0, 0, 0, 1], [0, 0, 1, 1]) == pytest.approx((2 / 3 + 1) / 2)


def test_balanced_accuracy_matches_sklearn_on_strings():
    sk = pytest.importorskip("sklearn.metrics")
    t = ["female", "male", "male", "female", "male", "female", "female"]
    p = ["female", "male", "female", "female", "female", "male", "female"]
    assert balanced_accuracy(t, p) == pytest.approx(sk.balanced_accuracy_score(t, p))


def test_mean_accuracy_per_attribute_and_mean():
    y = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    p = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    mean, per = mean_accuracy(y, p)
    # attr 0: TPR 1/2, TNR 2/2 -> 0.75; attr 1: TPR 1/2, TNR 2/2 -> 0.75
    assert per.tolist() == [0.75, 0.75]
    assert mean == 0.75


def test_mean_accuracy_skips_attribute_without_positives():
    y = np.array([[1, 0], [0, 0]])
    p = np.array([[1, 1], [0, 0]])
    mean, per = mean_accuracy(y, p)
    assert math.isnan(per[1]) and mean == 1.0


def test_single_binary_attribute_ma_equals_balanced_accuracy():
    t = np.array([1, 1, 0, 0, 0])
    p = np.array([1, 0, 0, 0, 1])
    assert mean_accuracy(t, p)[0] == pytest.approx(balanced_accuracy(t, p))


def test_ordinal_age_metrics():
    true, pred = [0, 1, 2, 3], [1, 1, 0, 3]          # distances 1, 0, 2, 0
    assert off_by_one_accuracy(true, pred) == 0.75
    assert mean_absolute_group_error(true, pred) == 0.75


def test_mcnemar_exact_for_few_discordant_items():
    a = np.array([False] * 5 + [True] * 10)           # A wrong on 5 items B gets right
    b = np.array([True] * 5 + [True] * 10)
    r = mcnemar(a, b)
    assert (r["b"], r["c"]) == (0, 5)
    assert r["p"] == pytest.approx(2 * 0.5 ** 5)        # two-sided binomial


def test_mcnemar_chi_square_with_continuity_correction():
    from scipy.stats import chi2
    a = np.array([True] * 10 + [False] * 30 + [True] * 60)
    b = np.array([False] * 10 + [True] * 30 + [True] * 60)
    r = mcnemar(a, b)
    assert (r["b"], r["c"]) == (10, 30)
    assert r["statistic"] == pytest.approx((20 - 1) ** 2 / 40)
    assert r["p"] == pytest.approx(chi2.sf((20 - 1) ** 2 / 40, 1))


def test_mcnemar_identical_models():
    x = np.array([True, False, True])
    assert mcnemar(x, x)["p"] == 1.0


def test_cohen_kappa_hand_computed_and_sklearn():
    # po = 0.75, pe = 0.5*0.25 + 0.5*0.75 = 0.5 -> kappa = 0.5
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    sk = pytest.importorskip("sklearn.metrics")
    a = ["f", "m", "m", "f", "m", "f", "f", "m"]
    b = ["f", "m", "f", "f", "m", "m", "f", "m"]
    assert cohen_kappa(a, b) == pytest.approx(sk.cohen_kappa_score(a, b))


def test_cohen_kappa_perfect_single_label():
    assert cohen_kappa(["f", "f"], ["f", "f"]) == 1.0


def test_bootstrap_ci_contains_point_and_is_deterministic():
    rng = np.random.default_rng(3)
    t = rng.integers(0, 2, 200)
    p = np.where(rng.random(200) < 0.8, t, 1 - t)
    acc = lambda a, b: float(np.mean(a == b))
    r1 = bootstrap_ci(acc, t, p, n_boot=500, seed=1)
    r2 = bootstrap_ci(acc, t, p, n_boot=500, seed=1)
    assert r1 == r2
    point, lo, hi = r1
    assert lo <= point <= hi and 0.7 < point < 0.9 and hi - lo < 0.15


def test_bootstrap_ci_of_a_perfect_model_is_degenerate():
    t = np.array([0, 1] * 20)
    assert bootstrap_ci(lambda a, b: float(np.mean(a == b)), t, t, n_boot=200) == (1.0, 1.0, 1.0)


def test_paired_diff_of_identical_predictions_is_zero():
    t = np.array([0, 1, 1, 0, 1, 0] * 10)
    p = np.array([0, 1, 0, 0, 1, 1] * 10)
    assert paired_bootstrap_diff(balanced_accuracy, t, p, p, n_boot=200) == (0.0, 0.0, 0.0)


def test_paired_diff_detects_a_clearly_better_model():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, 300)
    good = np.where(rng.random(300) < 0.95, t, 1 - t)
    bad = np.where(rng.random(300) < 0.6, t, 1 - t)
    point, lo, hi = paired_bootstrap_diff(balanced_accuracy, t, good, bad, n_boot=500)
    assert lo > 0 and point > 0.2


def test_mean_sd_over_seeds():
    assert mean_sd([1, 2, 3]) == (2.0, 1.0)
    assert mean_sd([0.9]) == (0.9, 0.0)
