# Hyperparameter Sensitivity Analysis (Pure SVM Baseline)

This folder documents the hyperparameter sensitivity experiments conducted for the **Pure SVM** age classification baseline, which replicates the original PETA benchmark methodology (region-based color + texture histograms with one intersection/RBF-kernel SVM per attribute).

The purpose of these experiments was to determine the feature-extraction and training configuration that offers the highest screening accuracy (average mA) on a 30% subsample of the training and validation data, before locking in the final configuration used for full-scale training.

## Summary of Hyperparameter Configurations

Below is a summary of the hyperparameters evaluated, the values tested, and the selected optimal configuration:

| Hyperparameter | Tested Values | Optimal Value | Conclusion & Impact |
| :--- | :--- | :--- | :--- |
| **[Number of Body Regions](n_regions.md)** | `2, 3, 4, 5, 6, 7` | **`4`** | Accuracy peaks at 4 regions (0.7457); finer slicing beyond this point fragments regions into strips too thin for reliable histograms, adding noise instead of detail. |
| **[Color Histogram Bin Count](color_bins.md)** | `8, 16, 24, 32` | **`16`** | 16 bins achieves the highest accuracy (0.7457); finer binning beyond 16 sparsifies histograms as region size and bin count both shrink relative to it. |
| **[LBP Sampling Points](lbp_points.md)** | `8, 16, 24` | **`16`** | 16 points achieves the highest accuracy (0.7493); 8 points misses discriminative detail, while 24 points shows diminishing returns from added noise. |
| **[Multi-scale LBP Radii](lbp_radii.md)** | `[1], [1,2], [1,2,3], [1,2,3,4], [2,3]` | **`[1, 2, 3]`** | All configurations score very similarly, indicating low sensitivity to the specific radii combination; `[1,2,3]` scores marginally highest. |
| **[Class Weighting](class_weight.md)** | `none`, `balanced` | **`balanced`** | Balanced weighting achieves a clearly higher accuracy (0.7493 vs. 0.7432), confirming the benefit of compensating for age-bucket imbalance during SVM training. |

---

## Detailed Analyses

To explore the findings, metrics, and visualization charts for each parameter, refer to the individual parameter files:

1. **[Number of Body Regions Analysis](n_regions.md)** - Details the accuracy trend across region counts from 2 to 7.
2. **[Color Histogram Bin Count Analysis](color_bins.md)** - Examines the effect of finer vs. coarser color binning.
3. **[LBP Sampling Points Analysis](lbp_points.md)** - Discusses the impact of texture descriptor resolution.
4. **[Multi-scale LBP Radii Analysis](lbp_radii.md)** - Compares different multi-scale radius combinations for texture extraction.
5. **[Class Weighting Analysis](class_weight.md)** - Evaluates the benefit of balanced class weighting for imbalanced age buckets.

---

## Final Chosen Hyperparameters

| Hyperparameter | Chosen value |
| :--- | :--- |
| N_REGIONS | 4 |
| COLOR_BINS | 16 |
| LBP_POINTS | 16 |
| LBP_RADII | [1, 2, 3] |
| class_weight | balanced |

**Final Val Mean Accuracy (per age bucket), on full-scale training with the chosen hyperparameters:**

- **Age16-30:** config=('rbf', 1, 1), val_mA=0.8005
- **Age31-45:** config=('rbf', 1, 1), val_mA=0.7829
- **Age46-60:** config=('rbf', 10, 0.1), val_mA=0.7866
- **AgeAbove61:** config=('rbf', 10, 0.1), val_mA=0.9204

> **Note:** The **Age<15** bracket was disqualified due to extremely low data, leading to repeated instances of underfitting.
