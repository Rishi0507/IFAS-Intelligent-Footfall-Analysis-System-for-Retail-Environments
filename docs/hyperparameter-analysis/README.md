# Hyperparameter Sensitivity Analysis (Stage 1 Training & Pure SVM Baseline for Age Classification)

This folder documents the following:
1. Hyperparameter sensitivity experiments conducted during the Stage 1 base attribute training of the **ViT-B/16** model on the **PETA dataset**.
2. Hyperparameter sensitivity experiments conducted for the **Pure SVM** age classification baseline, which replicates the original PETA benchmark methodology (region-based color + texture histograms with one intersection/RBF-kernel SVM per attribute).

The purpose of these experiments was to determine the configuration that offers optimal classification accuracy, smooth convergence, and generalization stability, while preventing overfitting collapse. Secondly, determine the feature-extraction and training configuration that offers the highest screening accuracy (average mA) on a 30% subsample of the training and validation data, before locking in the final configuration used for full-scale training.

## Summary of Hyperparameter Configurations

Below is a summary of the hyperparameters evaluated, the values tested, and the selected optimal configuration:

| Hyperparameter | Tested Values | Optimal Value | Conclusion & Impact |
| :--- | :--- | :--- | :--- |
| **[Backbone Learning Rate](learning_rate.md)** | `1e-4`, `1e-5`, `1e-7` | **`1e-5`** | Avoids underfitting seen at `1e-7` and offers a smoother, safer convergence curve than `1e-4` on longer runs. |
| **[Linear Probing Duration](linear_probing.md)** | `0` epochs, `3` epochs | **`3` epochs** | Warms up the random classification head first, preventing potential contamination of pre-trained backbone weights. |
| **[Weight Decay](weight_decay.md)** | `0.0`, `0.01`, `0.1`, `0.5` | **`0.01`** | Generalization gaps were similar at 4 epochs, but a standard non-zero weight decay is recommended for long-term weight stability. |
| **[Warmup Ratio](warmup_ratio.md)** | `0.0`, `0.1`, `0.4` | **`0.1`** | A 10% warmup duration successfully stabilizes early training without limiting progression in later steps. |

---

| Hyperparameter | Tested Values | Optimal Value | Conclusion & Impact |
| :--- | :--- | :--- | :--- |
| **[Number of Body Regions](n_regions.md)** | `2, 3, 4, 5, 6, 7` | **`4`** | Accuracy peaks at 4 regions (0.7457); finer slicing beyond this point fragments regions into strips too thin for reliable histograms, adding noise instead of detail. |
| **[Color Histogram Bin Count](color_bins.md)** | `8, 16, 24, 32` | **`16`** | 16 bins achieves the highest accuracy (0.7457); finer binning beyond 16 sparsifies histograms as region size and bin count both shrink relative to it. |
| **[LBP Sampling Points](lbp_points.md)** | `8, 16, 24` | **`16`** | 16 points achieves the highest accuracy (0.7493); 8 points misses discriminative detail, while 24 points shows diminishing returns from added noise. |
| **[Multi-scale LBP Radii](lbp_radii.md)** | `[1], [1,2], [1,2,3], [1,2,3,4], [2,3]` | **`[1, 2, 3]`** | All configurations score very similarly, indicating low sensitivity to the specific radii combination; `[1,2,3]` scores marginally highest. |
| **[Class Weighting](class_weight.md)** | `none`, `balanced` | **`balanced`** | Balanced weighting achieves a clearly higher accuracy (0.7493 vs. 0.7432), confirming the benefit of compensating for age-bucket imbalance during SVM training. |

## Detailed Analyses

To explore the findings, metrics, and visualization charts for each parameter, refer to the individual parameter files:

1. **[Backbone Learning Rate Analysis](learning_rate.md)** - Details the validation accuracy convergence charts for learning rates.
2. **[Linear Probing Duration Analysis](linear_probing.md)** - Explains the training loss comparison between skipping and incorporating probing epochs.
3. **[Weight Decay Analysis](weight_decay.md)** - Discusses the generalization gap findings over short training horizons.
4. **[Warmup Ratio Analysis](warmup_ratio.md)** - Examines the step-by-step training loss trends for different warmup schedules.

---

1. **[Number of Body Regions Analysis](n_regions.md)** - Details the accuracy trend across region counts from 2 to 7.
2. **[Color Histogram Bin Count Analysis](color_bins.md)** - Examines the effect of finer vs. coarser color binning.
3. **[LBP Sampling Points Analysis](lbp_points.md)** - Discusses the impact of texture descriptor resolution.
4. **[Multi-scale LBP Radii Analysis](lbp_radii.md)** - Compares different multi-scale radius combinations for texture extraction.
5. **[Class Weighting Analysis](class_weight.md)** - Evaluates the benefit of balanced class weighting for imbalanced age buckets.
