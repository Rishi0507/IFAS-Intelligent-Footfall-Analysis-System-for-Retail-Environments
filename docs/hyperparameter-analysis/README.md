# Hyperparameter Sensitivity Analysis (Stage 1 Training)

This folder documents the hyperparameter sensitivity experiments conducted during the Stage 1 base attribute training of the **ViT-B/16** model on the **PETA dataset**.

The purpose of these experiments was to determine the configuration that offers optimal classification accuracy, smooth convergence, and generalization stability, while preventing overfitting collapse.

## Summary of Hyperparameter Configurations

Below is a summary of the hyperparameters evaluated, the values tested, and the selected optimal configuration:

| Hyperparameter | Tested Values | Optimal Value | Conclusion & Impact |
| :--- | :--- | :--- | :--- |
| **[Backbone Learning Rate](learning_rate.md)** | `1e-4`, `1e-5`, `1e-7` | **`1e-5`** | Avoids underfitting seen at `1e-7` and offers a smoother, safer convergence curve than `1e-4` on longer runs. |
| **[Linear Probing Duration](linear_probing.md)** | `0` epochs, `3` epochs | **`3` epochs** | Warms up the random classification head first, preventing potential contamination of pre-trained backbone weights. |
| **[Weight Decay](weight_decay.md)** | `0.0`, `0.01`, `0.1`, `0.5` | **`0.01`** | Generalization gaps were similar at 4 epochs, but a standard non-zero weight decay is recommended for long-term weight stability. |
| **[Warmup Ratio](warmup_ratio.md)** | `0.0`, `0.1`, `0.4` | **`0.1`** | A 10% warmup duration successfully stabilizes early training without limiting progression in later steps. |

---

## Detailed Analyses

To explore the findings, metrics, and visualization charts for each parameter, refer to the individual parameter files:

1. **[Backbone Learning Rate Analysis](learning_rate.md)** - Details the validation accuracy convergence charts for learning rates.
2. **[Linear Probing Duration Analysis](linear_probing.md)** - Explains the training loss comparison between skipping and incorporating probing epochs.
3. **[Weight Decay Analysis](weight_decay.md)** - Discusses the generalization gap findings over short training horizons.
4. **[Warmup Ratio Analysis](warmup_ratio.md)** - Examines the step-by-step training loss trends for different warmup schedules.
