# Weight Decay Sensitivity Analysis

An analysis of the effects of various weight decay values on the generalization gap during Stage 1 training.

## Experiment Configuration
- **Tested Values:** `0.0`, `0.01`, `0.1`, `0.5`
- **Training Schedule:** 1 epoch of linear probing followed by 3 epochs of fine-tuning (4 epochs total).
- **Metric Measured:** The generalization gap (the difference between training accuracy and validation accuracy at the final epoch).

## Observations

- **Generalization Gap Consistency:**
  All four tested weight decay values resulted in a very similar generalization gap:
  - The gap ranged between **0.021 and 0.024** across all configurations.
  - No clear correlation or trend was observed between the magnitude of the weight decay and the size of the gap.

- **Analysis of the Training Horizon:**
  The lack of differentiation is primarily attributed to the short training horizon (4 epochs in total). The influence of weight decay on the generalization gap typically becomes more pronounced in later stages of training as the network starts to fit the training data more closely. At just 4 epochs, none of the configurations had sufficient time to manifest a significant divergence.

---

## Conclusion & Recommendation

> [!IMPORTANT]
> **Optimal Value: `0.01`**
>
> We will adopt a standard baseline weight decay of **`0.01`** for the final run. While weight decay did not make a measurable difference in this short experiment, a non-zero value is essential to prevent eventual overfitting and stabilize the weights over longer training durations.
