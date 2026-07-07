# Linear Probing Duration Sensitivity Analysis

An analysis of the influence of Linear Probing Duration (Phase A epochs) on model stability and convergence during Stage 1 training.

## Experiment Configuration
- **Configurations Compared:**
  1. **0 Probing Epochs:** Backbone unfrozen from the very beginning (fully fine-tuned for 5 epochs).
  2. **3 Probing Epochs:** Backbone frozen for the first 3 epochs (linear probing), then unfrozen for 2 epochs (total 5 epochs).

## Observations

- **Loss Comparison:**
  Contrary to the initial hypothesis that omitting the linear probing phase would destabilize early training (due to large gradients from random head weights backpropagating into the pre-trained backbone), the configuration with **0 probing epochs** demonstrated significantly lower training losses at each epoch compared to the **3 probing epochs** setup.

- **Imbalance in Training Horizon:**
  It is important to note that this comparison suffers from a structural imbalance:
  - The **0 probing epochs** setup trained the backbone (unfrozen) for all **5 epochs**.
  - The **3 probing epochs** setup trained the backbone (unfrozen) for only **2 epochs** (after 3 epochs of frozen probing).
  
  Therefore, the lower training losses in the 0-probing setup are primarily because the backbone had more active epochs to adapt, rather than demonstrating that skipping probing is inherently more stable.

## Loss Comparison Visualization

Below is the step-by-step training loss comparison between the two setups:

![Linear Probing Loss Comparison](images/probing_loss_comparison.jpeg)

---

## Conclusion & Recommendation

> [!IMPORTANT]
> **Optimal Configuration: 3 Probing Epochs**
>
> Despite the lower training loss observed in the 0-probing setup, we will use **3 probing epochs** in the final run. This ensures that the newly initialized classification head is warmed up first, preventing any risk of contaminating the valuable pre-trained backbone weights in longer or more complex training schedules.
