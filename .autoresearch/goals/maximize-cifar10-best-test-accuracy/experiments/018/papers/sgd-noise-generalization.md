# On the Generalization Benefit of Noise in Stochastic Gradient Descent
- **Authors**: Samuel Smith, Erich Elsen, Soham De
- **Venue**: ICML 2020
- **URL**: https://proceedings.mlr.press/v119/smith20a.html

## Key Contributions
- Uses controlled hyperparameter sweeps to show small or moderately large batches can generalize better than very large batches.
- Finds that lower training loss from large batches does not guarantee better test accuracy, even at equal iteration counts.
- Connects the benefit to stochastic-gradient noise and studies learning-rate schedule interaction.

## Relevance

EXP-013 already found that batch 256 offered only 1.189x image throughput, while this paper warns that reduced gradient noise can independently damage generalization. This strengthens the case against revisiting large batches merely for exposure and supports candidates that improve late solution quality without reducing the accepted batch-128 noise scale.

## Key Techniques
- Compare batch regimes at controlled iteration counts and tuned learning rates.
- Judge training loss and test generalization separately.
- Treat gradient-noise scale as a mechanism, not only batch throughput.

