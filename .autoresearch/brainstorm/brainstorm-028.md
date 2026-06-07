# Brainstorm EXP-028
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Experimental History Review
- 29 experiments, 96.39% baseline, 12 consecutive failures
- LR=0.1 has NEVER been tested at any other value (except EXP-010 which changed batch size AND LR simultaneously)
- Every other hyperparameter exhausted: WD (5e-4 optimal), CutMix alpha (1.0 optimal), CutMix prob (0.5 optimal), label smoothing (0.1 optimal), EMA decay (0.999 optimal)

## Chosen Idea
**Selected**: Peak LR 0.15

**Hypothesis**: A 50% higher peak LR (0.15 vs 0.1) will enable faster learning during the cosine schedule, reaching a better minimum in the same ~54 epochs. The cosine schedule still decays to near-zero, but the higher peak allows more aggressive exploration of the loss landscape in early-to-mid training.
