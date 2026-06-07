# Brainstorm EXP-008
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Stochastic Depth (Huang et al. 2016)** (https://arxiv.org/pdf/1603.09382): Randomly drop residual blocks during training with linearly increasing drop probability. Dual benefit: regularization (like dropout for layers) + speedup (skipped blocks = less compute). Achieved 5.23% error on CIFAR-10 with ResNet-110.

## Experimental History Review

Baseline: 95.73% (EXP-007, k=4 + EMA + WD=5e-4, 55 epochs). Width scaling exhausted (k>=6 too slow). Augmentation stacking fails (EXP-006). EMA+WD combination works well.

## Candidate Ideas

### 1. k=4 + Stochastic Depth (drop=0.2) + EMA

**Summary**: Add stochastic depth to the k=4 model. During training, each residual block has a survival probability that decreases linearly from 1.0 (first block) to 0.8 (last block). Skipped blocks pass input through identity. This provides block-level dropout regularization and ~10-15% speedup.

**Reasoning**: Stochastic depth targets a different dimension from what we've tried. It regularizes the model (complementary to EMA + WD + CutMix) and speeds up training (more epochs in 300s). Well-proven on ResNets for CIFAR-10.

**Sources**: Stochastic depth paper

**Estimated Effort**: medium (modify BasicBlock forward)

**Risk Assessment**: Low. drop_rate=0.2 is conservative. Graceful failure mode.

## Chosen Idea

**Selected**: k=4 + Stochastic Depth + EMA

**Hypothesis**: Stochastic depth (max drop=0.2) will improve from 95.73% to ~95.9-96.1% through additional regularization and more training epochs.
