# Brainstorm EXP-006
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **TrivialAugment (Muller & Hutter 2021)** (https://openaccess.thecvf.com/content/ICCV2021/papers/Muller_TrivialAugment_Tuning-Free_Yet_State-of-the-Art_Data_Augmentation_ICCV_2021_paper.pdf)
  Zero-hyperparameter augmentation — applies one random augmentation per image from a predefined set. Among top performers on CIFAR-10 in all benchmarks. Available in torchvision as `TrivialAugmentWide`.

- **ResNet Strikes Back (Wightman et al. 2021)** (https://arxiv.org/pdf/2110.00476)
  Improved training procedure for ResNets. Combines modern augmentation, longer training, and better regularization to significantly improve standard ResNet accuracy.

## Experimental History Review

**Trajectory** (5 experiments):
- BASE: 91.81% → EXP-000: 92.10% → EXP-001: 94.03% → EXP-003: 94.80% → **EXP-004: 95.25%** (current best)
- EXP-002: 94.09% (no-improvement, broken T_max)
- EXP-005: 94.52% (no-improvement, k=6 too wide, only 32 epochs)

**Key finding**: k=4 is the capacity sweet spot (4.3M params, 58 epochs). k=6 overshoots. Must improve from a different dimension.

**Untried**: TrivialAugment, Mixup, stochastic depth, pre-activation at k=4, higher LR

## Candidate Ideas

### 1. k=4 + TrivialAugment + CutMix

**Summary**: Keep k=4 architecture. Add TrivialAugmentWide to the transform pipeline (before ToTensor) — applies diverse random augmentations per image. Keep CutMix in the training loop. The two are complementary: TrivialAugment diversifies individual images, CutMix diversifies at batch level.

**Reasoning**: TrivialAugment is state-of-the-art on CIFAR-10 with zero hyperparameters. Adding it on top of CutMix provides richer regularization without any compute cost to epoch count.

**Sources**: TrivialAugment paper, ResNet Strikes Back

**Estimated Effort**: low (one line added to transforms)

**Risk Assessment**: Very low. TrivialAugment is well-validated and adds negligible compute. Worst case: no improvement.

### 2. k=4 + Pre-activation Blocks

**Summary**: Test pre-activation at k=4 (the proven scale). EXP-005 conflated pre-activation with k=6 (too wide), so the architectural effect is unknown at k=4.

**Reasoning**: Pre-activation improves gradient flow. At k=4 with 58 epochs, the model has enough training time to benefit.

**Sources**: He et al. 2016, EXP-005 (inconclusive due to k=6 confound)

**Estimated Effort**: medium (restructure BasicBlock)

**Risk Assessment**: Medium. Pre-activation changes training dynamics; may need LR tuning.

### 3. k=4 + Stochastic Depth + Mixup

**Summary**: Add stochastic depth (randomly skip blocks with linearly increasing probability) and Mixup (blend pairs of images and labels). Both regularize the wide model.

**Reasoning**: Stochastic depth speeds up training (skipped blocks = less compute) and regularizes. Mixup provides different regularization signal from CutMix.

**Sources**: Stochastic depth paper, Mixup paper

**Estimated Effort**: medium

**Risk Assessment**: Medium. Two new techniques with hyperparameters to tune.

## Idea Evaluation

Idea 1 is the lowest risk with proven effectiveness. Idea 2 is a single architectural change but harder to implement. Idea 3 adds two new techniques simultaneously.

TrivialAugment is the simplest change with the strongest evidence. It composes naturally with CutMix and requires zero tuning.

## Chosen Idea

**Selected**: k=4 + TrivialAugment + CutMix

**Why this idea**: Simplest change with strongest evidence. TrivialAugment is state-of-the-art, zero hyperparameters, negligible compute cost. Composes with existing CutMix.

**Hypothesis**: Adding TrivialAugmentWide to the k=4 pipeline will improve best_test_acc from 95.25% to approximately 95.5-96.0% through richer augmentation diversity.
