# Brainstorm EXP-007
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **EMA in modern training** (ResNet Strikes Back, timm library): EMA maintains a running exponential average of model weights during training, using the averaged weights for evaluation. Standard in modern training recipes. Typical decay=0.999 or 0.9999. Adds near-zero compute cost and typically improves accuracy by 0.1-0.5%.

- **Weight decay for WideResNets**: The WideResNet paper uses weight_decay=5e-4 (vs our current 1e-4). Higher weight decay provides stronger regularization for larger models.

## Experimental History Review

**Current trajectory**: 95.25% (k=4, EXP-004) is the baseline. Two consecutive failures:
- EXP-005: k=6 too wide (32 epochs insufficient)
- EXP-006: TrivialAugment + CutMix too aggressive

**k=4 sweet spot confirmed**: 4.3M params, 58 epochs, 95.25%. Need small, targeted improvements.

**Failed approaches**: excessive width (k>=6), excessive augmentation (TrivialAugment + CutMix)

## Candidate Ideas

### 1. k=4 + EMA + Weight Decay 5e-4

**Summary**: Keep k=4 architecture exactly as EXP-004. Add EMA (decay=0.999) on model weights — maintain a shadow copy updated each step, evaluate with the shadow model. Increase weight decay from 1e-4 to 5e-4 (WideResNet paper standard for larger models).

**Reasoning**: EMA smooths training noise — particularly helpful at the end of cosine schedule where LR is low and gradients are noisy. Weight decay 5e-4 is the standard for WideResNets and provides better regularization for 4.3M params. Both changes are safe and additive.

**Sources**: ResNet Strikes Back (timm), WideResNet paper

**Estimated Effort**: low

**Risk Assessment**: Very low. EMA is always >= baseline (worst case identical). Weight decay increase is modest and well-precedented.

### 2. k=4 + Stochastic Depth (drop_rate=0.1)

**Summary**: Randomly drop residual blocks during training with linearly increasing probability (max 0.1). Provides regularization and slightly speeds up training.

**Sources**: Huang et al. 2016 stochastic depth paper

**Estimated Effort**: medium

**Risk Assessment**: Medium. drop_rate needs tuning.

## Idea Evaluation

EMA + weight decay (Idea 1) is the safest bet after two consecutive failures. Both techniques are additive, well-understood, and commonly used in production training recipes. Stochastic depth (Idea 2) adds implementation complexity and a new hyperparameter.

## Chosen Idea

**Selected**: k=4 + EMA + Weight Decay 5e-4

**Why this idea**: After two failures from aggressive changes, this is a conservative experiment targeting well-understood, nearly risk-free improvements. EMA is essentially free accuracy, and weight decay 5e-4 is the WideResNet standard.

**Hypothesis**: EMA (decay=0.999) + weight_decay=5e-4 will improve best_test_acc from 95.25% to approximately 95.4-95.8%, primarily from EMA smoothing late-training noise and better regularization.
