# Brainstorm EXP-005
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **SE blocks on wide networks** (GitHub issue + SE paper): SE modules work better on thin networks than wide networks on CIFAR-10. At k=4, SE improvement would be marginal (~0.2%).
- **Pre-activation ResNet (He et al. 2016)**: Swap block ordering from Conv→BN→ReLU to BN→ReLU→Conv. Improves gradient flow through identity shortcuts. Consistently improves deeper/wider ResNets by 0.3-0.5% on CIFAR-10.

## Experimental History Review

**Width scaling trajectory**:
- k=1→k=2: +1.93%, k=2→k=3: +0.77%, k=3→k=4: +0.45%
- Diminishing returns clear. Current baseline: 95.25% at k=4 (4.3M params, 58 epochs)
- VRAM: 538MB (< 1% of 98GB available)

**What hasn't been tried**: pre-activation blocks, k>4, deeper models, advanced augmentation (TrivialAugment, RandAugment), stochastic depth

## Candidate Ideas

### 1. k=8 + Pre-activation Blocks

**Summary**: Jump to k=8 ({128,256,512}, ~17.3M params) with pre-activation block ordering (BN→ReLU→Conv instead of Conv→BN→ReLU). Pre-activation improves gradient flow, especially important for wider models. Estimated ~25-30 epochs with AMP+compile. T_max=20-25.

**Reasoning**: Width continues to show gains. k=8 is a 4x capacity increase from k=4. Pre-activation is essentially free in compute and improves convergence quality. Combined, these target both the capacity ceiling and training dynamics.

**Sources**: WideResNet paper, He et al. 2016 pre-activation paper

**Estimated Effort**: medium (pre-activation requires restructuring BasicBlock)

**Risk Assessment**: Medium-high. ~25-30 epochs is tight. Pre-activation changes gradient dynamics which could interact with existing hyperparameters. Multiple changes at once.

### 2. k=6 + Pre-activation Blocks

**Summary**: Moderate width increase to k=6 ({96,192,384}, ~9.7M params) with pre-activation blocks. Estimated ~40 epochs. More conservative than k=8.

**Reasoning**: k=6 is a moderate step from k=4, giving more training time (~40 epochs) for the pre-activation architecture to converge. This avoids the epoch-count risk of k=8 while still providing meaningful capacity increase.

**Sources**: Same as above

**Estimated Effort**: medium

**Risk Assessment**: Lower than k=8 due to more training epochs. Pre-activation is the main architectural change.

### 3. k=4 + Pre-activation + Stochastic Depth

**Summary**: Keep k=4 width, add pre-activation blocks and stochastic depth (randomly drop residual blocks during training with linearly increasing probability). Stochastic depth regularizes deep/wide networks and reduces effective training time per step.

**Reasoning**: Instead of making the model bigger, make it train better. Stochastic depth acts as both regularizer and speed-up (skipped blocks = faster steps). At k=4 this could improve both accuracy and allow more epochs. Pre-activation improves gradient flow.

**Sources**: Huang et al. 2016 (stochastic depth), He et al. 2016

**Estimated Effort**: medium

**Risk Assessment**: Medium. Stochastic depth survival rate needs tuning. Combined with pre-activation, two architectural changes.

## Idea Evaluation

**k=8 (Idea 1)** has highest ceiling but highest risk (few epochs). **k=6 (Idea 2)** balances capacity and training time. **k=4 + stochastic depth (Idea 3)** is different in kind — regularization rather than capacity.

Since width scaling is the proven lever and the trend is still positive, continuing to push width is the highest-probability path. k=6 gives ~40 epochs (safe for convergence) with 2.25x more capacity than k=4. Pre-activation is a proven architectural improvement.

## Chosen Idea

**Selected**: k=6 + Pre-activation Blocks

**Why this idea**: Balances meaningful capacity increase (k=6, ~9.7M params) with sufficient training epochs (~40) for convergence. Pre-activation is a well-evidenced architectural improvement that composes with width. Lower risk than k=8.

**Hypothesis**: k=6 with pre-activation blocks will improve best_test_acc from 95.25% to approximately 95.5-96.0%, combining capacity increase with better gradient flow.
