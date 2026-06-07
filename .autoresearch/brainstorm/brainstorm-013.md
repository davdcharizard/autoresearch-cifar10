# Brainstorm EXP-013
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **cifar10-airbench 96%** (https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench96.py)
  Achieves 96.05% in 46 A100-seconds with a custom VGG-style ConvNet (NOT ResNet). Key architecture: 3 ConvGroup blocks (128→384→512 channels), each with 1x1 conv → MaxPool → BN → GELU → [Conv → BN → GELU → Conv → BN + residual → GELU]. Uses patch whitening, GELU activations, GlobalMaxPool, batch 1024, LR=9.0, label smoothing 0.2, 12px CutOut, 37 epochs.

- **Key architectural insight**: VGG-style networks with MaxPool + residual pairs are MORE compute-efficient than ResNets for CIFAR-10 speedruns. Fewer layers, wider channels, simpler forward pass.

## Experimental History Review

**Current best**: 95.73% (EXP-007, ResNet-20-k4 + EMA + WD=5e-4, 55 epochs)

**What's been exhausted within ResNet-20**:
- Width: k=4 optimal, k=6+ too slow (EXP-005, 012)
- Architecture tweaks: pre-activation, stochastic depth, dropout all failed
- Augmentation stacking: TrivialAugment+CutMix too aggressive
- Hyperparameter changes: batch size, LR scaling all misfire on T_max

**What hasn't been tried**: fundamentally different architecture (VGG-style custom ConvNet), different optimizer (AdamW), snapshot ensemble

**Critical realization**: ResNet-20 with 18 conv layers is deeper than necessary for CIFAR-10. The airbench architecture uses only 10 conv layers but wider — achieving more with less sequential compute. This means more epochs per 300s for the same capacity.

## Candidate Ideas

### 1. Custom SpeedNet (VGG-style, airbench-inspired)

**Summary**: Replace the entire ResNet-20 architecture with a custom VGG-style ConvNet inspired by airbench96. Design: 3 blocks, each with a channel-expanding conv followed by MaxPool, then 2 convs with residual connection. Use GELU activations, GlobalMaxPool before classifier. Channels: 128→256→512. Total ~5-6M params in only 10 conv layers. Keep EMA, CutMix, label smoothing, AMP, compile. Use cosine LR with SGD Nesterov.

The key advantage: 10 conv layers instead of 18 means ~45% less sequential compute per forward pass, allowing more epochs in 300s at comparable total model capacity. MaxPool for spatial reduction is cheaper than strided convolutions.

**Reasoning**: The airbench results conclusively prove that a VGG-style architecture outperforms ResNet for compute-budgeted CIFAR-10 training. We have 6x more compute than airbench96 (300s vs 46s), so a scaled version should comfortably exceed 96%. The architecture shift is the single biggest untried dimension.

**Sources**: cifar10-airbench repository, airbench96.py architecture

**Estimated Effort**: high (complete architecture rewrite)

**Risk Assessment**: Medium. The architecture is proven at 96% in airbench. Risk is in adaptation (no whitening layer, different optimizer, different schedule). But the fundamental design is validated. Failure mode: if implementation details are wrong, accuracy could be low but the code should still run.

### 2. k=4 ResNet + AdamW + Snapshot Ensemble

**Summary**: Keep the k=4 ResNet architecture but switch to AdamW optimizer (lr=1e-3, wd=5e-2) and use CosineAnnealingWarmRestarts (T_0=10) to train with cyclic LR. At each cycle minimum, save the EMA weights. At eval time, build a wrapper model that averages logits from all snapshots (5 snapshots from 55 epochs). This gives an ensemble effect for ~0.5-1% boost.

**Reasoning**: Snapshot ensemble is "free" accuracy — same model, same training time, multiple optima captured. AdamW may converge faster than SGD for limited epochs.

**Sources**: Snapshot Ensemble paper (Huang et al. 2017), fast.ai super-convergence blog

**Estimated Effort**: medium

**Risk Assessment**: Medium. Cyclic LR + snapshot averaging is complex. The wrapper model for eval needs to handle prepare.py's evaluate function. Risk of implementation bugs.

### 3. k=5 ResNet + AdamW (faster convergence)

**Summary**: Try k=5 ({80, 160, 320}, ~6.8M params) with AdamW optimizer instead of SGD. AdamW converges faster than SGD in limited-epoch settings. With k=5 failing at SGD (k=6 failed at 32 epochs), AdamW's faster convergence might make k=5 viable at ~45 epochs.

**Reasoning**: Previous k=6 and k=5 attempts failed because SGD needs many epochs to converge large models. AdamW adapts per-parameter learning rates and converges faster. This could unlock the capacity sweet spot between k=4 (too small) and k=6 (too big for SGD).

**Sources**: AdamW paper, fast.ai blog on AdamW convergence speed

**Estimated Effort**: low

**Risk Assessment**: Medium. AdamW might not interact well with EMA or label smoothing. k=5 epoch estimate (~45) might still be insufficient.

## Idea Evaluation

**Evidence strength**: Idea 1 has the strongest evidence — the airbench96 architecture is proven at 96.05% in a comparable compute setting. Idea 2 has theoretical support (ensemble always helps) but complex implementation. Idea 3 is speculative (AdamW might help, might not).

**Mechanism clarity**: Idea 1's mechanism is crystal clear — fewer layers = more epochs = better convergence at the same total capacity. The architecture is proven efficient for CIFAR-10. Idea 2's mechanism (averaging multiple optima) is well-understood but implementation is complex. Idea 3's mechanism (adaptive LR → faster convergence) is reasonable but unverified for this setting.

**Expected impact**: Idea 1 targets 96%+ (based on airbench96 precedent). Idea 2 targets 96.0-96.2% (0.5-1% ensemble boost). Idea 3 targets 95.8-96.0%.

**Risk profile**: Idea 1 is a bigger change but proven architecture. Idea 2 has implementation risk with the eval wrapper. Idea 3 is the safest but lowest ceiling.

**Strategy**: This is the moment for a big swing. After 8 consecutive failures with incremental changes on ResNet-20, the architecture itself is the bottleneck. The custom ConvNet is the highest-expected-value change.

## Chosen Idea

**Selected**: Custom SpeedNet (VGG-style, airbench-inspired)

**Why this idea**: The airbench results prove this architecture family reaches 96%+ efficiently. It's the most fundamentally different thing we can try, and addresses the root cause of our plateau: ResNet-20's 18-layer sequential depth wastes compute on depth that CIFAR-10 doesn't need. A wider, shallower VGG-style network is provably more efficient for this task.

**Hypothesis**: A custom VGG-style ConvNet (10 conv layers, 128→256→512 channels, GELU, MaxPool, residual pairs) with EMA + CutMix + cosine LR will achieve 96.0-96.5% best_test_acc, significantly exceeding the 95.73% ResNet baseline, by using the 300s budget more efficiently through fewer sequential layers and wider channels.
