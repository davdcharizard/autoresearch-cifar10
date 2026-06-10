# Brainstorm EXP-031
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Torchvision RandomCrop padding reference** (`knowledge/references/torchvision-randomcrop-padding.md`)
  `RandomCrop` supports sibling padding modes such as `reflect` and `symmetric`, making crop-boundary experiments one-line, no-dependency, no-harness changes.

- **Cutout CIFAR regularization note** (`knowledge/papers/cutout-cifar-regularization.md`)
  Cutout-style masking can improve CIFAR recipes, but local cutout variants have already over-regularized this fixed-budget ResNet-20 setup.

- **PyTorch EMA Weight Averaging** (`knowledge/references/pytorch-ema-averaging.md`)
  Weight averaging remains a possible late-stability lever, but prior local averaging attempts were either too costly or unstable.

## Experimental History Review

- Current baseline is EXP-029 at `best_test_acc=93.58%`; under the goal's +0.10 percentage-point rule, EXP-031 must reach at least `93.68%`.
- The current anchor is `STAGE_WIDTHS = (28, 56, 112)`, reflected `RandomCrop`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 1e-4`, `LR_MILESTONES = [21000, 64000]`, FP32, channels-last, cuDNN benchmark, and `torch.compile`.
- EXP-029 showed that crop boundary fill is a meaningful no-overhead lever: reflected padding improved the anchor to 93.58% with 43,112 steps.
- EXP-030 showed that schedule-only second LR drops are not the next productive lever: the 32k second drop fired correctly but peaked at only 93.33%.
- High-importance failed approaches now include width beyond 28/56/112 and isolated second LR drop tuning. Momentum, lower weight decay, no-decay BN/bias, projection shortcuts, zero-gamma initialization, and cutout-style masking also have weak local evidence.
- The remaining high-signal gap is a direct sibling test inside the newly validated crop-boundary family, while preserving every other anchor setting.

## Candidate Ideas

### 1. Symmetric Padding for RandomCrop
**Summary**: Replace `padding_mode="reflect"` with `padding_mode="symmetric"` in the CIFAR training `RandomCrop`, leaving crop size, padding amount, horizontal flip, normalization, model width, optimizer, schedule, batch size, compile path, and validation cadence unchanged.

**Reasoning**: EXP-029 proved that replacing constant zero crop padding with reflected boundaries can produce a large no-overhead gain. Symmetric padding is the closest untested sibling: it also avoids artificial zero borders but differs in how edge pixels are mirrored. Because the change is isolated and throughput-neutral, even a small boundary-statistics advantage can plausibly clear the +0.10 threshold.

**Sources**: `knowledge/references/torchvision-randomcrop-padding.md`; EXP-029 report; EXP-030 report; current `train.py` transform.

**Estimated Effort**: low

**Risk Assessment**: Symmetric padding may be equivalent to or slightly worse than reflection, especially because reflection is already validated. The failure mode is a valid no-improvement run with preserved throughput and clean attribution.

### 2. Mild Batch Size 112 on Reflection Anchor
**Summary**: Change `BATCH_SIZE` from 128 to 112 while preserving reflection padding, 28/56/112 width, and the 21k first LR drop. Keep all other settings fixed unless planning confirms a minor schedule reachability adjustment is required.

**Reasoning**: Batch size 96 was too slow, but a smaller batch can sometimes add useful stochasticity and more update granularity. A milder reduction to 112 may preserve enough throughput to keep the 21k first drop reachable while slightly improving generalization.

**Sources**: EXP-025 report; current `train.py`; goal-learnings failed approach for batch size 96.

**Estimated Effort**: low

**Risk Assessment**: Throughput loss is still likely and may outweigh any stochastic benefit. If the run completes materially fewer steps, the result will mostly retest the known smaller-batch failure mode.

### 3. Short-Window Late Weight Averaging
**Summary**: Preserve the reflection anchor and evaluate a short-window averaged model only over a small number of late post-drop snapshots, avoiding long equal averaging across the full tail.

**Reasoning**: EXP-029 and EXP-030 both show late validation oscillation, so smoothing the very end of training remains conceptually plausible. A short window could avoid EXP-021's collapse from accumulating too many stale snapshots and avoid per-step EMA overhead from EXP-004.

**Sources**: `knowledge/references/pytorch-ema-averaging.md`; EXP-004; EXP-021; EXP-029/EXP-030 late evaluation trajectories.

**Estimated Effort**: medium

**Risk Assessment**: This adds implementation complexity and can mishandle BatchNorm statistics or reduce training throughput. It is more likely than symmetric padding to produce an invalid or misleading result if not planned carefully.

## Idea Evaluation

Symmetric padding has the strongest immediate evidence because it is a direct sibling of the only recent successful non-capacity change. It targets the crop-boundary mechanism validated by EXP-029, avoids the high-importance schedule-only second-drop failure from EXP-030, and does not change throughput, parameter count, optimizer dynamics, or validation cadence. Its expected impact is uncertain, but its causal mechanism and failure mode are clean.

Batch size 112 is plausible but less attractive because the known batch-size direction already failed at 96 by losing too much step budget. A milder variant may be safer, yet it still changes both optimization noise and training throughput, making attribution weaker than a one-line padding-mode probe. Short-window averaging addresses late oscillation but carries the most operational risk because prior averaging attempts failed in distinct ways.

EXP-031 should therefore test symmetric padding first. If it fails, the crop-boundary family will have a stronger local comparison, and future brainstorms can move to different no-overhead augmentation or carefully constrained averaging variants.

## Chosen Idea
**Selected**: Symmetric Padding for RandomCrop

**Why this idea**:
It is the closest untested sibling of EXP-029's successful reflection-padding intervention and keeps the entire current anchor fixed except for crop boundary fill semantics.

**Hypothesis**:
Changing `RandomCrop` from reflected to symmetric padding will preserve the no-zero-border augmentation benefit while altering edge statistics enough to improve `best_test_acc` from 93.58% to at least 93.68%.
