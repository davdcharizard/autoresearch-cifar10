# Brainstorm EXP-032
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Cutout CIFAR regularization note** (`knowledge/papers/cutout-cifar-regularization.md`)
  Cutout-style masking is a cheap CIFAR regularizer, but local cutout variants have already over-regularized this fixed-budget recipe.

- **PyTorch EMA Weight Averaging** (`knowledge/references/pytorch-ema-averaging.md`)
  Weight averaging can evaluate smoothed weights without changing the optimizer or harness, but prior local averaging attempts either cost steps or collapsed with stale snapshots.

- **Torchvision RandomCrop padding reference** (`knowledge/references/torchvision-randomcrop-padding.md`)
  Padding-mode probes are one-line augmentation tests; local evidence now favors reflection over symmetric padding.

- **Installed PyTorch loss API check**
  `uv run python` confirmed `torch.nn.functional.cross_entropy` supports `label_smoothing: float = 0.0`, so an isolated label-smoothing experiment requires no dependency or harness changes.

## Experimental History Review

- Current baseline is EXP-029 at `best_test_acc=93.58%`; under the goal's +0.10 percentage-point rule, EXP-032 must reach at least `93.68%`.
- The current anchor is `STAGE_WIDTHS = (28, 56, 112)`, reflected `RandomCrop`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 1e-4`, `LR_MILESTONES = [21000, 64000]`, FP32, channels-last, cuDNN benchmark, and `torch.compile`.
- EXP-030 and the goal-learnings file now make isolated second-LR-drop tuning a high-importance failed family.
- EXP-031 showed that symmetric crop padding is valid and throughput-neutral but weaker than reflection, so isolated padding-mode sibling tests should be deprioritized.
- Width increases beyond 28/56/112, projection shortcuts, zero-gamma initialization, lower weight decay, no-decay BN/bias, higher momentum, and aggressive smaller batch size have all underperformed the current anchor.
- The remaining plausible gaps are low-overhead regularization or late-stability changes that preserve the reflection anchor and the 21k first LR drop without spending step budget.

## Candidate Ideas

### 1. Mild Isolated Label Smoothing
**Summary**: Preserve the full reflection-padding anchor and change the training loss from `F.cross_entropy(outputs, targets)` to `F.cross_entropy(outputs, targets, label_smoothing=0.05)`.

**Reasoning**: EXP-029/031 show the current anchor can reach a strong mid-tail peak but then oscillates and finishes lower than the best epoch. Mild label smoothing is a no-overhead regularizer that may reduce overconfident updates without changing model capacity, batch size, schedule, data pipeline, or validation cadence. EXP-000 included label smoothing only inside a much stronger bundle with cutout, Nesterov, and cosine LR, so the isolated mild version has not been tested.

**Sources**: current `train.py`; EXP-000 index row; EXP-029/031 reports; installed PyTorch `F.cross_entropy` signature check.

**Estimated Effort**: low

**Risk Assessment**: Label smoothing can slow convergence or reduce peak top-1 accuracy, and other regularizers have often underperformed in this fixed budget. The failure mode should be a clean no-improvement run with preserved throughput and easy attribution.

### 2. Mild Batch Size 112 on Reflection Anchor
**Summary**: Change `BATCH_SIZE` from 128 to 112 while preserving reflection padding, 28/56/112 width, and the 21k first LR drop.

**Reasoning**: Batch size 96 was too slow and missed its planned second drop, but a milder batch-size reduction could introduce useful stochasticity while still hitting the 21k first LR drop. This tests whether the current anchor is slightly too deterministic after widening and reflection padding.

**Sources**: EXP-025 report and index row; current `train.py`; EXP-029 reflection anchor.

**Estimated Effort**: low

**Risk Assessment**: This directly neighbors a known failed approach. Even if 112 is less severe than 96, it may still reduce throughput and lower the best accuracy, making it less attractive than a no-throughput loss change.

### 3. Short-Window Late Weight Averaging
**Summary**: Preserve the reflection anchor and evaluate a small late-window averaged model built from only the most recent post-drop snapshots, avoiding long equal averaging across the whole tail.

**Reasoning**: EXP-029/031 late evaluations oscillate around the peak, so a short window could smooth transient weight noise. This is distinct from EXP-021's naive accumulating average because it would cap the window and avoid stale early snapshots.

**Sources**: `knowledge/references/pytorch-ema-averaging.md`; EXP-021 report; EXP-029/031 reports.

**Estimated Effort**: medium

**Risk Assessment**: More implementation complexity and BatchNorm-statistics risk than the other options. A careless implementation can be invalid or can spend extra validation budget, so it should wait until simpler no-overhead probes are exhausted.

## Idea Evaluation

Mild isolated label smoothing is the best next probe because it targets a real observed behavior: the reflection anchor peaks during late LR 0.01 refinement and then oscillates or finishes lower. It also has the cleanest operational profile: one loss-call change, no added dependencies, no architecture change, no schedule change, and essentially no throughput cost. Although EXP-000's combined regularization bundle failed, that result does not isolate mild label smoothing; it combined several disruptive choices that slowed useful convergence.

Batch size 112 is plausible but weaker because EXP-025 already showed that smaller batches can lose too much useful budget. A milder setting may avoid the full failure, but it still changes both optimization noise and step throughput, which makes attribution less clean. Short-window averaging addresses late oscillation more directly, but it is a higher-risk implementation and has a nearby failed predecessor in EXP-021.

EXP-032 should therefore test mild isolated label smoothing on the current reflection-padding anchor. If it fails, future brainstorms can move to either a carefully planned short-window averaging variant or a milder batch-size/stochasticity test.

## Chosen Idea
**Selected**: Mild Isolated Label Smoothing

**Why this idea**:
It is a one-line, no-dependency, no-throughput regularization test that preserves all validated anchor choices while isolating a component that has not been tested by itself.

**Hypothesis**:
Adding `label_smoothing=0.05` to the training cross-entropy will reduce overconfident late updates enough to improve `best_test_acc` from 93.58% to at least 93.68% without reducing step budget or violating any hard constraints.
