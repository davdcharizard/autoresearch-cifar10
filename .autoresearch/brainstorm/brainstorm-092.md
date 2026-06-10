# Brainstorm EXP-092
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Recent project reports** (`reports/exp-report-085.md`, `reports/exp-report-088.md`, `reports/exp-report-090.md`, `reports/exp-report-091.md`)
  The current best is the padding-3 / flip-p=0.4 spatial anchor. Static CutMix probability and alpha brackets are now closed, and stronger weight decay failed under this anchor.

No new external search was needed. EXP-092 is a local scalar regularization diagnostic driven by project history.

## Experimental History Review

- Current best is EXP-085 at `best_test_acc=94.51%` from commit `83d4e94`; the +0.10pp noise guard requires EXP-092 to reach at least 94.61%.
- The active anchor is reflection crop padding 3, `RandomHorizontalFlip(p=0.4)`, unit-std normalization, static CutMix alpha 1.0 / probability 0.5 / label smoothing 0.05, clean label smoothing 0.05, ResNet-20 `(28,56,112)`, `WEIGHT_DECAY=2e-4`, and first LR drop at step 21000.
- Isolated spatial tuning is closed: crop padding 2, flip p=0.425, and flip p=0.375 all failed.
- Static CutMix probability and alpha moves away from p=0.5 / alpha 1.0 are now high-importance failed families.
- EXP-088 showed `WEIGHT_DECAY=2.5e-4` over-regularizes the spatial anchor. EXP-041 showed `1.5e-4` was too weak on an older anchor, but a smaller lower-side move to `1.75e-4` remains untested on the current anchor.

## Candidate Ideas

### 1. Fine Lower Weight Decay 1.75e-4 on the Spatial Anchor
**Summary**: Keep the active spatial and CutMix anchor and reduce `WEIGHT_DECAY` from `2e-4` to `1.75e-4`. Preserve crop, flip, CutMix, label smoothing, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last, fixed budget, and validation cadence.

**Reasoning**: Stronger decay failed recently, and older lower decay was too weak before the current spatial anchor existed. A fine lower-side bracket tests whether the de-regularized spatial recipe now wants marginally less shrinkage without reopening closed CutMix or spatial families.

**Sources**: `reports/exp-report-041.md`; `reports/exp-report-085.md`; `reports/exp-report-088.md`; `reports/exp-report-091.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: Scalar decay evidence favors `2e-4`, so failure is likely. The experiment is safe, one-line, and should fail cleanly if the anchor is already optimal.

### 2. Higher BatchNorm Momentum 0.2 on the Spatial Anchor
**Summary**: Raise BatchNorm momentum from the default 0.1 to 0.2 in all BatchNorm2d layers, preserving the current augmentation, CutMix, optimizer, schedule, batch size, seed, compile/channels-last, fixed budget, and validation cadence.

**Reasoning**: Lower BN momentum failed, but faster running-stat adaptation is distinct and could better track mixed clean/CutMix batches. It is a non-CutMix, non-spatial lever with minimal parameter overhead.

**Sources**: `reports/exp-report-048.md`; `reports/exp-report-064.md`; `reports/exp-report-085.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: Evidence is weaker than the weight-decay bracket, and implementation is more invasive because it touches module construction across multiple BatchNorm layers.

### 3. Classical Momentum 0.85 on the Spatial Anchor
**Summary**: Reduce SGD momentum from 0.9 to 0.85 while preserving LR, weight decay, architecture, augmentation, CutMix, batch size, seed, compile/channels-last, fixed budget, and validation cadence.

**Reasoning**: Momentum 0.95 and Nesterov failed, but a lower classical momentum has not been tested under the final anchor. It could reduce overshoot after the first LR drop without changing schedule milestones.

**Sources**: `reports/exp-report-010.md`; `reports/exp-report-026.md`; `reports/exp-report-085.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: Optimizer-transient evidence is weak and LR startup deviations are a high-importance failed family. This is more speculative than a weight-decay closure test.

## Idea Evaluation

The fine lower weight-decay bracket is the most defensible next experiment. It is a one-line scalar diagnostic, does not retry high-importance CutMix/spatial/schedule families, and directly closes the remaining side around the current `2e-4` regularization anchor.

BatchNorm momentum 0.2 is more distinct, but the mechanism is weaker and changes model construction rather than a top-level optimizer scalar. Momentum 0.85 is also distinct, but optimizer transient tuning has sparse support and risks drifting toward the high-importance LR-startup failure cluster.

## Chosen Idea
**Selected**: Fine Lower Weight Decay 1.75e-4 on the Spatial Anchor

**Why this idea**:
It is the safest remaining scalar regularization diagnostic after spatial and CutMix scalar brackets closed. It tests the only unclosed near side around `WEIGHT_DECAY=2e-4` on the current best anchor.

**Hypothesis**:
If the padding-3 / flip-p=0.4 anchor is slightly over-regularized by `WEIGHT_DECAY=2e-4`, reducing weight decay to `1.75e-4` will raise `best_test_acc` from 94.51% to at least 94.61%.
