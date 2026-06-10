# Brainstorm EXP-026
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Still relevant background for CIFAR capacity scaling, but local evidence has made further width increases above 28/56/112 a recurring high-importance failure.

- **SGDR / cosine scheduling** (`knowledge/papers/sgdr-cosine-schedule.md`)
  Smooth scheduling remains a possible future direction, but local schedule-only bracketing and reachable second-drop tests have not cleared the current threshold.

- **PyTorch throughput tools** (`knowledge/references/pytorch-throughput-tools.md`)
  The current FP32 compile/channels-last path is a validated recipe component; new ideas should preserve it unless the experiment directly targets throughput.

No new external search was needed. The highest-signal guidance for the next move is the local trajectory: extra capacity, explicit regularization, aggressive smaller batch size, and schedule-only changes have all failed to beat the current 93.23% anchor.

## Experimental History Review

- Current baseline is EXP-016 at `best_test_acc=93.23%`; with the goal's +0.10 percentage-point rule, EXP-026 must reach at least `93.33%`.
- The current best recipe is `STAGE_WIDTHS = (28, 56, 112)`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 1e-4`, and `LR_MILESTONES = [21000, 64000]`.
- Width above 28/56/112 is now a recurring high-importance failure across proportional, minimal, and final-stage-only variants.
- Schedule-only changes around the current anchor are bounded: first-drop 20k and 23k both underperform 21k, and a reachable 36k second drop improves late refinement but not enough.
- Explicit regularization additions have generally hurt under the fixed budget: cutout variants, combined label smoothing/cosine/cutout, lower weight decay, and naive averaging all missed the threshold.
- Batch size 96 reduced throughput to 32,996 steps, missed the planned second drop, and peaked at 93.11%, so aggressive smaller-batch noise-scale changes are unattractive.
- The optimizer coefficient `MOMENTUM` remains mostly untested. EXP-010 tested Nesterov at the same coefficient, not the momentum value itself.

## Candidate Ideas

### 1. Momentum 0.95 on the Current Anchor
**Summary**: Keep the 28/56/112 anchor, batch size, LR, weight decay, and schedule unchanged, but increase classical SGD momentum from `0.9` to `0.95`.

**Reasoning**: Higher momentum may improve the optimizer's effective smoothing and late refinement without reducing throughput, increasing parameter count, adding explicit regularization, or changing validation cadence. This is distinct from EXP-010 because the failed Nesterov test changed the update formulation while keeping the coefficient at 0.9. The failure mode is likely a clean no-improvement rather than an invalid run.

**Sources**: `train.py` optimizer setup; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` EXP-010, EXP-016, EXP-025; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: The expected effect may be modest and could overshoot near the first LR drop. If it improves by less than +0.10 points, it still counts as no-improvement under the tightened threshold.

### 2. Standard CIFAR-10 Channel Std Normalization
**Summary**: Change training normalization from mean-only scaling with `std=(1, 1, 1)` to common CIFAR-10 channel standard deviations while leaving the model and optimizer path otherwise unchanged.

**Reasoning**: Per-channel standard deviation normalization is common in CIFAR recipes and could improve conditioning of the input distribution. It is a data preprocessing change inside `train.py` and does not touch the evaluation harness. Because the first convolution is followed by BatchNorm, the model may tolerate the scale shift, but the optimizer's effective first-layer step size changes.

**Sources**: `train.py` transform definition; existing CIFAR recipe knowledge; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv`.

**Estimated Effort**: low

**Risk Assessment**: The scale change may require LR retuning and could destabilize or slow early training. It is less isolated than momentum because it changes the effective optimization geometry from the input onward.

### 3. Mild Isolated Label Smoothing 0.05
**Summary**: Keep the current anchor and schedule, but use `F.cross_entropy(outputs, targets, label_smoothing=0.05)`.

**Reasoning**: Label smoothing can improve generalization and calibration without changing architecture or data augmentation. The only prior label-smoothing result was confounded with cutout, cosine LR, and Nesterov, so a weaker isolated smoothing value is not exactly exhausted.

**Sources**: `train.py` loss call; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` EXP-000; `goal-learnings/maximize-cifar10-best-test-accuracy.md` regularization failures.

**Estimated Effort**: low

**Risk Assessment**: Explicit regularization has repeatedly underperformed this fixed-budget recipe, so the prior is weaker than optimizer tuning. It may improve final loss while failing to lift peak accuracy enough for the +0.10 threshold.

## Idea Evaluation

Momentum 0.95 is the best next experiment because it targets optimizer dynamics while preserving the successful throughput path, architecture, batch size, regularization strength, and 21k first-drop schedule. It also avoids the known high-importance capacity failures and the new batch-size throughput failure. The mechanism is simple: smoother velocity updates may improve the post-drop refinement phase that currently peaks near 93.23%.

Standard channel-std normalization has plausible CIFAR recipe support, but it changes input scale and may need coupled LR retuning to be fair. That makes it a better later experiment if clean optimizer hyperparameters are exhausted. Mild label smoothing is cheap and isolated, but its mechanism overlaps with regularization changes that have repeatedly missed the threshold.

## Chosen Idea
**Selected**: Momentum 0.95 on the Current Anchor

**Why this idea**:
It is the cleanest untested optimizer-side lever remaining. It preserves the validated 28/56/112 anchor and fixed-budget throughput path while changing a single scalar that could improve convergence quality without reducing the step budget.

**Hypothesis**:
Changing `MOMENTUM` from `0.9` to `0.95` while preserving the current architecture, batch size, weight decay, and `[21000, 64000]` schedule will improve post-drop refinement and raise `best_test_acc` to at least 93.33% under the goal's +0.10 percentage-point rule.
