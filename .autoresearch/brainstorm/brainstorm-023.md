# Brainstorm EXP-023
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Widening is a plausible CIFAR mechanism in general, but local experiments now show a fixed-budget ceiling beyond 28/56/112.

- **PyTorch Throughput Tools for CNN Training** (`knowledge/references/pytorch-throughput-tools.md`)
  The current FP32 compile/channels-last path is a validated throughput component and should be preserved.

No new external search was needed. The next decision is determined by the local trajectory: the exact 28/56/112, 21k-drop anchor is still best, while capacity, adjacent schedule brackets, and averaging have recently failed.

## Experimental History Review

- Current baseline is EXP-016 at `best_test_acc=93.23%`; with the goal's +0.10 percentage-point rule, EXP-023 must reach at least `93.33%`.
- The proven anchor remains `STAGE_WIDTHS = (28, 56, 112)` with `LR_MILESTONES = [21000, 64000]`.
- Adjacent schedule bracketing is now bounded: 23k is too late, 20k is slightly too early, and 21k remains the best local first-drop milestone.
- Width increases beyond 28/56/112 are a high-priority recurring failure across proportional, minimal, and final-stage-only variants.
- Averaging has two distinct failures: per-step EMA missed the threshold through overhead, and long equal post-drop averaging collapsed as snapshots accumulated.
- Stronger regularization has generally hurt: combined cutout/label smoothing/cosine undertrained, 16x16 cutout failed, and weaker 8x8 cutout still missed the threshold. A reduction in regularization on the larger anchor remains untested.

## Candidate Ideas

### 1. Lower Weight Decay on the 28/56/112, 21k Anchor
**Summary**: Keep the current architecture, 21k first LR drop, batch size, optimizer, augmentation, and throughput settings unchanged. Reduce `WEIGHT_DECAY` from `1e-4` to `5e-5` to test whether the widened anchor is slightly over-regularized under the fixed 300s budget.

**Reasoning**: The current best is a substantially wider model than the original baseline, and multiple attempts to add regularization have failed. Reducing L2 regularization is a simple, isolated way to let the current anchor exploit its capacity more fully without changing throughput or schedule. It also avoids the now-bounded schedule and capacity paths.

**Sources**: `goal-learnings/maximize-cifar10-best-test-accuracy.md` failed regularization and schedule entries; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` EXP-000, EXP-005, EXP-009, EXP-016, EXP-022; `train.py` optimizer configuration.

**Estimated Effort**: low

**Risk Assessment**: Lower weight decay may overfit or raise final loss without improving the best epoch. If `5e-5` is too small, the result may be worse while leaving open a finer value such as `7.5e-5`.

### 2. Raise Weight Decay Slightly on the 28/56/112, 21k Anchor
**Summary**: Keep the anchor unchanged except increase `WEIGHT_DECAY` from `1e-4` to `1.5e-4`, testing whether late low-LR generalization needs slightly stronger L2 rather than less.

**Reasoning**: EXP-016 had a best/final gap, which could indicate late overfitting. A slightly larger L2 penalty is cheaper and less disruptive than cutout or label smoothing.

**Sources**: `reports/exp-report-016.md`; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv`; `train.py`.

**Estimated Effort**: low

**Risk Assessment**: This conflicts with the broader local pattern that added regularization has hurt. It may delay convergence and miss the fixed-budget peak.

### 3. Batch Size 96 on the 28/56/112, 21k Anchor
**Summary**: Reduce `BATCH_SIZE` from 128 to 96 while keeping LR and schedule initially unchanged, testing whether extra gradient noise improves generalization under the fixed time budget.

**Reasoning**: The current anchor may be in a narrow plateau; smaller batches can add regularization without explicit augmentation. It also may increase update count, though image throughput and effective epoch count could change.

**Sources**: `train.py`; experiment-index evidence that throughput and schedule calibration matter.

**Estimated Effort**: medium

**Risk Assessment**: This changes both optimization noise and throughput/step semantics, making the result harder to interpret. It may require LR or milestone retuning, so it is less isolated than weight decay.

## Idea Evaluation

Lower weight decay has the best evidence-to-risk balance. It targets an untested axis that is consistent with prior regularization failures, preserves every validated architecture and schedule component, and changes only one scalar in `train.py`.

Raising weight decay is possible but weaker: it is a clean scalar change, yet it runs against the evidence that regularization additions have slowed or hurt this benchmark. Batch size is more exploratory and could be useful later, but it confounds throughput, update count, and optimization noise.

## Chosen Idea
**Selected**: Lower Weight Decay on the 28/56/112, 21k Anchor

**Why this idea**:
It is the cleanest untested non-capacity, non-schedule perturbation after the 20k bracket failed. The current anchor should be preserved, and reducing weight decay directly tests whether the wider model is being held back by the original ResNet-20 regularization strength.

**Hypothesis**:
Changing `WEIGHT_DECAY` from `1e-4` to `5e-5` on the unchanged 28/56/112, 21k-drop anchor will improve best accuracy to at least 93.33% by reducing over-regularization without hurting throughput or validation cadence.
