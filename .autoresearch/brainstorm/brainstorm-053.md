# Brainstorm EXP-053
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Existing knowledge base** (`knowledge/README.md`)
  Existing entries cover CIFAR regularization, cosine scheduling, width scaling, throughput tools, EMA, residual initialization, and crop padding. No new external source was needed because the next useful gap is local recipe geometry under the fixed harness rather than a missing method reference.
- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup-beyond-erm.md`)
  Mixup remains externally supported and locally unmeasured because EXP-042 crashed before final metrics, but it carries known throughput risk under this harness.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Capacity and throughput tradeoffs matter strongly for CIFAR residual networks, but local widening beyond 28/56/112 is already a recurring failure.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; EXP-053 must reach at least `94.07%` to count as an improvement.
- The current anchor is `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, label smoothing 0.05, FP32 compile, and channels-last.
- EXP-052 closes the isolated cosine-tail idea: a smooth `0.01 -> 0.002` tail reached 93.87% but did not beat the flat-tail anchor.
- High-importance failed spaces include schedule-only second LR drops, weight averaging variants, and width increases beyond 28/56/112.
- Medium recurring failures now include cosine schedule variants, residual-branch BN down-scaling, scalar LR deviations, smaller batches, cutout, and label-smoothing deviations.
- The batch-size space has only been tested downward (`96`, `112`), where useful coverage fell or the plateau weakened. A modest upward batch probe remains untested and changes the update/image-coverage tradeoff without touching the validated model, LR, regularization, or evaluation path.

## Candidate Ideas

### 1. Modest Larger Batch Size 160
**Summary**: Increase `BATCH_SIZE` from 128 to 160 while preserving the model, optimizer, LR milestones, momentum, weight decay, augmentation, label smoothing, compile, channels-last, and validation cadence.

**Reasoning**: Smaller batches failed, but that does not rule out a larger batch. Batch size 160 may improve image throughput and epoch coverage while still reaching the validated step-21000 LR drop. It tests whether the current recipe is more limited by examples processed and gradient stability than by raw update count. The change is narrow, easy to verify, and does not overlap with the now-weak schedule-only search space.

**Sources**: EXP-025 and EXP-036 smaller-batch failures; `goal-learnings/maximize-cifar10-best-test-accuracy.md` smaller-batch entry; current anchor rows EXP-038 through EXP-052.

**Estimated Effort**: low

**Risk Assessment**: Larger batches may reduce beneficial SGD noise, increase step time, or produce too few post-drop updates. If step time rises enough to miss the first LR drop, the run becomes weakly attributable or no-improvement.

### 2. Reliable Mild Mixup Retry
**Summary**: Retry `MIXUP_ALPHA = 0.1` using the now-proven foreground execution path and clean GPU availability, preserving the EXP-042 implementation intent but treating the prior result as infrastructure failure rather than a metric result.

**Reasoning**: Mixup has direct CIFAR generalization evidence and EXP-042 did not produce a final metric, so the idea remains scientifically unmeasured. It targets the late generalization plateau through data/label interpolation rather than another scalar LR, weight decay, or schedule tweak.

**Sources**: `knowledge/papers/mixup-beyond-erm.md`; `reports/exp-report-042.md`; EXP-042 execution logs showing clean startup but interrupted runs.

**Estimated Effort**: medium

**Risk Assessment**: Prior partial runs showed material throughput risk, and the run may barely reach or miss the step-21000 LR drop. It also interacts with existing label smoothing and could underfit.

### 3. Very Mild Residual Drop-Path Regularization
**Summary**: Add a tiny stochastic-depth style training-only mask to residual branches, with evaluation deterministic and parameter count unchanged.

**Reasoning**: This regularizes internal residual co-adaptation rather than images, labels, weight decay, or schedule. It is distinct from failed residual BN scaling because it does not bias initialization toward identity; it only adds a mild train-time branch-noise regularizer.

**Sources**: `train.py` `BasicBlock`; EXP-028 and EXP-051 residual initialization failures; failed isolated image-augmentation entries in goal learnings.

**Estimated Effort**: medium

**Risk Assessment**: It can reduce effective capacity or slow compile/runtime. The mechanism is less locally supported than the batch-size probe and may undertrain within the fixed budget.

## Idea Evaluation

Batch size 160 is the best next candidate because it tests a still-open local axis with a narrow implementation and clear interpretation. The existing batch failures are downward probes: they show smaller batches lose useful coverage, not that the batch-size optimum cannot be above 128. A modest upward probe can increase examples processed per epoch and potentially improve gradient stability while preserving the validated anchor recipe.

Reliable mixup is still scientifically interesting because EXP-042 crashed before producing a result, but it is operationally riskier. The prior partial logs already showed that mixup can slow the run enough to threaten the first LR drop, making a no-improvement harder to interpret. Residual drop-path is a distinct generalization idea, but it has weaker local support and risks becoming another residual-branch intervention after two residual initialization failures.

The larger-batch test has the strongest combination of feasibility, distinctness from recurring failures, and low analysis ambiguity. If it fails cleanly, it closes an obvious untested side of the batch-size bracket.

## Chosen Idea
**Selected**: Modest Larger Batch Size 160

**Why this idea**:
It is a conservative, non-schedule experiment that probes an untested direction in the update-count versus image-coverage tradeoff. It preserves the successful 28/56/112, `2e-4`, label-smoothed reflection anchor and avoids the recurring failed spaces now dominating the history.

**Hypothesis**:
Increasing batch size from 128 to 160 will preserve the step-21000 LR drop while improving image coverage and gradient stability enough to raise `best_test_acc` to at least `94.07%`.
