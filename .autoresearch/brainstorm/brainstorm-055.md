# Brainstorm EXP-055
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup-beyond-erm.md`; source: https://arxiv.org/abs/1710.09412)
  Mixup trains on convex combinations of inputs and labels, providing a CIFAR-relevant regularizer that changes only the training loop and preserves the evaluation graph. The local knowledge entry explicitly notes that EXP-042 crashed before final metrics, so mixup remains unproven rather than disproven.
- **Existing knowledge base** (`knowledge/README.md`)
  Prior entries cover cutout, RandAugment, cosine schedules, width scaling, EMA, residual initialization, stochastic depth, and crop padding. Most single-axis local brackets around the current anchor are now negative, making the unresolved mixup path comparatively valuable despite its execution risk.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; EXP-055 must reach at least `94.07%` to count as an improvement under the goal's +0.10 percentage-point rule.
- The active anchor is `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, label smoothing 0.05, FP32 compile, channels-last, and once-per-epoch validation.
- EXP-042 tried `MIXUP_ALPHA=0.1` but both attempts ended before a final `best_test_acc`, so it is indexed as `crash` with metric `NaN`. The report states this does not provide evidence for or against mixup accuracy because neither attempt reached the post-drop regime.
- Foreground attached runs after EXP-042 have completed repeatedly, including EXP-052, EXP-053, and EXP-054, so the run-control failure mode is now better understood: avoid detached/nohup launch forms and run in an attached foreground session.
- High-importance failed spaces include schedule-only second drops, weight averaging variants, isolated batch-size deviations, and width increases beyond 28/56/112. Medium recurring failures include cosine schedules, residual-branch BN down-scaling, scalar LR deviations, cutout, and label-smoothing deviations.
- The most important open gap is a completed result for label-space/input interpolation regularization. It is distinct from failed photometric or erased-patch augmentation because it changes the supervised target geometry rather than only input appearance.

## Candidate Ideas

### 1. Reliable Mild Mixup Retry
**Summary**: Retry the EXP-042 mild mixup intervention with `MIXUP_ALPHA = 0.1`, using the now-proven foreground attached execution path and preserving all current anchor settings. Implement per-batch on-device permutation, mixed inputs, and weighted two-target cross entropy while keeping `label_smoothing=0.05` and evaluation unchanged.

**Reasoning**: Mixup has direct CIFAR regularization support and remains scientifically unresolved locally because EXP-042 crashed before a final metric. It targets a different generalization mechanism than the heavily explored scalar LR, weight decay, schedule, batch-size, width, residual-init, and augmentation-only spaces. A direct retry answers whether the original mild mixup idea can clear 94.07% when the run actually survives.

**Sources**: `knowledge/papers/mixup-beyond-erm.md`; `reports/exp-report-042.md`; completed foreground runs EXP-052, EXP-053, EXP-054; goal-learnings failed-approach sections through EXP-054.

**Estimated Effort**: medium

**Risk Assessment**: Prior partial logs suggested mixup step timing may slow after the early phase, so the run could miss or barely reach the 21k LR drop. If it completes but misses the first drop, attribution will be weak. If it reaches the drop and remains below 94.07%, mixup should be treated as a clean no-improvement rather than an infrastructure failure.

### 2. Lower-Strength Mixup Alpha 0.05
**Summary**: Add the same mixup implementation but reduce `MIXUP_ALPHA` from 0.1 to 0.05 to make interpolation less aggressive while preserving the training/evaluation structure.

**Reasoning**: The anchor already includes label smoothing and stronger weight decay, so alpha 0.1 might over-regularize even if it completes. A weaker alpha could add target smoothing through interpolation without pushing samples too far off-manifold.

**Sources**: `knowledge/papers/mixup-beyond-erm.md`; label-smoothing pattern EXP-032; decay anchor EXP-038; EXP-042 crash report.

**Estimated Effort**: medium

**Risk Assessment**: Alpha 0.05 does not materially reduce the tensor overhead that threatened EXP-042, and it weakens the scientific continuity with the prior crash. If it underperforms, alpha 0.1 would remain unmeasured.

### 3. Final-Layer Dropout 0.05
**Summary**: Add a small training-only dropout just before the final linear classifier, with probability 0.05, while preserving architecture width, schedule, optimizer, augmentation, and evaluation behavior.

**Reasoning**: This is a low-overhead regularizer that targets classifier co-adaptation rather than residual branches or image augmentations. It should not affect the LR milestone or parameter count and is simpler than DropBlock.

**Sources**: EXP-054 stochastic-depth report; failed residual regularization EXP-028/EXP-051/EXP-054; failed augmentation entries EXP-044/EXP-050.

**Estimated Effort**: low

**Risk Assessment**: The model is shallow and already regularized by label smoothing plus stronger weight decay; final-layer dropout may simply underfit. It also has weaker direct evidence than mixup and overlaps broadly with the failed isolated-regularizer pattern.

## Idea Evaluation

Reliable Mild Mixup Retry has the strongest evidence and the cleanest reason to run now. The external paper support is direct for CIFAR-style classification, and the local negative record is a crash rather than a measured no-improvement. Since later foreground attached runs complete reliably, the previous failure mode is less likely to recur if the plan uses a monitored foreground session.

Lower-strength mixup is plausible but less decisive. It may reduce over-regularization, yet the main unresolved question is not whether alpha 0.05 beats alpha 0.1; it is whether mixup can produce any completed, post-drop result in this fixed-budget harness. Running alpha 0.1 first preserves continuity with EXP-042 and gives a sharper scientific answer.

Final-layer dropout is easy and low-overhead, but its evidence base is weaker. Recent failures show isolated regularization often harms this anchor, especially residual or augmentation-like regularizers. It is better held as a backup if mixup proves infeasible or cleanly negative.

The lead candidate should therefore be the direct reliable mild mixup retry. The plan must treat the LR drop as a key verification point and classify any completed sub-threshold run as no-improvement rather than retrying indefinitely.

## Chosen Idea
**Selected**: Reliable Mild Mixup Retry

**Why this idea**:
It is the most important unresolved mechanism left in the local trajectory. Unlike many already-closed hyperparameter brackets, EXP-042 did not produce a metric, and mixup tests input/label interpolation rather than another schedule, batch, width, decay, or residual regularization tweak. Foreground execution is now reliable enough to make the retry interpretable.

**Hypothesis**:
A completed foreground run with `MIXUP_ALPHA=0.1` will preserve the step-21000 LR drop and may improve late generalization enough to reach at least `94.07%`. If it reaches the LR drop but remains below threshold, mild mixup should be considered a valid no-improvement for this anchor.
