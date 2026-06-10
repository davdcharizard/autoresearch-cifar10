# Brainstorm EXP-056
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Existing knowledge base** (`knowledge/README.md`)
  Current entries cover CIFAR augmentation, mixup, cosine schedules, width scaling, EMA, residual initialization, stochastic depth, throughput, and crop padding. After EXP-055, most direct augmentation and generic regularization paths around the active anchor are measured negative.
- **Local experiment trajectory** (`experiment-indices/maximize-cifar10-best-test-accuracy.tsv`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`)
  The strongest validated current recipe combines reflection padding, label smoothing 0.05, 28/56/112 width, 21k first LR drop, and stronger coupled weight decay `2e-4`. This makes optimizer/normalization parameter treatment one of the few remaining low-overhead spaces with a plausible mechanism.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; EXP-056 must reach at least `94.07%` to count as an improvement.
- The current anchor is `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, label smoothing 0.05, FP32 compile, channels-last, and once-per-epoch validation.
- EXP-055 resolved the mixup uncertainty: `MIXUP_ALPHA=0.1` completed and reached the LR drop, but peaked at 93.85%, below the anchor and threshold.
- High-importance failed spaces include schedule-only second drops, weight averaging variants, isolated batch-size deviations, and width increases beyond 28/56/112. Medium recurring failures include cosine schedules, residual-branch BN down-scaling, scalar LR deviations, cutout, and label-smoothing deviations.
- EXP-027 previously excluded BatchNorm and bias from weight decay and failed on an older anchor, peaking at 92.99% below a 93.23% baseline. That was before the current label-smoothed reflection anchor and before `WEIGHT_DECAY=2e-4` became the best setting, so the same mechanism may behave differently as a coupled strong-decay refinement.
- Recent low-importance failures now include mild RandAugment, ColorJitter, stochastic depth, and mixup. That weakens another isolated regularization attempt and increases the relative value of a cheap optimizer parameter-group probe.

## Candidate Ideas

### 1. Strong Weight Decay on Weights Only
**Summary**: Keep `WEIGHT_DECAY=2e-4` for convolution and linear weights, but put BatchNorm parameters and biases into a zero-weight-decay optimizer group. Preserve every other current anchor setting.

**Reasoning**: EXP-038 showed stronger coupled decay improves the label-smoothed reflection anchor, but global decay also shrinks BatchNorm affine parameters and biases. Excluding those parameters could preserve the useful conv/linear shrinkage while allowing normalization and bias terms to calibrate late refinement. This differs from EXP-027 because it tests the mechanism on the stronger `2e-4` anchor where over-regularizing affine parameters is more plausible.

**Sources**: EXP-027 no-decay BN/bias failure; EXP-038 stronger-decay improvement; `goal-learnings/maximize-cifar10-best-test-accuracy.md` failed approaches and patterns; current `train.py` optimizer construction.

**Estimated Effort**: low

**Risk Assessment**: The same idea already failed once on an older anchor, so prior evidence is negative. If the stronger-decay context does not change the mechanism, this will likely land below baseline. Implementation must avoid accidentally excluding all weights or changing optimizer momentum behavior.

### 2. Tiny Final Classifier Dropout
**Summary**: Add `F.dropout(out, p=0.05, training=self.training)` after global average pooling and before the final linear layer.

**Reasoning**: This is a low-overhead regularizer that targets classifier co-adaptation rather than residual branch dynamics or input augmentation. It should preserve step count, parameter count, evaluation determinism, and the 21k LR drop.

**Sources**: EXP-054 stochastic-depth report; EXP-055 mixup report; failed isolated augmentation/regularization entries in goal learnings.

**Estimated Effort**: low

**Risk Assessment**: Isolated regularizers have recently underperformed, and the model already has label smoothing plus stronger weight decay. Even small dropout may soften final fitting and repeat the mixup/stochastic-depth pattern.

### 3. Mixup Without Label Smoothing
**Summary**: Retry mixup alpha 0.1 but set label smoothing to 0.0 only inside the mixup loss, leaving the rest of the anchor unchanged.

**Reasoning**: EXP-055 may have over-regularized by combining mixup target interpolation with label smoothing. Removing smoothing only for mixup would test a coupled regularization-balance explanation rather than a direct mixup retry.

**Sources**: EXP-055 no-improvement report; label-smoothing pattern EXP-032 and failed smoothing deviations EXP-033/EXP-037; `knowledge/papers/mixup-beyond-erm.md`.

**Estimated Effort**: medium

**Risk Assessment**: This reopens mixup immediately after a clean no-improvement and also weakens a validated anchor component. It is scientifically coherent but lower priority unless optimizer/normalization probes fail.

## Idea Evaluation

Strong Weight Decay on Weights Only has the best balance of novelty, mechanism clarity, and cost. It is not a broad retread of EXP-027 because the active anchor now depends on stronger `2e-4` decay; the failure mechanism may change when global shrinkage is stronger and label smoothing/reflection padding are already in place. It also has almost no throughput risk and keeps the model/evaluation graph unchanged.

Tiny final classifier dropout is easy, but recent evidence disfavors isolated regularization. Stochastic depth and mixup both reached the LR drop and still underperformed, while cutout and ColorJitter are already recurring failures. Dropout remains a backup, not the best next experiment.

Mixup without label smoothing has a plausible interaction story, but it is too close to EXP-055 for the immediate next loop and touches a validated anchor component. It should wait until a few non-mixup, non-augmentation mechanisms are tested.

The lead candidate should therefore be a parameter-group optimizer probe: preserve strong decay on true weights while excluding normalization and bias parameters from decay.

## Chosen Idea
**Selected**: Strong Weight Decay on Weights Only

**Why this idea**:
It targets the current anchor's most successful recent mechanism, stronger weight decay, but tests whether the useful part is conv/linear weight shrinkage rather than global shrinkage of BatchNorm affine and bias parameters. It is cheap, interpretable, and distinct from the now-weak direct regularization and augmentation paths.

**Hypothesis**:
Keeping `WEIGHT_DECAY=2e-4` on convolution/linear weights while setting weight decay to zero for BatchNorm and bias parameters will improve late calibration enough to reach at least `94.07%`, without changing throughput, parameter count, or the step-21000 LR drop.
