# Brainstorm EXP-091
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  CutMix remains the strongest successful non-spatial regularizer in this project; alpha controls the Beta-distributed patch-area mix while probability controls how often regional mixing is applied.
- **Recent project reports** (`reports/exp-report-085.md`, `reports/exp-report-089.md`, `reports/exp-report-090.md`)
  The spatial anchor is now padding 3 / flip p=0.4, isolated flip brackets are closed, and static CutMix probability away from p=0.5 is now a high-importance failed direction.
- **CutMix alpha bracket reports** (`reports/exp-report-067.md`, `reports/exp-report-068.md`)
  Broad alpha moves to 0.5 and 2.0 failed under the older anchor, but no fine alpha move has been tested on the stronger spatial anchor.

No new external search was needed. EXP-091 is a local closure experiment around the existing CutMix implementation and the current spatial anchor.

## Experimental History Review

- Current best is EXP-085 at `best_test_acc=94.51%` from commit `83d4e94`; with the +0.10pp noise guard, EXP-091 must reach at least 94.61% to count as an improvement.
- The active anchor is reflection crop padding 3, `RandomHorizontalFlip(p=0.4)`, unit-std normalization, static CutMix alpha 1.0 / probability 0.5 / label smoothing 0.05, clean label smoothing 0.05, ResNet-20 `(28,56,112)`, `WEIGHT_DECAY=2e-4`, and first LR drop at step 21000.
- EXP-086, EXP-087, and EXP-089 close isolated crop/flip retuning around the spatial anchor; EXP-090 promotes static CutMix probability moves away from p=0.5 to high-importance failure.
- EXP-067 and EXP-068 show broad CutMix alpha brackets away from 1.0 failed under the older pre-spatial anchor, so alpha retuning is a medium-importance failed family.
- EXP-088 showed stronger weight decay over-regularizes the spatial anchor, while older lower-decay tests argue scalar decay is unlikely to be the missing lever.
- High-importance failed families remain off the table for direct retry: schedule-only second drops, weight averaging, batch-size deviations, label-smoothing deviations, LR startup changes, and CutMix probability moves away from p=0.5.

## Candidate Ideas

### 1. Fine Lower CutMix Alpha 0.75 on the Spatial Anchor
**Summary**: Keep the padding-3 / flip-p=0.4 spatial anchor and reduce `CUTMIX_ALPHA` from 1.0 to 0.75. Preserve `CUTMIX_PROB=0.5`, CutMix endpoint label smoothing 0.05, clean-batch label smoothing 0.05, unit-std normalization, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last, fixed 300s budget, and validation cadence.

**Reasoning**: CutMix probability is now closed, but alpha controls a different part of the regional-mixing mechanism: patch-area distribution. The broad alpha 0.5 and 2.0 brackets failed under the older anchor, yet a smaller alpha 0.75 move on the stronger spatial anchor could slightly adjust target-noise/patch-size balance without changing application frequency.

**Sources**: `knowledge/papers/cutmix-regularization.md`; `reports/exp-report-067.md`; `reports/exp-report-068.md`; `reports/exp-report-085.md`; `reports/exp-report-090.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: This re-enters a medium-importance failed family, so the likely outcome is no-improvement. It is still a clean one-scalar closure test and should fail safely if alpha 1.0 remains optimal.

### 2. Fine Lower Weight Decay 1.75e-4 on the Spatial Anchor
**Summary**: Keep the active spatial and CutMix anchor and reduce `WEIGHT_DECAY` from `2e-4` to `1.75e-4`. Preserve crop, flip, CutMix, label smoothing, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last, fixed budget, and validation cadence.

**Reasoning**: EXP-088 closed the stronger-decay side under the spatial anchor, while the older `1.5e-4` lower-decay test was run before the best spatial recipe existed. A smaller lower-side move could test whether the current anchor wants marginally less shrinkage after spatial de-regularization.

**Sources**: `reports/exp-report-041.md`; `reports/exp-report-085.md`; `reports/exp-report-088.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: Scalar decay evidence increasingly favors `2e-4`; failure would likely be a clean no-improvement caused by under-regularization.

### 3. Higher BatchNorm Momentum 0.2 on the Spatial Anchor
**Summary**: Keep the current recipe and raise BatchNorm momentum from PyTorch's default 0.1 to 0.2 in all BatchNorm2d layers. Preserve all augmentation, CutMix, optimizer, schedule, batch size, seed, compile/channels-last, fixed budget, and validation cadence.

**Reasoning**: Lower BN momentum 0.05 failed, but the opposite direction has not been tested. Faster BN running-stat adaptation might better track the mixed clean/CutMix training distribution under the current anchor, while leaving parameter count and compute nearly unchanged.

**Sources**: `reports/exp-report-048.md`; `reports/exp-report-064.md`; `reports/exp-report-085.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: This is more invasive than a scalar hyperparameter because it changes module construction. It may destabilize eval statistics and has weaker evidence than the CutMix alpha closure test.

## Idea Evaluation

The fine alpha 0.75 test is the best next experiment because it closes the remaining nearby static CutMix strength question without touching high-importance failed dimensions. The evidence is mixed: broad alpha brackets failed, but the current spatial anchor is stronger and alpha controls patch-area distribution rather than frequency. A no-improvement result would promote alpha retuning toward closure and help move the loop away from CutMix scalar tuning.

The 1.75e-4 weight-decay bracket is safe but less compelling. Stronger decay failed recently, lower decay failed historically, and goal learnings increasingly identify `2e-4` as the scalar decay anchor.

The BN momentum 0.2 diagnostic is distinct, but its mechanism is weaker and implementation touches module construction rather than only top-level scalar regularization. It is better saved for after the CutMix alpha closure and remaining scalar diagnostics.

## Chosen Idea
**Selected**: Fine Lower CutMix Alpha 0.75 on the Spatial Anchor

**Why this idea**:
It is the cleanest remaining static CutMix scalar closure test after probability brackets failed. It changes the regional patch-area distribution while preserving application frequency, spatial anchor, label smoothing, optimizer, schedule, and architecture.

**Hypothesis**:
If the padding-3 / flip-p=0.4 anchor is slightly over-regularized by the current CutMix patch-area distribution, reducing `CUTMIX_ALPHA` from 1.0 to 0.75 will raise `best_test_acc` from 94.51% to at least 94.61%.
