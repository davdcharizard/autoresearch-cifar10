# Brainstorm EXP-067
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features** (`knowledge/papers/cutmix-regularization.md`)
  CutMix relies on beta-distributed lambda sampling to choose patch area and mixed-label weights. After probability brackets around `p=0.5` failed, alpha is the next clean static way to alter CutMix strength without changing the benchmark or the model.

## Experimental History Review

- Current best remains EXP-064 at `best_test_acc=94.11%`, commit `1119ff8`, with `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, and endpoint label smoothing 0.05.
- The active goal requires `best_test_acc >= 94.21%` to count as an improvement; ties and smaller gains are classified as no-improvement.
- EXP-065 tested `CUTMIX_PROB=0.25` and reached 94.09%. EXP-066 tested `CUTMIX_PROB=0.75` and tied 94.11% but failed the noise guard. This closes the simple probability bracket and keeps `p=0.5` as the best tested frequency.
- Goal learnings now record CutMix probability brackets away from `p=0.5` as a medium-importance failed approach, while preserving probabilistic CutMix itself as a medium-importance validated pattern.
- Label-smoothing deviations, schedule-only changes, direct mixup variants, and Cutout masking are repeated failed families, so the next test should stay inside the regional-mixing mechanism but avoid changing smoothing or schedule.

## Candidate Ideas

### 1. CutMix Alpha 0.5
**Summary**: Restore the validated `CUTMIX_PROB=0.5` frequency and lower `CUTMIX_ALPHA` from 1.0 to 0.5. This changes the beta distribution used for lambda sampling, testing a higher-variance patch-area distribution while leaving all other recipe components unchanged.

**Reasoning**: EXP-064 proved CutMix helps, while EXP-065/066 show frequency changes do not improve the anchor. Alpha controls a different part of CutMix strength: how large and variable the pasted regions and label weights are. A lower alpha may produce more decisive regional replacements that improve invariance without increasing the fraction of mixed batches.

**Sources**: `knowledge/papers/cutmix-regularization.md`; `reports/exp-report-064.md`; `reports/exp-report-066.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: More extreme patch sizes may over-regularize or make training noisier, yielding a valid no-improvement. Code risk is minimal because this is a one-line top-level constant change.

### 2. CutMix Alpha 2.0
**Summary**: Restore `CUTMIX_PROB=0.5` and raise `CUTMIX_ALPHA` from 1.0 to 2.0. This biases lambda values closer to 0.5 and tests more consistently mid-sized regional replacements.

**Reasoning**: The opposite alpha bracket may smooth the patch-area distribution and reduce very small or very large replacements. If EXP-064's late peak was helped by regularization but hurt by high area variance, a higher alpha could stabilize refinement.

**Sources**: `knowledge/papers/cutmix-regularization.md`; `reports/exp-report-064.md`; `reports/exp-report-066.md`.

**Estimated Effort**: low

**Risk Assessment**: More consistently mid-sized patches may behave like stronger target noise and reduce clean image fitting. It is also less directly motivated than alpha 0.5 because EXP-066 already suggests stronger regularization can tie but not beat the anchor.

### 3. Post-Drop CutMix Probability Schedule
**Summary**: Use `CUTMIX_PROB=0.5` before the first LR drop, then reduce CutMix probability after step 21000 to let low-LR refinement train on cleaner labels.

**Reasoning**: EXP-064 and EXP-066 show useful late best checkpoints but lower final accuracy, suggesting the refinement phase may not always benefit from the same mixed-label pressure. A schedule could preserve early CutMix invariance and reduce late noise.

**Sources**: `reports/exp-report-064.md`; `reports/exp-report-066.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: medium

**Risk Assessment**: Schedule-only and smoothing deviations are recurring failed families. This adds conditional training-loop logic and makes attribution weaker than a static alpha bracket.

## Idea Evaluation

The strongest next experiment is an alpha bracket rather than another probability change. The probability bracket now has two valid no-improvement results around the successful `p=0.5` anchor, while CutMix itself remains the only recent mechanism that produced a new baseline. Alpha changes the CutMix patch-area and label-weight distribution without touching model capacity, optimizer, label smoothing, schedule, or validation cadence.

`CUTMIX_ALPHA=0.5` is the preferred first alpha bracket because it tests whether more varied and sometimes more decisive regional replacement gives a stronger best-checkpoint signal than the symmetric `alpha=1.0` anchor. `CUTMIX_ALPHA=2.0` is useful as the opposite bracket, but it is better as a follow-up because the main question after frequency closure is whether the region-size distribution needs more expressivity. The post-drop schedule is plausible but less clean and conflicts with repeated schedule/smoothing failure patterns.

## Chosen Idea
**Selected**: CutMix Alpha 0.5

**Why this idea**:
It is the cleanest remaining static bracket inside the validated CutMix mechanism. It preserves the locally best probability `p=0.5` while changing only the patch-area distribution, giving a direct read on whether EXP-064's alpha setting is locally optimal.

**Hypothesis**:
If the current CutMix anchor benefits from more varied regional replacement sizes, setting `CUTMIX_ALPHA=0.5` with `CUTMIX_PROB=0.5` will raise `best_test_acc` to at least 94.21%; otherwise it will establish `alpha=1.0` as the better side of the first alpha bracket.
