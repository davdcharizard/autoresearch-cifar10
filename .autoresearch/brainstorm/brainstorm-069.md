# Brainstorm EXP-069
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features** (`knowledge/papers/cutmix-regularization.md`)
  CutMix regionally mixes image patches and labels using a beta-sampled patch area. The current project has validated the mechanism with EXP-064, but static alpha/probability brackets now show the simple strength axis is locally optimized.

## Experimental History Review

- Current best remains EXP-064 at `best_test_acc=94.11%`, commit `1119ff8`, with `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, endpoint label smoothing 0.05, `WEIGHT_DECAY=2e-4`, reflection crop padding, and the 21k first LR drop.
- The goal requires `best_test_acc >= 94.21%` to count as an improvement; smaller gains and ties are `no-improvement`.
- EXP-065 and EXP-066 bracketed static CutMix probability away from `p=0.5`: `p=0.25` reached 94.09%, and `p=0.75` tied 94.11% but missed the threshold.
- EXP-067 and EXP-068 bracketed static CutMix alpha away from `alpha=1.0`: `alpha=0.5` reached 94.07%, and `alpha=2.0` reached 94.00%.
- The history now supports keeping the static `alpha=1.0, p=0.5` anchor. The remaining gap is temporal: several CutMix runs peak around epochs 77-98 and finish lower, so late mixed-label noise may be limiting low-LR refinement.
- Strong warnings remain against exact retries of label-smoothing changes, direct mixup, Cutout, SE, EMA/SWA, isolated batch-size retunes, isolated LR changes, and isolated cosine/schedule-only changes.

## Candidate Ideas

### 1. Post-Drop CutMix Probability Taper to 0.25
**Summary**: Keep `CUTMIX_ALPHA=1.0` and `CUTMIX_PROB=0.5` through the first LR drop, then reduce CutMix probability to 0.25 once `step >= 21000`.

**Reasoning**: EXP-064 shows early-to-mid CutMix at `p=0.5` is valuable, while EXP-065 shows full-run `p=0.25` remains close to the anchor. A post-drop taper composes those signals: keep the stronger regional regularization during high-LR representation learning, then reduce mixed-label noise during low-LR refinement. This is not an isolated LR schedule change because the intervention targets augmentation strength conditional on an already validated LR phase boundary.

**Sources**: `knowledge/papers/cutmix-regularization.md`; `reports/exp-report-064.md`; `reports/exp-report-065.md`; `reports/exp-report-068.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: The most likely failure is no-improvement if late CutMix still helps generalization or if the taper is too small to matter. Code risk is low because the training loop already has `step` before sampling CutMix.

### 2. Post-Drop CutMix Off Switch
**Summary**: Keep `CUTMIX_ALPHA=1.0` and `CUTMIX_PROB=0.5` through step 21000, then disable CutMix entirely for clean low-LR refinement.

**Reasoning**: The late plateau and lower final accuracy may come from persistent mixed-label noise after the model has already learned useful invariances. Turning CutMix off after the LR drop is a sharper version of candidate 1 and gives the model fully clean labels for the refinement phase while preserving early regularization.

**Sources**: `reports/exp-report-064.md`; `reports/exp-report-067.md`; `reports/exp-report-068.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: The abrupt regularization change may overfit or erase the benefit of regional mixing too early, especially because post-drop label-smoothing annealing already failed. It is more aggressive than a taper to 0.25.

### 3. CutMix-Specific Label Smoothing Reduction
**Summary**: Keep static `CUTMIX_ALPHA=1.0` and `CUTMIX_PROB=0.5`, but lower `CUTMIX_LABEL_SMOOTHING` from 0.05 to 0.03 only for CutMix batches.

**Reasoning**: CutMix already blends targets, so endpoint smoothing may be slightly redundant on mixed batches. Lowering only the CutMix-batch smoothing tests an interaction that differs from the failed global label-smoothing deviations.

**Sources**: `reports/exp-report-033.md`; `reports/exp-report-064.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: Label-smoothing deviations are a high-importance failed family, so this has a worse prior than the CutMix scheduling candidates. It is still scoped, but a failure would be less informative than testing the temporal CutMix-noise hypothesis first.

## Idea Evaluation

Candidate 1 has the best evidence balance. It preserves the validated EXP-064 early CutMix anchor, uses the 21k LR drop already present in the recipe, and makes the smallest temporal adjustment needed to test whether late mixed-label noise is limiting refinement. Candidate 2 tests the same mechanism but is more abrupt and overlaps more with the failed post-drop hard-label sharpening pattern from EXP-057. Candidate 3 is a plausible interaction, but the goal learnings strongly caution against label-smoothing deviations unless there is a stronger reason.

The expected impact of a taper is modest but credible. EXP-065's `p=0.25` full-run result was only 0.02pp below the baseline, while EXP-064's `p=0.5` established the current best. Combining stronger early regional mixing with milder late regularization is the cleanest remaining CutMix experiment after static probability and alpha were bracketed.

## Chosen Idea
**Selected**: Post-Drop CutMix Probability Taper to 0.25

**Why this idea**:
It is the lowest-risk way to test the remaining temporal hypothesis inside the only mechanism that recently improved the baseline. It avoids retrying static CutMix scalar brackets and stays distinct from recurring failed global schedule or label-smoothing changes.

**Hypothesis**:
If persistent low-LR CutMix noise is limiting final refinement, reducing `CUTMIX_PROB` from 0.5 to 0.25 after step 21000 will preserve early regional regularization while improving late accuracy enough to reach at least 94.21%.
