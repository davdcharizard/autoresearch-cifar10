# Brainstorm EXP-068
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features** (`knowledge/papers/cutmix-regularization.md`)
  CutMix samples beta-distributed lambda values, converts them to rectangular patch areas, and trains with area-adjusted mixed labels. After `p=0.5` succeeded and alpha 0.5 failed, alpha remains the cleanest static control over patch-area distribution.

## Experimental History Review

- Current best remains EXP-064 at `best_test_acc=94.11%`, commit `1119ff8`, with `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, endpoint label smoothing 0.05, and the 2e-4 weight-decay reflection-padding anchor.
- The goal requires `best_test_acc >= 94.21%` to count as an improvement; smaller gains and ties are no-improvement under the explicit noise guard.
- EXP-065 and EXP-066 closed the simple probability bracket around `p=0.5`: `p=0.25` peaked at 94.09%, while `p=0.75` tied 94.11% but missed the threshold.
- EXP-067 tested the lower alpha bracket, `CUTMIX_ALPHA=0.5`, and peaked at 94.07%. This preserves most of the CutMix benefit but suggests higher-variance patch sizes do not beat the anchor.
- Goal learnings now recommend keeping `CUTMIX_PROB=0.5`, keeping `CUTMIX_ALPHA=1.0` unless the opposite alpha bracket wins, and avoiding repeated label-smoothing, batch-size, schedule-only, mixup, Cutout, SE, EMA/SWA, and isolated LR/weight-decay deviations.
- The main untried gap inside the successful regional-mixing mechanism is the opposite alpha side: a smoother beta distribution with fewer extreme patch-area samples.

## Candidate Ideas

### 1. CutMix Alpha 2.0
**Summary**: Keep the validated `CUTMIX_PROB=0.5` and raise `CUTMIX_ALPHA` from 1.0 to 2.0. This tests a less-variable lambda distribution with more consistently mid-sized CutMix regions while preserving all other anchor settings.

**Reasoning**: EXP-067 showed that more variable patch areas at alpha 0.5 slightly underperform the alpha 1.0 anchor. The opposite bracket has a clean mechanism: reduce extreme small/large replacements while preserving the regional-mixing frequency that produced the current best. It is the last low-risk static scalar bracket before treating CutMix strength as locally optimized.

**Sources**: `knowledge/papers/cutmix-regularization.md`; `reports/exp-report-064.md`; `reports/exp-report-067.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: More consistently mid-sized patches may over-regularize or inject too much label ambiguity, yielding a no-improvement. Code risk is minimal because this is a one-line constant change.

### 2. Post-Drop CutMix Probability Taper
**Summary**: Keep `CUTMIX_PROB=0.5` before the first LR drop, then reduce CutMix probability after step 21000, for example to 0.25 or 0.0. This tests whether early regional mixing helps invariance while late low-LR refinement benefits from cleaner labels.

**Reasoning**: EXP-064 and EXP-067 both show late best checkpoints followed by lower final accuracy, which may indicate mixed-label noise during refinement. A post-drop taper could preserve the successful early CutMix mechanism while reducing late regularization pressure. This is more complex than an alpha scalar but targets a plausible late-phase failure mode.

**Sources**: `reports/exp-report-064.md`; `reports/exp-report-067.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: medium

**Risk Assessment**: Schedule-only and label-smoothing schedule changes have repeatedly failed, so a CutMix schedule may inherit that risk. It also adds conditional loop logic and weaker attribution than a static bracket.

### 3. Smaller CutMix Label Smoothing Interaction
**Summary**: Keep `CUTMIX_ALPHA=1.0` and `CUTMIX_PROB=0.5`, but lower the endpoint label smoothing only for the CutMix anchor from 0.05 to 0.03. This tests whether CutMix already supplies enough target smoothing and needs slightly sharper labels.

**Reasoning**: Direct label-smoothing deviations have failed, but CutMix is a distinct target-mixing mechanism that may change the best smoothing balance. EXP-033's 0.03 smoothing was a near miss before CutMix existed, so this is a plausible interaction rather than an exact retry.

**Sources**: `reports/exp-report-033.md`; `reports/exp-report-064.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: Label-smoothing deviations are now a high-importance failed family, so this has a higher prior risk than finishing the alpha bracket. A no-improvement would be scientifically valid but less informative than the opposite alpha bracket.

## Idea Evaluation

`CUTMIX_ALPHA=2.0` has the strongest evidence and cleanest mechanism. It directly complements EXP-067, keeps the successful `p=0.5` regional-mixing frequency, and tests the only remaining static CutMix-strength axis with a one-line change. A post-drop taper is mechanistically plausible, but prior schedule-style interventions have failed often enough that it should wait until static alpha bracketing is complete. The smoothing interaction is also plausible, but the goal-learnings file strongly warns against label-smoothing deviations; it is better reserved for after CutMix-specific controls are exhausted.

The expected impact of alpha 2.0 is modest but credible: it can potentially improve late refinement by avoiding extreme patch/label mixtures while preserving the mechanism that produced the current best. Its failure mode is safe and highly informative, because a valid no-improvement would bracket alpha on both sides and establish the EXP-064 `alpha=1.0, p=0.5` setting as locally best.

## Chosen Idea
**Selected**: CutMix Alpha 2.0

**Why this idea**:
It is the cleanest remaining static experiment inside the only recent mechanism that produced a new baseline. It pairs directly with EXP-067 and avoids revisiting recurring failed families before the CutMix alpha bracket is complete.

**Hypothesis**:
If EXP-064's CutMix anchor is limited by overly variable patch-area samples, setting `CUTMIX_ALPHA=2.0` with `CUTMIX_PROB=0.5` will stabilize regional replacement enough to reach at least 94.21%; otherwise alpha 1.0 is locally best.
