# Brainstorm EXP-090
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  CutMix remains the strongest successful non-spatial regularizer in this project, and its regional label mixing differs from the failed direct mixup and Cutout families.
- **Recent project reports** (`reports/exp-report-085.md`, `reports/exp-report-086.md`, `reports/exp-report-087.md`, `reports/exp-report-088.md`, `reports/exp-report-089.md`)
  Project-specific evidence now dominates ideation: padding 3 / flip p=0.4 is the current best anchor, and isolated crop/flip/stronger-decay retunes around it have failed.
- **CutMix probability bracket reports** (`reports/exp-report-064.md`, `reports/exp-report-065.md`, `reports/exp-report-066.md`)
  The older broad probability bracket showed p=0.5 was best under the pre-spatial anchor, but p=0.25 was a near miss and p=0.75 only tied, leaving a smaller lower-side probability adjustment plausible on the stronger spatial anchor.

No new external search was needed. EXP-090 is an exploitation experiment driven by the local CIFAR-10 trajectory and already recorded CutMix/spatial knowledge.

## Experimental History Review

- Current best is EXP-085 at `best_test_acc=94.51%` from commit `83d4e94`; with the +0.10pp noise guard, EXP-090 must reach at least 94.61% to count as an improvement.
- The active anchor is reflection crop padding 3, `RandomHorizontalFlip(p=0.4)`, unit-std normalization, static CutMix alpha 1.0 / probability 0.5 / label smoothing 0.05, clean label smoothing 0.05, ResNet-20 `(28,56,112)`, `WEIGHT_DECAY=2e-4`, and first LR drop at step 21000.
- EXP-086, EXP-087, and EXP-089 close the isolated spatial bracket around the EXP-085 anchor: crop padding 2, flip p=0.425, and flip p=0.375 all regressed below 94.51%.
- EXP-088 closes the stronger scalar decay side around the spatial anchor: `WEIGHT_DECAY=2.5e-4` regressed to 94.07%.
- CutMix probability away from p=0.5 is a medium-importance failed family under the older anchor, but the current spatial anchor is materially different and the broad p=0.25 move was close enough to motivate a finer p=0.4 test before abandoning CutMix interaction tuning.
- High-importance failed families remain off the table for isolated retry: schedule-only second drops, EMA/SWA-style averaging, batch-size deviations, label-smoothing deviations, and LR startup changes.
- Medium-importance failed families to avoid as isolated broad retries include CutMix alpha brackets, policy augmentation, SE gates, downsampling smoothing, transition topology changes, head-only tweaks, and further isolated flip probability adjustments.

## Candidate Ideas

### 1. Fine Lower CutMix Probability p=0.4 on the Spatial Anchor
**Summary**: Keep the padding-3 / flip-p=0.4 spatial anchor and reduce `CUTMIX_PROB` from 0.5 to 0.4. Preserve `CUTMIX_ALPHA=1.0`, CutMix endpoint label smoothing 0.05, clean-batch label smoothing 0.05, unit-std normalization, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last, fixed 300s budget, and validation cadence.

**Reasoning**: Spatial tuning is now locally closed, so the next plausible interaction is how often the successful regional-mixing regularizer should be applied under the less spatially aggressive anchor. The older p=0.25 probability bracket failed, but it was a broad reduction under a different spatial recipe and still landed close to the then-baseline. A finer p=0.4 reduction may reduce mixed-label pressure while retaining most of the CutMix signal, potentially improving peak accuracy on the stronger padding-3 / flip-p=0.4 anchor.

**Sources**: `knowledge/papers/cutmix-regularization.md`; `reports/exp-report-064.md`; `reports/exp-report-065.md`; `reports/exp-report-066.md`; `reports/exp-report-085.md`; `reports/exp-report-089.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: The main risk is that `CUTMIX_PROB=0.5` remains optimal and p=0.4 simply removes useful regional regularization. Failure should be a clean no-improvement rather than a crash, and the run preserves all hard constraints.

### 2. Fine Lower CutMix Alpha 0.75 on the Spatial Anchor
**Summary**: Keep the active spatial anchor and reduce `CUTMIX_ALPHA` from 1.0 to 0.75 while preserving `CUTMIX_PROB=0.5` and all other anchor settings. This tests a smaller patch-area-distribution change than the failed alpha 0.5 bracket.

**Reasoning**: CutMix alpha 0.5 and 2.0 both failed under the older anchor, but those were broad moves. Alpha 0.75 could slightly alter patch-area variance while keeping the successful application frequency. This is mechanistically plausible if the spatial anchor changes the amount of local corruption the model can tolerate.

**Sources**: `knowledge/papers/cutmix-regularization.md`; `reports/exp-report-067.md`; `reports/exp-report-068.md`; `reports/exp-report-085.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: CutMix alpha brackets are already a medium-importance failed family, so this risks being a low-value local retry. If it fails, it should still provide clean closure on finer CutMix strength tuning under the spatial anchor.

### 3. Fine Lower Weight Decay 1.75e-4 on the Spatial Anchor
**Summary**: Keep the active spatial and CutMix anchor and reduce `WEIGHT_DECAY` from `2e-4` to `1.75e-4`. Preserve crop, flip, CutMix, label smoothing, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last, fixed budget, and validation cadence.

**Reasoning**: EXP-088 showed stronger decay over-regularizes the current anchor, while EXP-041's `1.5e-4` lower-decay test was on an older recipe. A smaller lower-side bracket could test whether the spatially de-regularized anchor now wants slightly less shrinkage. This is less compelling than CutMix probability because the accumulated evidence increasingly favors `2e-4` as the scalar decay anchor.

**Sources**: `reports/exp-report-041.md`; `reports/exp-report-085.md`; `reports/exp-report-088.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: Prior lower-decay and stronger-decay experiments both argue against scalar decay retuning as the missing lever. The likely outcome is under-regularization and no-improvement, but the experiment is safe and scoped.

## Idea Evaluation

The p=0.4 CutMix probability test has the best balance of evidence and novelty. It uses the strongest successful non-spatial mechanism, respects the newly closed spatial anchor, and differs from the failed p=0.25 / p=0.75 brackets by being both finer and evaluated on a materially stronger spatial recipe. Its expected lift is modest, but the causal mechanism is clear: slightly reduce regional mixed-label frequency after spatial de-regularization has already improved the anchor.

The alpha 0.75 candidate is also a CutMix interaction, but it re-enters a medium-importance failed family with less direct evidence. The older alpha brackets were both below threshold, while the probability lower bracket was closer to baseline and may be more sensitive to the spatial anchor.

The 1.75e-4 weight-decay candidate is safe but weaker. EXP-088 and the broader decay history suggest `2e-4` is already the local decay optimum, and the project should avoid spending too many loops on scalar brackets once distinct regularization interactions remain.

## Chosen Idea
**Selected**: Fine Lower CutMix Probability p=0.4 on the Spatial Anchor

**Why this idea**:
It is the cleanest next coupled-regularization test after isolated spatial tuning closed. It keeps the validated padding-3 / flip-p=0.4 anchor intact, changes only one CutMix scalar, and probes whether the current recipe wants slightly less regional mixed-label pressure than the older p=0.5 CutMix anchor.

**Hypothesis**:
If the padding-3 / flip-p=0.4 anchor is now slightly over-regularized by applying CutMix to half of batches, reducing `CUTMIX_PROB` from 0.5 to 0.4 will raise `best_test_acc` from 94.51% to at least 94.61%.
