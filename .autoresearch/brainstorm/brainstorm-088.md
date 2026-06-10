# Brainstorm EXP-088
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  The current best recipe still depends on static regional mixing, so candidate ideas should preserve `CUTMIX_ALPHA=1.0` and `CUTMIX_PROB=0.5` unless directly testing a strong CutMix interaction.
- **CIFAR crop augmentation** (`knowledge/references/torchvision-randomcrop-padding.md`)
  Reflection crop padding remains a no-dependency spatial knob, but the recent local history now points to padding 3 / flip p=0.4 as the active spatial anchor rather than more crop reduction.
- **Recent project reports** (`reports/exp-report-085.md`, `reports/exp-report-086.md`, `reports/exp-report-087.md`, `reports/exp-report-038.md`, `reports/exp-report-039.md`, `reports/exp-report-041.md`)
  The strongest evidence is project-specific: EXP-085 established the 94.51% spatial anchor, EXP-086/087 closed adjacent spatial weakening and upper flip restoration, and EXP-038/039/041 bracketed older-anchor weight decay around 2e-4.

No new external search was needed. The next decision is local to the fixed CIFAR recipe and is better grounded by the dense experiment history than by broad literature.

## Experimental History Review

- Current best is EXP-085 at `best_test_acc=94.51%` from commit `83d4e94`; the +0.10 percentage-point noise guard requires `best_test_acc >= 94.61%` for EXP-088 to count as an improvement.
- The current validated anchor is reflection crop padding 3 plus `RandomHorizontalFlip(p=0.4)`, with unit-std normalization, static CutMix alpha/probability/label smoothing, clean label smoothing 0.05, `WEIGHT_DECAY=2e-4`, `LR=0.1`, and the 21k first LR drop.
- EXP-086 showed crop padding below 3 under flip p=0.4 regresses to 94.22%, and EXP-087 showed p=0.425 under padding 3 regresses to 94.34%. Together with EXP-083/084, p=0.4 is now a strong local spatial anchor.
- EXP-038 showed increasing weight decay from 1e-4 to 2e-4 improved the older reflection/label-smoothed anchor by +0.27pp. EXP-039 and EXP-041 showed 3e-4 and 1.5e-4 were poor on that older anchor, but the newer spatial anchor is less spatially regularized and may need a slightly different shrinkage balance.
- Closed or low-priority families remain fixed: broad width increases, schedule-only second drops, weight averaging, batch-size deviations, label-smoothing deviations, LR startup changes, CutMix alpha/probability broad brackets, policy augmentation, cutout, SE, downsampling smoothing, pre-activation blocks, crop padding below 3, and flip probability above p=0.4.

## Candidate Ideas

### 1. Fine Stronger Weight Decay 2.5e-4 on the Padding-3 / Flip-p=0.4 Anchor
**Summary**: Keep the EXP-085 spatial anchor unchanged and increase `WEIGHT_DECAY` from `2e-4` to `2.5e-4`. Preserve reflection crop padding 3, `RandomHorizontalFlip(p=0.4)`, unit-std normalization, static CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer type, LR milestones, batch size, seed, compile/channels-last, fixed 300s budget, and validation cadence.

**Reasoning**: EXP-085's improvement came from coordinated spatial de-regularization. After EXP-086 and EXP-087, additional spatial moves look unlikely, so the next plausible lever is to rebalance non-spatial regularization around the new spatial anchor. EXP-038 proved that stronger decay can be a real improvement axis, while EXP-039 only rules out the larger `3e-4` step on the older anchor. A smaller `2.5e-4` bracket may add enough shrinkage to support the milder spatial augmentation without over-regularizing as much as `3e-4`.

**Sources**: `reports/exp-report-038.md`; `reports/exp-report-039.md`; `reports/exp-report-041.md`; `reports/exp-report-085.md`; `reports/exp-report-086.md`; `reports/exp-report-087.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: The older-anchor decay bracket may transfer directly, in which case any increase above `2e-4` will underperform. Failure should be a clean no-improvement with no expected crash or throughput risk.

### 2. Fine Lower Flip Bracket p=0.375 Under Padding 3
**Summary**: Keep reflection crop padding 3 and reduce `RandomHorizontalFlip(p=0.4)` to `p=0.375`, preserving all other EXP-085 anchor settings.

**Reasoning**: This is the remaining clean local spatial closure test. EXP-083's p=0.35 failure happened under padding 4, not padding 3, so p=0.375 under padding 3 is not an exact retry. However, EXP-086 already showed further spatial de-regularization through crop padding 2 is harmful, and EXP-083 makes the mechanism weak.

**Sources**: `reports/exp-report-083.md`; `reports/exp-report-085.md`; `reports/exp-report-086.md`; `reports/exp-report-087.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: This may under-regularize horizontal invariance and peak below baseline. It is useful for map closure but has lower expected improvement than a coupled non-spatial retune.

### 3. Fine Lower CutMix Probability 0.4 on the Spatial Anchor
**Summary**: Keep the padding-3 / flip-p=0.4 spatial anchor and lower `CUTMIX_PROB` from 0.5 to 0.4 while preserving CutMix alpha 1.0 and label smoothing 0.05.

**Reasoning**: The spatial anchor now uses milder geometric augmentation, and a slightly lower CutMix frequency might reduce late underfitting while retaining regional mixing. This is distinct from the broad failed p=0.25 bracket, but it re-enters a medium-importance failed family and is therefore lower confidence.

**Sources**: `reports/exp-report-064.md`; `reports/exp-report-065.md`; `reports/exp-report-066.md`; `reports/exp-report-085.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `knowledge/papers/cutmix-regularization.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: CutMix probability brackets away from 0.5 have already failed on the older anchor. The smaller move may be too weak to clear the noise guard or may simply reduce a validated regularizer.

## Idea Evaluation

The fine stronger weight-decay bracket has the strongest mechanism for the current state. The last two experiments indicate isolated spatial retuning is running out of room, but the successful EXP-085 spatial de-regularization changes the balance of regularization in the recipe. A modest increase to `2.5e-4` tests that interaction without repeating the failed `3e-4` jump exactly.

The p=0.375 flip bracket is a clean unresolved local test, but its expected impact is lower because every nearby spatial direction except the EXP-085 combination has regressed. It remains useful if the goal is exhaustive spatial bracket closure, not if the goal is highest expected next improvement.

The CutMix p=0.4 candidate is plausible as a coupled regularization retune, but CutMix probability is already a medium-importance failed family. It should wait until the less-closed weight-decay interaction is tested.

## Chosen Idea
**Selected**: Fine Stronger Weight Decay 2.5e-4 on the Padding-3 / Flip-p=0.4 Anchor

**Why this idea**:
It targets the most plausible remaining interaction created by the current best recipe: after lowering spatial augmentation strength, a slightly stronger non-spatial shrinkage term may recover generalization without changing throughput or benchmark scope. It is a low-risk, single-line test and is not an exact retry of the older-anchor `3e-4` failure.

**Hypothesis**:
If the padding-3 / flip-p=0.4 anchor is now slightly under-regularized outside the spatial augmentation axis, increasing `WEIGHT_DECAY` from `2e-4` to `2.5e-4` will raise `best_test_acc` from 94.51% to at least 94.61%.
