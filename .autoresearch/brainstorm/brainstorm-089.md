# Brainstorm EXP-089
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CIFAR crop augmentation** (`knowledge/references/torchvision-randomcrop-padding.md`)
  Reflection crop padding remains validated, but recent experiments indicate the active crop setting should stay at padding 3.
- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  Static CutMix remains part of the best recipe, and recent failed retunes argue against changing it before closing the remaining spatial bracket.
- **Recent project reports** (`reports/exp-report-085.md`, `reports/exp-report-086.md`, `reports/exp-report-087.md`, `reports/exp-report-088.md`)
  Project-specific evidence dominates this choice: padding 3 / flip p=0.4 is the current best anchor, crop padding 2 failed, upper flip p=0.425 failed, and stronger decay p=2.5e-4 failed.

No new external search was needed. EXP-089 is a local closure experiment around an empirically discovered spatial anchor.

## Experimental History Review

- Current best is EXP-085 at `best_test_acc=94.51%` from commit `83d4e94`; the +0.10 percentage-point noise guard requires `best_test_acc >= 94.61%` for EXP-089 to count as an improvement.
- The validated spatial anchor is reflection crop padding 3 plus `RandomHorizontalFlip(p=0.4)`.
- EXP-086 showed crop padding 2 under flip p=0.4 regressed to 94.22%, closing further crop-jitter reduction.
- EXP-087 showed p=0.425 under padding 3 regressed to 94.34%, and EXP-084 had already shown p=0.45 was poor under padding 4. The upper flip side is now a medium-importance failed direction.
- EXP-083 showed p=0.35 under padding 4 regressed to 94.17%, but p=0.375 under padding 3 remains untested and is not an exact retry.
- EXP-088 showed `WEIGHT_DECAY=2.5e-4` on the spatial anchor regressed to 94.07%, closing stronger scalar decay as a complement to this anchor.
- Closed or low-priority families remain fixed: stronger weight decay, crop padding below 3, flip above p=0.4, broad CutMix probability/alpha brackets, label-smoothing deviations, LR startup changes, batch-size changes, schedule-only changes, policy augmentation, cutout, SE, downsampling smoothing, pre-activation blocks, and broad architecture changes.

## Candidate Ideas

### 1. Fine Lower Flip Bracket p=0.375 Under Padding 3
**Summary**: Keep reflection crop padding 3 and reduce `RandomHorizontalFlip(p=0.4)` to `p=0.375`. Preserve unit-std normalization, static CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last, fixed 300s budget, and validation cadence.

**Reasoning**: This is the last clean local spatial bracket around the current best anchor. It tests whether the older p=0.35 failure was too large a reduction and whether padding 3 changes the flip/crop interaction enough for a smaller lower-side move to help. The expected value is modest, but the experiment gives useful closure before moving away from spatial tuning.

**Sources**: `reports/exp-report-083.md`; `reports/exp-report-085.md`; `reports/exp-report-086.md`; `reports/exp-report-087.md`; `reports/exp-report-088.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: The most likely failure mode is under-regularizing horizontal invariance, similar to p=0.35 and crop padding 2. Failure should be a clean no-improvement rather than a crash.

### 2. Fine Lower CutMix Probability 0.4 on the Spatial Anchor
**Summary**: Keep the padding-3 / flip-p=0.4 spatial anchor and lower `CUTMIX_PROB` from 0.5 to 0.4 while preserving CutMix alpha 1.0 and label smoothing 0.05.

**Reasoning**: With spatial tuning mostly bracketed, a small reduction in regional mixing frequency might reduce late underfitting while retaining the proven CutMix mechanism. This differs from the broad failed p=0.25 bracket, but it re-enters a medium-importance failed family and should wait until the spatial bracket is closed.

**Sources**: `reports/exp-report-064.md`; `reports/exp-report-065.md`; `reports/exp-report-066.md`; `reports/exp-report-085.md`; `reports/exp-report-088.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `knowledge/papers/cutmix-regularization.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: CutMix probability brackets away from p=0.5 already failed on the older anchor; the smaller p=0.4 move may be too weak or may simply reduce a validated regularizer.

### 3. Fine Lower Weight Decay 1.75e-4 on the Spatial Anchor
**Summary**: Keep the spatial anchor and reduce `WEIGHT_DECAY` from `2e-4` to `1.75e-4`, preserving all other settings.

**Reasoning**: EXP-088 closed the stronger side of decay under the spatial anchor, while the older `1.5e-4` test was bad before the spatial anchor existed. A smaller lower-side move could test whether the current anchor wants marginally less shrinkage. This is lower priority because goal learnings now strongly favor keeping `2e-4`.

**Sources**: `reports/exp-report-041.md`; `reports/exp-report-085.md`; `reports/exp-report-088.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: Prior lower-decay results were poor, and scalar decay retuning is now a low-priority family. Failure should be clean but likely.

## Idea Evaluation

The p=0.375 flip bracket is the best next experiment because it closes the only remaining local spatial direction around the current best anchor. Its expected improvement is not high, but it is more scientifically clean than reopening medium-importance CutMix probability or scalar weight-decay families.

The CutMix p=0.4 candidate is a plausible future coupled retune, but both p=0.25 and p=0.75 have already failed away from p=0.5. It should wait until the spatial bracket is fully closed.

The lower weight-decay p=1.75e-4 candidate is less attractive after EXP-088: the evidence increasingly says `2e-4` is the scalar decay anchor, and stronger/lower scalar decay moves are not the missing lever.

## Chosen Idea
**Selected**: Fine Lower Flip Bracket p=0.375 Under Padding 3

**Why this idea**:
It is the cleanest remaining local test around the EXP-085 spatial anchor and avoids re-entering stronger failed families. If it fails, the loop can treat padding 3 / flip p=0.4 as fully bracketed and move to distinct coupled mechanisms with a clearer conscience.

**Hypothesis**:
If the padding-3 / flip-p=0.4 anchor still has a small amount of excess horizontal invariance, reducing `RandomHorizontalFlip` from p=0.4 to p=0.375 will raise `best_test_acc` from 94.51% to at least 94.61%.
