# Brainstorm EXP-087
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Torchvision RandomCrop padding modes** (`knowledge/references/torchvision-randomcrop-padding.md`)
  `RandomCrop` padding is the relevant no-dependency spatial knob, but EXP-086 now shows further reducing reflection padding below 3 is harmful under the current anchor.
- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  The current best recipe still depends on static CutMix region/label mixing, so EXP-087 should preserve `CUTMIX_ALPHA=1.0` and `CUTMIX_PROB=0.5`.
- **Recent project reports** (`reports/exp-report-083.md`, `reports/exp-report-084.md`, `reports/exp-report-085.md`, `reports/exp-report-086.md`)
  The current spatial optimum is a coupled padding-3 / flip-p=0.4 anchor. The broad flip bracket under padding 4 failed on both sides, but fine flip brackets under padding 3 remain untested.

No new external search was needed. The strongest evidence is project-specific, and the next decision is a local bracket around the newly validated spatial anchor.

## Experimental History Review

- Current best is EXP-085 at `best_test_acc=94.51%` from commit `83d4e94`; the +0.10pp noise guard requires `best_test_acc >= 94.61%` for EXP-087 to count as an improvement.
- EXP-085 validated coordinated mild spatial de-regularization: reflection crop padding 3 plus `RandomHorizontalFlip(p=0.4)` reached 94.51%.
- EXP-086 closed the simple lower crop-padding direction: padding 2 with flip p=0.4 peaked at 94.22%, below baseline and threshold.
- EXP-083 and EXP-084 bracketed flip p=0.4 under padding 4: p=0.35 reached 94.17%, and p=0.45 reached 94.05%. Those failures warn against large flip-probability moves, but they do not close smaller p=0.375 / p=0.425 tests under padding 3.
- Closed or low-priority families remain fixed: CutMix alpha/probability/timing, label smoothing deviations, LR startup changes, batch-size deviations, weight averaging, schedule-only changes, policy augmentation, cutout, broad architecture changes, crop padding below 3, and broad flip-probability brackets.

## Candidate Ideas

### 1. Fine Upper Flip Bracket p=0.425 Under Padding 3
**Summary**: Keep reflection crop padding 3 and increase `RandomHorizontalFlip(p=0.4)` to `p=0.425`. Preserve unit-std normalization, CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last, fixed 300s budget, and validation cadence.

**Reasoning**: EXP-086 indicates the current anchor should not lose more spatial augmentation strength on the crop axis. A small increase in flip probability may restore a little spatial invariance while staying below the failed p=0.45 broad bracket. Because padding 3 already reduces crop translation compared with the old padding-4 anchor, p=0.425 is a more balanced upper-side test than simply returning to p=0.45.

**Sources**: `reports/exp-report-084.md`; `reports/exp-report-085.md`; `reports/exp-report-086.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: The effect may be too small to clear the +0.10pp guard, or p=0.425 may restore the over-regularization seen at p=0.45. Failure should be a clean no-improvement rather than a crash.

### 2. Fine Lower Flip Bracket p=0.375 Under Padding 3
**Summary**: Keep reflection crop padding 3 and reduce `RandomHorizontalFlip(p=0.4)` to `p=0.375`, preserving all other EXP-085 anchor settings.

**Reasoning**: The old p=0.35 failure happened under padding 4, not padding 3. If crop padding 3 changes the crop/flip interaction enough, a smaller downward flip adjustment might still improve the anchor without falling as far as p=0.35.

**Sources**: `reports/exp-report-083.md`; `reports/exp-report-085.md`; `reports/exp-report-086.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: EXP-086 already suggests further spatial de-regularization is risky, and p=0.35 previously regressed. This candidate is still valid, but its mechanism is weaker than the upper bracket.

### 3. Tiny Coupled Clean Label Smoothing Nudge to 0.04 on Padding-3 / p=0.4 Anchor
**Summary**: Keep the EXP-085 spatial anchor and lower clean-batch label smoothing from 0.05 to 0.04 while preserving CutMix endpoint smoothing at 0.05.

**Reasoning**: Spatial de-regularization improved the anchor, so a very small clean-label smoothing nudge might reduce late underfitting while keeping CutMix labels stable. This differs from prior broader label-smoothing deviations by making a smaller change and leaving CutMix endpoints unchanged.

**Sources**: `reports/exp-report-033.md`; `reports/exp-report-037.md`; `reports/exp-report-057.md`; `reports/exp-report-085.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: Label-smoothing deviations are a high-importance recurring failure family, so this is lower priority despite being simple. It risks re-opening a mostly closed mechanism for a marginal expected gain.

## Idea Evaluation

The fine upper flip bracket has the best balance of evidence and mechanism. EXP-086 showed that simply weakening crop translation further is harmful, so a small amount of restored flip invariance is the most defensible next local bracket. It also tests the current spatial anchor directly while avoiding the failed broad p=0.45 move.

The fine lower flip bracket remains a legitimate symmetric test, but its mechanism is less supported after EXP-086. Both padding 2 and p=0.35 point toward under-regularization when spatial invariance is reduced too far.

The label-smoothing nudge is a possible coupled adjustment, but prior smoothing variants are heavily negative. It should wait until the remaining high-signal spatial brackets are exhausted.

## Chosen Idea
**Selected**: Fine Upper Flip Bracket p=0.425 Under Padding 3

**Why this idea**:
This is the cleanest follow-up after EXP-086. It preserves the validated padding-3 anchor while testing whether a very small amount of restored horizontal invariance can improve beyond 94.51% without reverting to the failed p=0.45 setting.

**Hypothesis**:
If EXP-085's padding-3 anchor is slightly under-regularized on horizontal invariance after the crop-padding reduction, increasing `RandomHorizontalFlip` from p=0.4 to p=0.425 will raise `best_test_acc` from 94.51% to at least 94.61%.
