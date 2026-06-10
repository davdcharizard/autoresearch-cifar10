# Brainstorm EXP-085
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  The current best recipe depends on static regional image/label mixing, so the next spatial-augmentation experiment should preserve `CUTMIX_ALPHA=1.0` and `CUTMIX_PROB=0.5`.
- **Torchvision RandomCrop padding modes** (`knowledge/references/torchvision-randomcrop-padding.md`)
  `RandomCrop` padding amount and padding mode are one-line, no-dependency spatial augmentation controls; reflection padding is the validated boundary-fill anchor in this repo.
- **EXP-081, EXP-082, EXP-083, and EXP-084 reports** (`reports/exp-report-081.md`, `reports/exp-report-082.md`, `reports/exp-report-083.md`, `reports/exp-report-084.md`)
  Padding 3 was a prior near miss under the older flip setting, while flip p=0.4 is now bracketed by worse p=0.35 and p=0.45 results.

No new external search was needed. The most relevant evidence is now project-specific: the current anchor is a calibrated CutMix plus reflection-crop plus flip-probability recipe, and the next question is whether a previously near-miss crop-strength change becomes useful under that anchor.

## Experimental History Review

- Current best is EXP-082 at `best_test_acc=94.36%` from commit `e859ac5`; the +0.10pp noise guard requires `best_test_acc >= 94.46%`.
- EXP-082 established `RandomHorizontalFlip(p=0.4)` as the current spatial anchor, improving over the prior 94.11% CutMix recipe by +0.25pp.
- EXP-083 and EXP-084 closed the local flip bracket: p=0.35 reached 94.17%, while p=0.45 reached 94.05%. Further broad flip-probability retuning is lower priority.
- EXP-081 reduced reflection crop padding from 4 to 3 and reached 94.18%, a sub-threshold near miss under the older flip setting. The result suggests crop jitter strength may still matter, but padding 3 alone was not enough before the p=0.4 flip anchor existed.
- Failed additive experiments warn against naive near-miss stacking, especially EXP-075's fan-out plus hard CutMix endpoint regression. The crop-padding-plus-flip combination is still defensible because EXP-082 explicitly changed the spatial augmentation anchor, not an unrelated knob.
- Closed or low-priority families remain fixed: CutMix alpha/probability/timing, label smoothing deviations, LR startup changes, batch-size deviations, weight averaging, schedule-only changes, policy augmentation, cutout, and further broad flip-probability brackets.

## Candidate Ideas

### 1. Crop Padding 3 on the Flip p=0.4 Anchor
**Summary**: Keep `transforms.RandomHorizontalFlip(p=0.4)` and reduce reflection `RandomCrop` padding from 4 to 3. Preserve unit-std normalization, CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last, and validation cadence.

**Reasoning**: EXP-081 showed padding 3 was a near miss before the flip-probability improvement, reaching 94.18% against a 94.11% baseline but missing the then-required 94.21% threshold. EXP-082 then found that the stronger spatial anchor is not default flip behavior but `p=0.4`. Combining the validated flip anchor with slightly weaker crop jitter tests a coherent spatial de-regularization interaction rather than randomly stacking unrelated near misses.

**Sources**: `reports/exp-report-081.md`; `reports/exp-report-082.md`; `reports/exp-report-084.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `knowledge/references/torchvision-randomcrop-padding.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: Prior near-miss combinations have regressed, and padding 3 could remove too much spatial coverage when paired with lower flip probability. The failure mode should be a clean no-improvement with no added infrastructure risk.

### 2. Fine Upper-Side Flip Bracket at 0.425
**Summary**: Change `transforms.RandomHorizontalFlip(p=0.4)` to `transforms.RandomHorizontalFlip(p=0.425)`, preserving all other anchor settings.

**Reasoning**: EXP-084 showed p=0.45 is worse, but it does not mathematically exclude a narrow optimum slightly above 0.4. A finer bracket could test whether 0.4 is exactly optimal or whether 0.425 restores a small amount of useful invariance without the over-regularization seen at 0.45.

**Sources**: `reports/exp-report-082.md`; `reports/exp-report-084.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: The expected effect is likely too small to clear the +0.10pp guard. This is scientifically tidy but less likely than a crop-strength interaction to produce a meaningful enough move.

### 3. Crop Padding 5 on the Flip p=0.4 Anchor
**Summary**: Keep `RandomHorizontalFlip(p=0.4)` and increase reflection crop padding from 4 to 5, preserving all other anchor settings.

**Reasoning**: If lowering flip probability improved accuracy by reducing one kind of spatial regularization, a larger crop jitter could compensate by increasing translation diversity while retaining the better flip anchor. This tests whether the useful direction is not simply less spatial augmentation overall, but a different balance between translation and horizontal-flip invariance.

**Sources**: `reports/exp-report-081.md`; `reports/exp-report-082.md`; `knowledge/references/torchvision-randomcrop-padding.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: Increasing crop padding likely over-regularizes or introduces too much reflected-border content. The mechanism is plausible but weaker than padding 3 because the strongest observed signal was a reduced, not increased, crop jitter near miss.

## Idea Evaluation

The crop-padding-3 interaction has the strongest evidence path. It combines a validated new anchor, `RandomHorizontalFlip(p=0.4)`, with a prior near miss in the same spatial-augmentation family. The mechanism is clear: reduce two different spatial regularization pressures just enough to improve clean late accuracy while preserving the CutMix recipe that currently defines the baseline.

The finer `p=0.425` flip bracket is lower risk but lower expected impact. Since p=0.45 regressed to 94.05% and p=0.35 regressed to 94.17%, p=0.4 is already locally bracketed. A 0.425 test may produce a narrow directional result, but the +0.10pp noise guard makes tiny scalar tweaks unattractive.

The padding-5 test is a useful conceptual control, but it has less support. EXP-081's signal came from less crop jitter, not more, and the current goal should prioritize candidates with enough expected effect to clear 94.46%.

## Chosen Idea
**Selected**: Crop Padding 3 on the Flip p=0.4 Anchor

**Why this idea**:
This is the best next spatial-augmentation interaction after closing the flip-probability bracket. It preserves every validated non-spatial anchor while testing whether the prior padding-3 near miss becomes meaningful when paired with the now-proven p=0.4 flip setting. It also avoids closed families such as CutMix strength/timing, label smoothing, LR startup, and batch-size changes.

**Hypothesis**:
If EXP-082's p=0.4 anchor still carries slightly too much crop-translation regularization, then reducing reflection crop padding from 4 to 3 while keeping `RandomHorizontalFlip(p=0.4)` will raise `best_test_acc` from 94.36% to at least 94.46%.
