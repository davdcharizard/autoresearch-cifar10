# Brainstorm EXP-086
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Torchvision RandomCrop padding modes** (`knowledge/references/torchvision-randomcrop-padding.md`)
  `RandomCrop` padding amount is a one-line, no-dependency spatial augmentation control; reflection padding is already validated in this repo.
- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  The current best recipe still depends on static CutMix region/label mixing, so the next spatial experiment should preserve `CUTMIX_ALPHA=1.0` and `CUTMIX_PROB=0.5`.
- **Recent project reports** (`reports/exp-report-081.md`, `reports/exp-report-082.md`, `reports/exp-report-085.md`)
  Padding 3 was only a near miss before flip p=0.4, then became a true improvement when coupled with that lower flip-probability anchor.

No new external search was needed. The most relevant evidence is now project-specific: EXP-085 validated a coordinated mild spatial de-regularization interaction.

## Experimental History Review

- Current best is EXP-085 at `best_test_acc=94.51%` from commit `83d4e94`; the +0.10pp noise guard requires `best_test_acc >= 94.61%` for EXP-086 to count as an improvement.
- EXP-085 improved the EXP-082 spatial anchor by reducing reflection crop padding from 4 to 3 while keeping `RandomHorizontalFlip(p=0.4)`.
- EXP-083 and EXP-084 bracketed flip p=0.4 under padding 4: p=0.35 reached 94.17%, p=0.45 reached 94.05%. Fine flip retuning is not closed under padding 3, but the expected effect is likely small.
- EXP-081 showed isolated padding 3 was only a sub-threshold gain before the flip p=0.4 anchor. The new success means the productive family is not "padding 3 alone", but coupled crop/flip spatial de-regularization.
- Closed or low-priority families remain fixed: CutMix alpha/probability/timing, label smoothing deviations, LR startup changes, batch-size deviations, weight averaging, schedule-only changes, policy augmentation, cutout, broad architecture changes, and broad flip-probability brackets.

## Candidate Ideas

### 1. Crop Padding 2 on the Padding-3 / Flip p=0.4 Anchor
**Summary**: Keep `RandomHorizontalFlip(p=0.4)` and reduce reflection `RandomCrop` padding from 3 to 2. Preserve unit-std normalization, CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last, and validation cadence.

**Reasoning**: EXP-085 showed that reducing crop jitter from 4 to 3 is beneficial only after the flip p=0.4 anchor exists. A padding-2 bracket directly tests whether the useful direction has more headroom or whether padding 3 is already the lower spatial-jitter edge. This is the cleanest one-axis follow-up to the new baseline and has the same low infrastructure risk as EXP-085.

**Sources**: `reports/exp-report-085.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `knowledge/references/torchvision-randomcrop-padding.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: Padding 2 may under-regularize translation invariance and reduce crop diversity enough to regress. The likely failure mode is a clean no-improvement, not a crash or invalid run.

### 2. Fine Lower-Side Flip Bracket at p=0.375 Under Padding 3
**Summary**: Keep reflection crop padding 3 and lower `RandomHorizontalFlip(p=0.4)` to `p=0.375`, preserving all other EXP-085 anchor settings.

**Reasoning**: The old p=0.35 failure happened under padding 4, not padding 3. Reducing crop jitter may shift the optimal flip balance slightly, and p=0.375 is a smaller move than the failed p=0.35 test. This tests whether the new anchor still wants less spatial regularization, but through flip probability rather than crop padding.

**Sources**: `reports/exp-report-083.md`; `reports/exp-report-085.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: The expected effect may be too small to clear the +0.10pp guard, and p=0.35 already warned that too little flip probability under-regularizes. This is a sensible bracket, but weaker than the direct crop-padding bracket.

### 3. Crop Padding 5 Control on the Flip p=0.4 Anchor
**Summary**: Keep `RandomHorizontalFlip(p=0.4)` and increase reflection crop padding from 3 to 5, preserving all other anchor settings.

**Reasoning**: EXP-085 supports weaker crop jitter, but a padding-5 control would test the opposite hypothesis: lower flip probability might benefit from compensating translation diversity. This is useful scientifically, but the evidence is weaker because the recent positive movement came from reducing, not increasing, crop padding.

**Sources**: `reports/exp-report-085.md`; `reports/exp-report-081.md`; `knowledge/references/torchvision-randomcrop-padding.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: Padding 5 likely over-regularizes and introduces too much reflected-border content. It is a clean test but lower expected value than continuing the successful de-regularization bracket.

## Idea Evaluation

Crop padding 2 has the strongest evidence path because it directly brackets the newly successful EXP-085 mechanism. It asks whether the new 94.51% anchor is still over-regularized by translation jitter, and it changes only one scalar in an already-validated spatial family.

The p=0.375 flip bracket is plausible because padding 3 may shift the crop/flip balance, but the old p=0.35 result warns that lowering flip probability too far can under-regularize. Since the +0.10pp guard now requires 94.61%, a tiny flip-probability adjustment may also be too small.

Padding 5 is the cleanest control for the opposite direction, but it has the weakest support. The most recent successful signal is reduced spatial augmentation strength, so increasing crop jitter should wait until the lower-side bracket is tested.

## Chosen Idea
**Selected**: Crop Padding 2 on the Padding-3 / Flip p=0.4 Anchor

**Why this idea**:
This is the most direct continuation of EXP-085's successful mechanism. It preserves the validated CutMix, schedule, optimizer, architecture, and flip settings while testing whether crop jitter can be reduced one more notch without losing generalization.

**Hypothesis**:
If the EXP-085 anchor still has slightly too much crop-translation regularization, reducing reflection crop padding from 3 to 2 while keeping `RandomHorizontalFlip(p=0.4)` will raise `best_test_acc` from 94.51% to at least 94.61%.
