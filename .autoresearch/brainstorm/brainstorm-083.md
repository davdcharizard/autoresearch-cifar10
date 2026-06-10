# Brainstorm EXP-083
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  The current anchor still depends on regional image/label mixing, so next experiments should preserve `CUTMIX_ALPHA=1.0` and `CUTMIX_PROB=0.5` unless directly bracketing CutMix again.
- **Torchvision RandomCrop padding modes** (`knowledge/references/torchvision-randomcrop-padding.md`)
  Reflection crop padding remains the validated crop-boundary behavior. EXP-081 showed reducing padding to 3 was only a sub-threshold near miss under the prior flip setting.
- **EXP-082 report** (`reports/exp-report-082.md`)
  Lowering horizontal flip probability to 0.4 improved the CutMix anchor to 94.36%, confirming mild spatial de-regularization can clear the noise guard.

No new external search was needed. The immediate next question is local hyperparameter bracketing around a just-validated setting, where project-specific evidence is higher signal than generic CIFAR augmentation guidance.

## Experimental History Review

- Current best is EXP-082 at `best_test_acc=94.36%` from commit `e859ac5`; the +0.10pp noise guard now requires `best_test_acc >= 94.46%`.
- EXP-082 changed only horizontal flip probability from 0.5 to 0.4 and produced a real +0.25pp gain. This creates a new spatial augmentation anchor worth bracketing before adding other changes.
- EXP-081 reduced reflection crop padding from 4 to 3 and reached 94.18%, a near miss under the old default flip setting. This suggests spatial de-regularization is promising, but the most direct next step is to bracket the successful flip knob.
- Added augmentation remains a weak family: Cutout, ColorJitter, RandAugment, and AutoAugment underperformed. The successful direction is reducing existing augmentation pressure, not adding new transforms.
- Closed families should remain fixed: CutMix alpha/probability/timing, label smoothing deviations, startup LR changes, batch-size deviations, weight averaging, and scheduler-only changes all have recurring failures or recent sub-threshold results.

## Candidate Ideas

### 1. Lower Horizontal Flip Probability to 0.35
**Summary**: Change `transforms.RandomHorizontalFlip(p=0.4)` to `transforms.RandomHorizontalFlip(p=0.35)`. Preserve reflection crop padding 4, CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer, LR schedule, batch size, seed, compile/channels-last, and validation cadence.

**Reasoning**: EXP-082 shows that reducing flip probability from 0.5 to 0.4 improves the anchor by a meaningful margin. Testing 0.35 follows the validated direction with a modest additional step, asking whether the optimum lies below 0.4 or whether 0.4 already balances invariance and over-regularization.

**Sources**: `reports/exp-report-082.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py` transform pipeline

**Estimated Effort**: low

**Risk Assessment**: Further reducing flips may remove too much useful CIFAR horizontal invariance and regress. The failure mode should be a clean no-improvement with no infrastructure risk.

### 2. Upper-Side Flip Bracket at 0.45
**Summary**: Change `transforms.RandomHorizontalFlip(p=0.4)` to `transforms.RandomHorizontalFlip(p=0.45)`, preserving all other anchor settings.

**Reasoning**: If 0.4 was beneficial but slightly too low, a 0.45 bracket could keep most of the de-regularization gain while restoring some useful invariance. This is the natural upper-side bracket for the new successful knob.

**Sources**: `reports/exp-report-082.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: Moving back toward 0.5 may partially undo the gain. It is a strong follow-up if 0.35 regresses, but less aligned with the current success direction than first testing the lower side.

### 3. Crop Padding 3 on the New Flip Anchor
**Summary**: Keep `RandomHorizontalFlip(p=0.4)` and reduce reflection `RandomCrop` padding from 4 to 3. Preserve all other settings.

**Reasoning**: EXP-081's padding 3 result was sub-threshold under flip p=0.5, but the new lower-flip anchor may shift the optimal crop strength. This would test whether two spatial de-regularization knobs interact positively.

**Sources**: `reports/exp-report-081.md`; `reports/exp-report-082.md`; `knowledge/references/torchvision-randomcrop-padding.md`

**Estimated Effort**: low

**Risk Assessment**: Prior near-miss combinations have regressed, and padding 3 alone did not clear threshold. This should wait until the flip-probability bracket is better understood.

## Idea Evaluation

The strongest next experiment is the lower-side flip bracket at 0.35. It directly follows the only recent intervention that cleared the noise guard and tests a single scalar around the new anchor. The expected effect is uncertain because CIFAR horizontal flips are useful, but this bracket gives the cleanest information about whether `p=0.4` is a local optimum or merely an intermediate point on a better lower-probability slope.

The 0.45 upper-side bracket is also valuable, but it is better as the next response if 0.35 underperforms. The crop-padding combination is plausible but less isolated; EXP-075 showed naive near-miss stacking can regress, so it should wait until the single-axis flip bracket is complete.

## Chosen Idea
**Selected**: Lower Horizontal Flip Probability to 0.35

**Why this idea**:
It is the most direct bracket of the newly validated spatial de-regularization mechanism. The change is one line, preserves every other successful anchor component, and answers whether further lowering flip frequency can move the new 94.36% baseline to at least 94.46%.

**Hypothesis**:
If the current anchor remains slightly over-regularized even at horizontal flip probability 0.4, then reducing flip probability to 0.35 will improve late clean accuracy enough to raise `best_test_acc` from 94.36% to at least 94.46%.
