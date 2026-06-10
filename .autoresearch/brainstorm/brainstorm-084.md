# Brainstorm EXP-084
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  The current best recipe depends on static regional image/label mixing, so the next augmentation experiment should preserve `CUTMIX_ALPHA=1.0` and `CUTMIX_PROB=0.5`.
- **Torchvision RandomCrop padding modes** (`knowledge/references/torchvision-randomcrop-padding.md`)
  Reflection crop padding remains the validated crop-boundary anchor; isolated padding reduction was only a sub-threshold near miss before the new flip-probability anchor.
- **EXP-082 and EXP-083 reports** (`reports/exp-report-082.md`, `reports/exp-report-083.md`)
  Lowering horizontal flip probability to 0.4 improved the anchor to 94.36%, while lowering further to 0.35 regressed to 94.17%. This brackets the lower side and points to testing the upper side.

No new external search was needed. The immediate decision is a local hyperparameter bracket around a project-specific successful augmentation setting, where the experiment history is more relevant than generic augmentation guidance.

## Experimental History Review

- Current best is EXP-082 at `best_test_acc=94.36%` from commit `e859ac5`; the +0.10pp noise guard requires `best_test_acc >= 94.46%`.
- EXP-082 showed that reducing horizontal flip probability from 0.5 to 0.4 was a real improvement, clearing the prior threshold by +0.15pp and raising the baseline by +0.25pp.
- EXP-083 showed that reducing further to 0.35 regressed to 94.17%, below the current baseline. This suggests 0.4 is not simply an intermediate point on a lower-is-better slope.
- The natural remaining single-axis bracket is the upper side, `p=0.45`, to test whether EXP-082 was slightly under-flipped or whether 0.4 is the local optimum.
- Adding new augmentation remains weak: Cutout, ColorJitter, RandAugment, and AutoAugment underperformed. The useful recent direction is calibrating existing augmentation pressure, not adding new transforms.
- Closed or low-priority families remain fixed: CutMix alpha/probability/timing, label smoothing deviations, LR startup changes, batch-size deviations, weight averaging, schedule-only changes, and lower flip probabilities.

## Candidate Ideas

### 1. Upper-Side Horizontal Flip Bracket at 0.45
**Summary**: Change `transforms.RandomHorizontalFlip(p=0.4)` to `transforms.RandomHorizontalFlip(p=0.45)`. Preserve reflection crop padding 4, CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer, LR schedule, batch size, seed, compile/channels-last, and validation cadence.

**Reasoning**: EXP-082 proved that reducing flip pressure from 0.5 to 0.4 improves the CutMix anchor, while EXP-083 showed that reducing further to 0.35 regresses. The highest-signal next test is the upper side of that local bracket: 0.45 may restore enough useful horizontal invariance while keeping most of the de-regularization gain from moving away from the default 0.5.

**Sources**: `reports/exp-report-082.md`; `reports/exp-report-083.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: Moving toward 0.5 may partially undo the EXP-082 gain and land between 94.17 and 94.36. The failure mode is a clean no-improvement with no added infrastructure risk.

### 2. Finer Upper-Side Horizontal Flip Bracket at 0.425
**Summary**: Change `transforms.RandomHorizontalFlip(p=0.4)` to `transforms.RandomHorizontalFlip(p=0.425)`, preserving all other anchor settings.

**Reasoning**: If 0.4 is close to the optimum, a smaller upper-side move may be less likely to over-correct than 0.45. This is a finer local search step that could capture a narrow optimum just above 0.4.

**Sources**: `reports/exp-report-082.md`; `reports/exp-report-083.md`; `train.py`

**Estimated Effort**: low

**Risk Assessment**: The expected effect size may be too small to clear the +0.10pp noise guard. It is scientifically precise but may be less likely than 0.45 to move the metric by the required margin.

### 3. Crop Padding 3 on the Flip p=0.4 Anchor
**Summary**: Keep `RandomHorizontalFlip(p=0.4)` and reduce reflection `RandomCrop` padding from 4 to 3. Preserve all other settings.

**Reasoning**: EXP-081's padding 3 result reached 94.18% under the older flip setting, and EXP-082 showed flip p=0.4 is a stronger spatial anchor. Combining the validated flip setting with the near-miss crop reduction could expose a beneficial interaction between two spatial de-regularization knobs.

**Sources**: `reports/exp-report-081.md`; `reports/exp-report-082.md`; `knowledge/references/torchvision-randomcrop-padding.md`

**Estimated Effort**: low

**Risk Assessment**: Prior near-miss combinations have regressed, and this changes two spatial pressures relative to the original baseline. It is less isolated than completing the flip-probability bracket first.

## Idea Evaluation

The upper-side `p=0.45` bracket has the clearest evidence path. EXP-082 validated lower flip pressure as a real improvement, while EXP-083 closed the lower side by showing `p=0.35` regresses. Testing `p=0.45` directly asks whether 0.4 is too low or whether it is the local optimum. The mechanism is clear: preserve useful class-invariant horizontal flips while still reducing over-regularization relative to default 0.5.

The finer `p=0.425` bracket is defensible, but its expected effect is likely smaller. Because the goal requires at least +0.10pp over the current baseline, a very fine scalar adjustment may be too noisy to count even if directionally correct. It becomes more attractive if `p=0.45` underperforms but remains close to 94.36.

The crop-padding interaction remains plausible, but EXP-075 showed that combining near-misses can regress. It should wait until the single-axis flip bracket is closed, because otherwise a failure would be harder to attribute.

## Chosen Idea
**Selected**: Upper-Side Horizontal Flip Bracket at 0.45

**Why this idea**:
It is the most direct completion of the successful flip-probability bracket. EXP-082 established `p=0.4` as the current best anchor, and EXP-083 showed the lower side is worse. The upper-side bracket has a simple one-line implementation, preserves all validated anchor settings, and cleanly tests whether restoring a little horizontal invariance can reach at least 94.46%.

**Hypothesis**:
If `p=0.4` removed slightly too much useful horizontal flip invariance, then increasing to `RandomHorizontalFlip(p=0.45)` will improve late clean accuracy enough to raise `best_test_acc` from 94.36% to at least 94.46%.
