# Brainstorm EXP-060
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup-beyond-erm.md`, source URL: https://arxiv.org/abs/1710.09412)
  Mixup trains on convex combinations of images and labels, giving a label-interpolation regularizer that can improve CIFAR generalization without modifying the model architecture or evaluation path. The knowledge note warns that fixed-budget overhead must be measured.
- **Existing autoresearch knowledge base** (`knowledge/README.md`)
  The standing knowledge base now covers mixup, RandAugment, cosine schedules, stochastic depth, SE attention, crop padding, and downsampling tweaks. Recent entries show broad architecture tweaks and isolated regularizers generally underperform the current anchor.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; EXP-060 must reach at least `94.07%` to count as improvement under the +0.10 percentage-point noise rule.
- The validated anchor is `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, full-run label smoothing 0.05, FP32 compile, channels-last, and once-per-epoch validation.
- Many isolated scalar and architecture perturbations are now closed: LR up/down, weight-decay brackets, batch-size deviations, second-drop/cosine schedules, BN momentum, no-decay parameter groups, residual BN down-scaling, stochastic depth, SE blocks, projection shortcuts, and average-pool option-A downsampling.
- EXP-055 is the strongest recent regularization miss: mild `MIXUP_ALPHA=0.1` completed cleanly, reached the 21k LR drop, and peaked at 93.85%, only 0.12 pp below the baseline. Its implementation preserved `label_smoothing=0.05` inside each mixup target loss, which may have compounded target softening.
- Goal learnings now say direct mixup retries are low priority, but revisiting target regularization with a distinct coupled mechanism is still allowed. Removing label smoothing only inside mixup is such a coupled mechanism because mixup already softens labels through interpolation.
- EXP-059 closes isolated shortcut-transition smoothing: average-pool option-A downsampling preserved throughput and LR timing but plateaued at 93.42%. This reduces the priority of further shortcut-only experiments.

## Candidate Ideas

### 1. Mixup Without Additional Label Smoothing
**Summary**: Reintroduce mild `MIXUP_ALPHA=0.1`, but compute the weighted two-target mixup loss with `label_smoothing=0.0` instead of `0.05`. Keep all non-mixup anchor settings unchanged. Add a startup print such as `Mixup alpha: 0.1, label smoothing in mixup loss: 0.0` for verification.

**Reasoning**: EXP-055 showed mixup is executable and relatively close at 93.85%, but it combined interpolation targets with smoothed hard-label losses. That may over-soften supervision under the already regularized 2e-4/label-smoothed anchor. Removing smoothing only inside the mixup loss tests a distinct regularization balance while preserving mixup's core mechanism and avoiding another scalar bracket of alpha.

**Sources**: `knowledge/papers/mixup-beyond-erm.md`; `reports/exp-report-055.md`; goal-learnings entry "Mild mixup alpha 0.1 remains below the current anchor"; label-smoothing failure pattern in `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: medium

**Risk Assessment**: The run may still underperform because mixup overhead and interpolation remain enough to soften or slow final fitting. It also intentionally changes the anchor's label-smoothing behavior during mixed batches, but within a targeted coupled regularization test rather than an isolated smoothing deviation.

### 2. Final Classifier Dropout
**Summary**: Add a small dropout layer before the final fully connected classifier, likely `p=0.1`, active only during training. Preserve the residual body and all current optimizer, augmentation, schedule, smoothing, and width settings.

**Reasoning**: This is a low-overhead regularizer distinct from residual stochastic depth and label-space regularization. It targets overconfident late classifier fitting without perturbing all residual blocks. However, isolated regularization has a weak recent record: stochastic depth, RandAugment, ColorJitter, and label-smoothing deviations all missed the anchor.

**Sources**: Recent failed-approach entries in `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `reports/exp-report-054.md` for residual stochastic depth; `reports/exp-report-044.md` and `reports/exp-report-050.md` for augmentation regularizers.

**Estimated Effort**: low

**Risk Assessment**: It may simply underfit the already well-regularized 2e-4 anchor. The expected effect size is likely small, and a small drop in classifier fit could be enough to miss 94.07%.

### 3. Stage-3-Only SE
**Summary**: Add Squeeze-and-Excitation gates only to the final residual stage instead of every block. This keeps early feature extraction untouched and reduces the parameter/compute overhead compared with EXP-058.

**Reasoning**: EXP-058 rejected all-block SE, but it does not fully rule out a stage-limited version. Final-stage-only gating might calibrate higher-level channels while avoiding early-stage overhead. Still, EXP-058 reached only 93.71% and added per-block gating cost, so this is more speculative than fixing the mixup/smoothing interaction.

**Sources**: `knowledge/papers/squeeze-and-excitation-networks.md`; `reports/exp-report-058.md`; goal-learnings entry "SE channel attention underperforms the current block".

**Estimated Effort**: medium

**Risk Assessment**: This is close to a fresh negative result and may again lose useful step budget or add an unnecessary representation mechanism. It also requires passing stage context into block construction or adding a block-level flag, increasing implementation surface.

## Idea Evaluation

Mixup Without Additional Label Smoothing has the strongest project-specific evidence. EXP-055 already established that mild mixup is feasible, reaches the LR milestone, and lands closer to the anchor than most recent no-improvements. The unresolved mechanism is precise: mixup supplies soft labels by interpolation, while the EXP-055 loss also applied `label_smoothing=0.05` to both endpoint targets. Removing that extra smoothing is not a direct retry; it tests whether the previously close run was over-regularized.

Final Classifier Dropout is cheaper but less grounded. It is distinct from residual stochastic depth, yet the recent trajectory repeatedly shows isolated regularizers lagging the anchor. Stage-3-Only SE has a plausible narrower implementation, but EXP-058's all-block result and EXP-059's shortcut result both argue against more immediate architecture tweaks unless they have a much stronger mechanism.

The lead candidate is therefore the coupled mixup-loss adjustment. It targets one of the few recent near-misses with a concrete implementation change, while preserving the validated optimizer, schedule, width, augmentation, and evaluation harness.

## Chosen Idea
**Selected**: Mixup Without Additional Label Smoothing

**Why this idea**:
It has the clearest path from evidence to a new outcome: EXP-055 was close but may have combined two target-softening mechanisms. The proposed change preserves mixup alpha and all anchor settings except the smoothing used inside the mixup loss, directly testing whether removing compounded soft labels restores enough final fit to clear 94.07%.

**Hypothesis**:
Using `MIXUP_ALPHA=0.1` with unsmoothed endpoint cross-entropy will keep the step-21000 LR drop reachable and improve on EXP-055's 93.85%, potentially reaching at least `94.07%` by reducing over-softening while retaining mixup's regularization benefit.
