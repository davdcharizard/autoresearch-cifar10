# Brainstorm EXP-077
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **ResNet downsampling tweaks** (`knowledge/papers/resnet-downsampling-tweaks.md`)
  ResNet shortcut/downsampling details can affect accuracy; low-overhead transition changes are valid architecture levers when they preserve the benchmark and stay inside `train.py`.
- **Wide residual networks** (`knowledge/papers/wide-residual-networks.md`)
  CIFAR residual models can benefit from architecture changes, but this repo's fixed 300s budget makes full width/depth changes risky unless step budget is preserved.
- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  The current best already uses regional mixing; new ideas should preserve `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, and endpoint smoothing unless the experiment specifically has strong evidence for changing them.

## Experimental History Review

- Current best remains EXP-064: probabilistic CutMix reached `best_test_acc=94.11%`; the active noise guard requires `best_test_acc >= 94.21%`.
- EXP-065 through EXP-076 have bracketed most nearby CutMix, label-smoothing, classifier-head, and initialization variants. Recent near-misses above baseline but below threshold were EXP-072 fan-out Conv2d init at 94.16%, EXP-073 clean warmup at 94.14%, and EXP-074 hard CutMix endpoints at 94.17%; EXP-075 and EXP-076 showed those near-miss directions do not compose or transfer.
- Recurring failures now discourage isolated label-smoothing deviations, static CutMix alpha/probability changes, Conv2d fan-out retries, classifier-head-only tweaks, batch-size deviations, scalar LR changes, and scalar weight-decay retuning.
- Architecture remains plausible only if localized and throughput-aware. Full capacity changes and shortcut-only smoothing have underperformed, but the residual branch's stride-2 transition itself has not been isolated under the current CutMix anchor.

## Candidate Ideas

### 1. Anti-Aliased Residual Downsample
**Summary**: For BasicBlock transitions with `stride=2`, average-pool the residual branch input before `conv1` and set that transition convolution's stride to 1. Keep the option-A shortcut exactly as it is, with strided slicing plus zero-channel padding. This tests whether smoothing the learned residual path's downsampling improves transition features without adding parameters or changing CutMix, optimizer, schedule, labels, batch size, validation cadence, or evaluation.

**Reasoning**: EXP-059 tested average-pooling the shortcut and failed, but the current residual branch still performs stride-2 sampling inside a 3x3 convolution. Anti-aliased downsampling targets a different transition path: it reduces spatial aliasing before the learned residual transform while preserving the proven parameter-free shortcut. This is a localized architecture change with no new dependencies and likely small overhead.

**Sources**: `knowledge/papers/resnet-downsampling-tweaks.md`; `reports/exp-report-059.md`; `train.py` `BasicBlock`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`

**Estimated Effort**: low

**Risk Assessment**: Average pooling may blur useful high-frequency CIFAR details or reduce throughput enough to cost steps. Worst case is a clean no-improvement with the first LR drop still reached.

### 2. Pre-Activation BasicBlock
**Summary**: Convert `BasicBlock` from post-activation ResNet-v1 style to a pre-activation layout using BN/ReLU before each convolution and no final ReLU after residual addition. Keep the same stage widths, block count, option-A shortcuts, CutMix anchor, optimizer, schedule, and time budget.

**Reasoning**: Pre-activation residual blocks can improve gradient flow in residual networks and are a distinct architecture lever from width scaling, SE attention, projection shortcuts, and final-head changes. It changes representation dynamics without directly tuning the already-bracketed CutMix and optimizer settings.

**Sources**: `train.py` `BasicBlock`; `knowledge/papers/wide-residual-networks.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`

**Estimated Effort**: medium

**Risk Assessment**: This is a larger architecture change than EXP-077 needs, may alter BatchNorm parameterization and require final activation details, and could reduce comparability if implemented too broadly. It is better kept as a later architecture experiment with a careful plan.

### 3. Short LR Warmup Into CutMix Anchor
**Summary**: Keep all CutMix and label-smoothing settings unchanged, but linearly ramp LR from a small value to 0.1 over the first 500 optimizer steps before returning to the existing `MultiStepLR` behavior. This tests whether the current recipe loses accuracy from too-aggressive early optimization on mixed labels.

**Reasoning**: EXP-073 showed a 2000-step clean data warmup reached 94.14%, suggesting the very early CutMix phase can matter. An LR warmup is a different early-stability mechanism that preserves CutMix exposure, parameter count, and augmentation semantics.

**Sources**: `reports/exp-report-073.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py` scheduler setup

**Estimated Effort**: medium

**Risk Assessment**: Initial LR deviations and schedule-only changes have a poor record in this project, and LR warmup may simply undertrain within the fixed 300s budget. It is a lower-priority fallback because it sits near already-failed schedule families.

## Idea Evaluation

Anti-aliased residual downsampling has the cleanest separation from recently failed local families. It is not another CutMix bracket, label-smoothing tweak, classifier-head change, or Conv2d initialization retry. It also avoids full capacity scaling, which has a strong fixed-budget failure pattern. The mechanism is concrete: reduce aliasing in the learned stride-2 residual branch while preserving the validated option-A shortcut and the CutMix anchor.

Pre-activation blocks may have higher upside as an architecture change, but the implementation has more degrees of freedom and a larger behavioral surface. Under the current autopilot loop, the safer next step is a narrower transition-path experiment before rewriting the block topology. LR warmup is easy to motivate from EXP-073, but it lives close to schedule-only and early-training families that have repeatedly stayed below threshold.

## Chosen Idea
**Selected**: Anti-Aliased Residual Downsample

**Why this idea**:
It is a localized architecture experiment that remains distinct from the now-bracketed CutMix, classifier-head, label-smoothing, and initialization directions. It directly targets a still-untested transition mechanism in `BasicBlock` and should preserve parameter count and most throughput, making it a clean validity-preserving test under the 94.21% threshold.

**Hypothesis**:
If stride-2 convolutional downsampling in the residual branch loses useful CIFAR transition information through aliasing, then average-pooling before the learned transition convolution will improve post-drop features enough to raise `best_test_acc` from 94.11% to at least 94.21%.
