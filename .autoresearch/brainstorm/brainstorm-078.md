# Brainstorm EXP-078
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Wide residual networks** (`knowledge/papers/wide-residual-networks.md`)
  CIFAR residual models can benefit from residual-block topology and width/depth choices, but fixed-budget experiments need to preserve enough step coverage and avoid broad capacity changes unless the mechanism is strong.
- **ResNet downsampling tweaks** (`knowledge/papers/resnet-downsampling-tweaks.md`)
  Residual shortcut/downsampling details can affect accuracy, but EXP-059 and EXP-077 now show that isolated average-pool transition smoothing is not enough for the current CutMix anchor.
- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  The current best already uses probabilistic regional mixing. New candidates should preserve `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, and endpoint `label_smoothing=0.05` unless explicitly testing an interaction.

No new external searches were added in this pass; the existing knowledge base already covers the relevant CIFAR architecture, scheduling, and augmentation mechanisms, and the last several experiments provide stronger local evidence than generic references.

## Experimental History Review

- Current best remains EXP-064 at `best_test_acc=94.11%` from commit `1119ff8`; the active +0.10pp noise guard requires `best_test_acc >= 94.21%`.
- EXP-065 through EXP-077 have bracketed nearby CutMix strength, endpoint smoothing, classifier-head initialization/dropout, Conv2d fan-out initialization, policy augmentation, input normalization, and two transition downsampling smoothing variants. None cleared the 94.21% threshold.
- Recent near-threshold positive signals were small and non-additive: EXP-072 fan-out Conv2d initialization reached 94.16%, EXP-073 short clean warmup reached 94.14%, and EXP-074 hard CutMix endpoints reached 94.17%, but EXP-075 and EXP-076 showed those signals do not compose cleanly.
- Failed-approach memory now strongly discourages more isolated label-smoothing changes, static CutMix alpha/probability brackets, classifier-head-only tweaks, transition downsampling smoothing, batch-size changes, scalar LR or weight-decay retuning, SE gates, policy augmentation, EMA/SWA, and second-drop/cosine schedule-only probes.
- The main untested architecture family is block topology. Pre-activation changes normalization/activation placement throughout the residual block, which is distinct from projection shortcuts, SE gates, reduced depth, width changes, and average-pool transition tweaks.

## Candidate Ideas

### 1. Pre-Activation BasicBlock
**Summary**: Convert `BasicBlock` from post-activation ResNet-v1 style to a CIFAR pre-activation residual block: apply BatchNorm/ReLU before each convolution and remove the final ReLU after residual addition. Preserve the same stage widths, number of blocks, option-A shortcut, CutMix anchor, optimizer, LR schedule, transforms, batch size, validation cadence, and evaluation harness.

**Reasoning**: This is a distinct architecture/topology test rather than another scalar regularizer. Pre-activation changes gradient flow and residual identity behavior across the whole network while keeping parameter count and the fixed benchmark intact. It could improve the current plateau by making residual optimization smoother without changing CutMix or the time budget. The risk is higher than a one-line hyperparameter change, but most safer local families are now bracketed.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `train.py` `BasicBlock`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `reports/exp-report-077.md`

**Estimated Effort**: medium

**Risk Assessment**: The topology change may disrupt the tuned ResNet-20 recipe or slightly affect compile behavior. It should still preserve parameter count, throughput, and the first LR drop if implemented narrowly. Worst case is a valid no-improvement, not an invalid benchmark change.

### 2. Short CutMix Probability Ramp
**Summary**: Preserve CutMix alpha, endpoint smoothing, and static long-run `CUTMIX_PROB=0.5`, but linearly ramp the active CutMix probability from 0.25 to 0.5 over the first 1000 optimizer steps. This tests whether the earliest representation phase benefits from less regional label mixing without fully disabling CutMix.

**Reasoning**: EXP-073's clean warmup produced a sub-threshold positive signal, suggesting early CutMix exposure may be slightly harsh, while EXP-065 showed static `p=0.25` is not better long-run. A ramp is distinct from both: it keeps CutMix present early and restores the validated anchor quickly. It is also cheaper than architecture changes.

**Sources**: `reports/exp-report-073.md`; `reports/exp-report-065.md`; `knowledge/papers/cutmix-regularization.md`; `train.py` CutMix loop

**Estimated Effort**: low

**Risk Assessment**: This sits near already failed temporal/strength CutMix variants. It may produce another 94.1x near-miss rather than a true +0.10pp improvement. It also adds one training-loop branch that must not alter validation or the fixed schedule.

### 3. Narrow Layer3 Width Rebalance
**Summary**: Keep depth and the first two stage widths unchanged but make only the final stage slightly narrower, for example `STAGE_WIDTHS=(28, 56, 104)`, to trade a small amount of capacity for more optimizer steps under the same CutMix anchor.

**Reasoning**: Full width increases and reduced-depth variants failed, but the current CutMix anchor may be near a throughput/capacity boundary. A small final-stage reduction could increase step coverage and low-LR refinement while preserving most representation capacity. This is different from EXP-062's shallower wider model and from earlier width increases.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `reports/exp-report-062.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py` `STAGE_WIDTHS`

**Estimated Effort**: low

**Risk Assessment**: Width/capacity changes have a mixed history and often underperform once the 28/56/112 anchor is established. A smaller final stage may simply reduce representational strength and fail below baseline despite more steps.

## Idea Evaluation

Pre-activation has the best separation from recently failed mechanisms. It is a real block-topology experiment, not another variant of CutMix strength, endpoint smoothing, initialization, downsampling smoothing, attention, or schedule retuning. Its mechanism is clear: change residual optimization and activation placement while keeping the benchmark and anchor recipe intact. It also keeps parameter count close to unchanged and should not require new dependencies or evaluation changes.

The CutMix probability ramp is attractive because it builds on a recent near-miss, but it remains very close to the static probability bracket and clean-warmup family. Given the +0.10pp threshold, another small temporal CutMix tweak is likely to land in the same sub-threshold band unless the ramp timing is unusually well matched. It is a good fallback but not the strongest next test.

The layer3 width rebalance is simple, yet history around capacity changes is weak: widening, final-stage-only widening, shallow-wide designs, and batch-size/throughput changes have repeatedly underperformed. Reducing final-stage capacity may buy steps, but EXP-064's strength appears to come from the current 28/56/112 capacity plus CutMix recipe rather than needing less compute.

## Chosen Idea
**Selected**: Pre-Activation BasicBlock

**Why this idea**:
It is the clearest remaining architecture lever that has not already been bracketed by the experiment history. It preserves the current CutMix anchor and fixed benchmark while testing whether the residual block's optimization topology, rather than downsampling, attention, classifier, or scalar regularization, is the missing source of improvement.

**Hypothesis**:
If the current post-activation block is slightly limiting residual optimization under the CutMix anchor, then switching to pre-activation blocks while preserving all other anchor settings will improve post-drop convergence enough to raise `best_test_acc` from 94.11% to at least 94.21%.
