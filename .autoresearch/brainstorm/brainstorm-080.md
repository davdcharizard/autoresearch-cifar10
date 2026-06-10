# Brainstorm EXP-080
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  The current best anchor is already a CutMix recipe, but EXP-065 through EXP-079 now show static strength brackets and early CutMix timing tweaks do not clear the +0.10pp threshold.
- **SGD scheduling** (`knowledge/papers/sgdr-cosine-schedule.md`)
  Prior cosine and second-drop schedule variants underperformed locally, but those changed late training behavior; a short startup warmup is a different early-stability mechanism.
- **Wide residual networks** (`knowledge/papers/wide-residual-networks.md`)
  Local capacity/topology history suggests broad residual rewrites are weak under the fixed budget, but a very small width rebalance remains a low-cost way to test the step-budget/capacity tradeoff.
- **PyTorch throughput tools** (`knowledge/references/pytorch-throughput-tools.md`)
  The current FP32 compile/channels-last execution path remains the best validated throughput mode and should be preserved for the next experiment.

No new external searches were added. The local experiment trajectory is now more informative than generic CIFAR sources for this narrow benchmark.

## Experimental History Review

- Current best remains EXP-064 at `best_test_acc=94.11%` from commit `1119ff8`; the active +0.10pp noise guard requires `best_test_acc >= 94.21%`.
- EXP-072, EXP-073, and EXP-074 produced near-misses at 94.16%, 94.14%, and 94.17%, but EXP-075 showed two near-miss changes were not additive.
- EXP-079 tested the most direct follow-up to EXP-073: a 1000-step CutMix probability ramp. It peaked at 94.09%, below baseline, so early CutMix frequency scheduling is now a medium-importance failed approach.
- Static CutMix probability and alpha brackets are closed around the anchor: `p=0.25`, `p=0.75`, `alpha=0.5`, and `alpha=2.0` all missed the threshold.
- Schedule-only late changes have a poor history, including second drops and cosine tails, and static LR deviations from 0.1 also failed. A very short warmup remains distinct because it changes only the first few hundred unstable updates.
- Architecture and topology changes have mostly underperformed: SE gates, shortcut smoothing, pre-activation blocks, classifier-head changes, residual downsampling tweaks, and shallow-wide depth reduction all missed threshold.
- The fixed-budget anchor appears bounded by small early-stability and late-generalization effects rather than a large missing component. The next experiment should be narrow, preserve the validated anchor, and avoid broad changes that reduce throughput or alter the benchmark.

## Candidate Ideas

### 1. Very Short Linear LR Warmup
**Summary**: Preserve the existing `LR=0.1`, milestones `[21000, 64000]`, CutMix anchor, architecture, transforms, batch size, compile/channels-last path, and validation cadence, but linearly ramp the optimizer LR from 0.02 to 0.1 over the first 500 optimizer steps. After step 500, use the unchanged MultiStep schedule behavior.

**Reasoning**: EXP-073 and EXP-079 suggest the remaining near-threshold signal is in early training stability, but changing CutMix timing did not help enough. LR warmup targets the same high-variance startup window through optimizer dynamics instead of weakening regional mixing. It differs from failed LR 0.08/0.12 scalar retunes because the long-run LR remains 0.1, and it differs from failed second-drop/cosine schedules because the post-drop refinement phase remains unchanged.

**Sources**: `reports/exp-report-073.md`; `reports/exp-report-079.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `knowledge/papers/sgdr-cosine-schedule.md`; `train.py` optimizer/scheduler loop

**Estimated Effort**: low-to-medium

**Risk Assessment**: Schedule work has a weak history, and even 500 lower-LR steps may slightly undertrain the fixed budget. The failure mode should be clean no-improvement rather than crash, as this changes only optimizer LR assignment.

### 2. Slightly Weaker Reflection Crop Jitter
**Summary**: Keep reflection padding mode and all CutMix/optimization settings unchanged, but reduce `RandomCrop(32, padding=4, padding_mode="reflect")` to padding 3. This narrows spatial jitter while preserving the validated reflection boundary behavior.

**Reasoning**: The current anchor combines reflection crop jitter with CutMix. Several added regularizers now over-regularize or land in a near-miss band, so slightly weakening spatial augmentation could improve late clean accuracy without changing labels, model capacity, or schedule. This is distinct from padding-mode siblings because it keeps reflection and only changes crop displacement magnitude.

**Sources**: `reports/exp-report-029.md`; `reports/exp-report-031.md`; `reports/exp-report-071.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `knowledge/references/torchvision-randomcrop-padding.md`

**Estimated Effort**: low

**Risk Assessment**: The effect may be too small or may reduce useful translation invariance. It is also less directly supported than the early-stability signal, because prior crop work validated reflection mode rather than padding magnitude.

### 3. Narrow Layer3 Width Rebalance
**Summary**: Keep ResNet-20 depth and the first two stage widths unchanged, but reduce the final stage from 112 to 104 channels: `STAGE_WIDTHS=(28, 56, 104)`. Preserve CutMix, optimizer, LR schedule, transforms, compile/channels-last, batch size, seed, and validation cadence.

**Reasoning**: EXP-078 completed fewer optimizer steps with a topology change, reinforcing that fixed-budget step coverage matters. A small layer3 reduction could trade modest final-stage capacity for more post-drop optimization while staying much closer to the successful 28/56/112 anchor than prior broad architecture changes.

**Sources**: `reports/exp-report-078.md`; `reports/exp-report-062.md`; `knowledge/papers/wide-residual-networks.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py` `STAGE_WIDTHS`

**Estimated Effort**: low

**Risk Assessment**: Capacity reductions can lower the representational ceiling, and local architecture evidence is weak. This should be considered only after the cleaner optimizer-startup test.

## Idea Evaluation

The very short LR warmup has the clearest connection to the current state of evidence. The last two early-CutMix timing tests suggest the earliest updates are worth probing, but they also show that weakening CutMix itself is not sufficient. A 500-step warmup changes the early optimizer impulse while preserving the validated long-run CutMix and LR schedule, so it tests a different mechanism with low code risk.

The crop-padding reduction is simple and may address over-regularization, but the evidence is indirect. Reflection padding is validated, while padding magnitude has not been bracketed. It is a reasonable fallback if early optimizer stabilization fails, especially because it preserves all training-loop mechanics.

The layer3 width rebalance is also easy, but architecture changes have been a long weak family. It may improve step budget, but the current best likely needs the final-stage capacity; this idea has lower expected impact than the warmup and higher risk of reducing the ceiling.

## Chosen Idea
**Selected**: Very Short Linear LR Warmup

**Why this idea**:
It is the most distinct remaining early-stability test after EXP-073 and EXP-079. It avoids further CutMix temporal weakening, preserves the validated post-drop behavior, and only changes the first 500 optimizer steps. The expected effect is small, but it is a clean way to test whether early optimizer shock rather than early CutMix frequency is the remaining bottleneck.

**Hypothesis**:
If the current CutMix anchor loses a small amount of quality from overly aggressive LR during the earliest mixed-label updates, then linearly warming LR from 0.02 to 0.1 over the first 500 steps will improve post-drop convergence enough to raise `best_test_acc` from 94.11% to at least 94.21%.
