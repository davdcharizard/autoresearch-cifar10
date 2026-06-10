# Brainstorm EXP-022
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Widening can improve CIFAR residual networks, but this local run has already found a fixed-budget width ceiling at 28/56/112.

- **PyTorch Throughput Tools for CNN Training** (`knowledge/references/pytorch-throughput-tools.md`)
  The current anchor already uses the successful FP32 compile/channels-last path; further ideas should preserve that throughput unless they have a clear accuracy mechanism.

No new external search was needed. The next decision is dominated by local trajectory: the current best is a schedule-calibrated 28/56/112 ResNet-20, and the most recent failures came from exceeding that capacity ceiling or evaluating unstable averaged weights.

## Experimental History Review

- Current baseline is EXP-016 at `best_test_acc=93.23%`; with the goal's +0.10 percentage-point rule, EXP-022 must reach at least `93.33%`.
- The proven anchor is `STAGE_WIDTHS = (28, 56, 112)` with `LR_MILESTONES = [21000, 64000]`.
- Schedule history around this anchor is informative: EXP-014 used a 22k first drop and reached 93.09%; EXP-015 moved later to 23k and fell to 92.88%; EXP-016 moved earlier to 21k and improved to 93.23%.
- A clean 20k first-drop bracket has not been tested on the exact anchor. EXP-020 used 20k, but it was confounded by final-stage widening to 128 channels and peaked at 92.60%.
- Width increases beyond 28/56/112 are now a high-priority recurring failure across proportional, minimal, and final-stage-only variants.
- Averaging is also deprioritized: per-step EMA missed the noise-margin threshold, and EXP-021's long equal post-drop averaging collapsed as snapshots accumulated.
- Regularization additions have generally hurt under the fixed budget, but a mild reduction in regularization remains untested.

## Candidate Ideas

### 1. Schedule-Only 20k First LR Drop on 28/56/112
**Summary**: Keep the current-best architecture, optimizer, batch size, weight decay, compile/channels-last settings, and second milestone unchanged. Modify only `LR_MILESTONES` from `[21000, 64000]` to `[20000, 64000]` to give the 28/56/112 anchor one more epoch-scale slice of LR 0.01 refinement.

**Reasoning**: This is the cleanest unresolved local bracket. On the exact anchor, moving from 23k to 22k to 21k improved the metric, and 20k remains the next adjacent test. Unlike EXP-020, this isolates schedule from capacity changes. It also avoids the high-priority failure mode of widening beyond 28/56/112 and the newly observed averaging collapse.

**Sources**: `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` EXP-014 through EXP-016 and EXP-020; `goal-learnings/maximize-cifar10-best-test-accuracy.md` schedule pattern and width-failure entries; `train.py` `LR_MILESTONES`.

**Estimated Effort**: low

**Risk Assessment**: The expected effect may be small and could miss the +0.10 threshold even if directionally favorable. If 20k is too early, the model may underfit the LR 0.1 phase and fall below the current baseline. Worst case is a valid no-improvement with a precise boundary on the schedule optimum.

### 2. Lower Weight Decay on the 28/56/112 Anchor
**Summary**: Keep the current architecture and schedule unchanged, but reduce `WEIGHT_DECAY` from `1e-4` to `5e-5` to test whether the widened anchor is slightly over-regularized under this fixed training budget.

**Reasoning**: Cutout and other explicit regularization attempts have hurt, suggesting the current setup may not need more regularization. The 28/56/112 model is larger than the original baseline, so a smaller L2 penalty could let it exploit capacity better without changing throughput.

**Sources**: `goal-learnings/maximize-cifar10-best-test-accuracy.md` failed regularization entries; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` EXP-000, EXP-005, EXP-009; `train.py` optimizer configuration.

**Estimated Effort**: low

**Risk Assessment**: The evidence is indirect. Lower weight decay may increase overfitting or worsen calibration without improving best accuracy. A single value could also be too coarse; if it fails, the mechanism is not fully exhausted.

### 3. Conservative Short-Window Post-Drop Averaging
**Summary**: Revisit averaging only as a bounded short-window method: keep the current anchor and evaluate an average of only the most recent few post-drop snapshots rather than all snapshots since the first LR drop.

**Reasoning**: EXP-021 did show an early averaged peak at 91.85% before the collapse, so the failure may be long-horizon incompatibility rather than all averaging. A short window might preserve compatibility and avoid the late collapse.

**Sources**: `reports/exp-report-021.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` EXP-004 and EXP-021 averaging failures; `knowledge/references/pytorch-ema-averaging.md`.

**Estimated Effort**: medium

**Risk Assessment**: This is implementation-heavy relative to the likely gain and still risks BatchNorm mismatch, validation-cadence mistakes, or another no-improvement from evaluating averaged weights. It should wait until cleaner low-cost anchor perturbations are exhausted.

## Idea Evaluation

The 20k first-drop bracket has the strongest direct evidence because it extends a successful local schedule trend on the exact current-best architecture. It also has the cleanest causal mechanism: more time at LR 0.01 may improve refinement after the widened model has already learned enough high-LR features. Its failure mode is informative and low-cost.

Lower weight decay is plausible but less grounded in direct anchor-specific evidence. It could be valuable after the schedule bracket because it also preserves throughput, but the current results do not clearly identify over-regularization as the limiting factor.

Short-window averaging is a different approach from EXP-021's failed long equal average, but it is still adjacent to two averaging failures and adds implementation risk. The local trajectory currently favors simple, isolated anchor perturbations over another mechanism-heavy averaging run.

## Chosen Idea
**Selected**: Schedule-Only 20k First LR Drop on 28/56/112

**Why this idea**:
It is the highest-signal untested local bracket around the current best. It preserves every successful component of EXP-016 except the first LR milestone, avoids the recurring width-above-anchor failure, and cleanly tests whether the 28/56/112 model benefits from even earlier LR 0.01 refinement.

**Hypothesis**:
Changing `LR_MILESTONES` from `[21000, 64000]` to `[20000, 64000]` on the unchanged 28/56/112 anchor will increase low-LR refinement enough to reach at least 93.33% `best_test_acc` without reducing throughput or violating any fixed-budget constraints.
