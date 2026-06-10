# Brainstorm EXP-011
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Widening can improve CIFAR residual-network accuracy more efficiently than simply increasing depth, but fixed-budget runtime needs careful schedule planning.

- **EXP-006 ResNet-32 report** (`reports/exp-report-006.md`)
  Depth scaling failed because the model reached only 23,642 steps and missed its first LR drop; this argues for smaller capacity changes and earlier milestones.

- **EXP-010 Nesterov report** (`reports/exp-report-010.md`)
  Another cheap recipe ablation preserved throughput but hurt accuracy, so the loop should move toward a higher-ceiling mechanism.

No new external search was needed; the candidate is grounded in the saved WRN note and local capacity-scaling failure analysis.

## Experimental History Review

- Current baseline remains EXP-002 at `best_test_acc=91.95%`; success requires `>=92.05%`.
- The validated base recipe is still FP32 compile/channels-last ResNet-20 with classical momentum and milestones `[32000, 48000]`.
- Cheap recipe modifications have mostly failed: cutout is recurring negative, Nesterov is negative, EMA's +0.03 is below threshold, BF16/TF32 precision changes hurt, and nearby schedule-only changes hurt.
- EXP-006 does not rule out capacity, but it rules out the tested depth/schedule combination: ResNet-32 was too slow for a 26k first milestone.
- A smaller width increase can test capacity without adding residual-block depth, and an earlier first drop can avoid spending the whole run at LR 0.1 if the step budget falls.

## Candidate Ideas

### 1. ResNet-20 Width 1.25x with Earlier First LR Drop
**Summary**: Keep the ResNet-20 depth but widen the channel stages from 16/32/64 to 20/40/80, preserving the EXP-002 optimizer, augmentation, precision, compile, and channels-last path. Use a calibrated schedule such as `[24000, 64000]` so the run reaches LR 0.01 even if wider channels reduce the step budget.

**Reasoning**: WRN evidence supports width as a better CIFAR scaling direction than depth. EXP-006 failed because deeper ResNet-32 missed its first LR drop; widening by only 1.25x should be less expensive than adding two blocks per stage, and an earlier first drop directly addresses the prior capacity-schedule failure mode.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `reports/exp-report-006.md`; `reports/exp-report-002.md`.

**Estimated Effort**: medium

**Risk Assessment**: The wider model may still reduce steps enough to undertrain, or the earlier milestone may be miscalibrated. Worst case is a valid no-improvement that better maps the capacity/runtime frontier.

### 2. Isolated Mild Label Smoothing
**Summary**: Preserve the EXP-002 recipe and add only mild label smoothing to `F.cross_entropy`, such as `label_smoothing=0.03`.

**Reasoning**: Label smoothing remains unisolated from EXP-000's failed bundle and has no expected throughput overhead. However, it may lower exact top-1 accuracy even if it improves loss or calibration, and the recent Nesterov failure makes cheap bundle-isolation ablations less compelling.

**Sources**: `reports/exp-report-000.md`; `reports/exp-report-010.md`.

**Estimated Effort**: low

**Risk Assessment**: It is easy to implement but likely low ceiling; a single coefficient may miss the narrow useful range.

### 3. Late Averaging Without Per-Step EMA
**Summary**: Revisit the averaging idea by averaging only late checkpoints or low-frequency updates instead of per-step EMA, aiming to recover EXP-004's small accuracy gain without its throughput cost.

**Reasoning**: EXP-004 reached 91.98%, the closest non-accepted result, but update overhead reduced steps and the gain was below the +0.10 threshold. A lower-overhead averaging variant could preserve more steps while smoothing the final model.

**Sources**: `reports/exp-report-004.md`; `references/pytorch-ema-averaging.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: medium

**Risk Assessment**: More implementation surface than width or label smoothing, and any model-weight swapping must avoid adding extra validation or changing the evaluator path.

## Idea Evaluation

Width 1.25x has the highest expected upside because the recipe-only search is now crowded with clean failures. It also directly follows the WRN evidence while avoiding EXP-006's exact depth-scaling failure. The schedule is a risk, but the plan can make the earlier first drop explicit and treat step count as a key diagnostic.

Label smoothing is cheaper, but the expected impact is small and may optimize calibration rather than best top-1. Late averaging is interesting because EXP-004 was close, but it has more implementation risk and could again lose step budget if not carefully engineered.

## Chosen Idea
**Selected**: ResNet-20 Width 1.25x with Earlier First LR Drop

**Why this idea**:
It is the most credible higher-ceiling move after several clean recipe ablations failed. It tests width rather than depth, preserves the proven FP32 throughput path, and explicitly addresses EXP-006's schedule-miss mechanism.

**Hypothesis**:
A 1.25x wider ResNet-20 with an earlier first LR drop will gain enough capacity-driven accuracy to reach at least `92.05%` `best_test_acc`, while still completing enough steps for low-LR refinement under the 300s budget.
