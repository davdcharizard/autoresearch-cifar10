# Brainstorm EXP-040
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **EXP-038 report** (`reports/exp-report-038.md`)
  `WEIGHT_DECAY = 2e-4` improved the label-smoothed reflection anchor to 93.97% and is now the active regularization anchor.
- **EXP-039 report** (`reports/exp-report-039.md`)
  `WEIGHT_DECAY = 3e-4` over-regularized the same anchor and fell to 93.55%, bounding isolated stronger decay above 2e-4.
- **Goal learnings** (`goal-learnings/maximize-cifar10-best-test-accuracy.md`)
  Failed approaches now discourage further isolated weight-decay increases, schedule-only second drops, smaller batches, width increases beyond 28/56/112, and smoothing deviations from 0.05.

## Experimental History Review

- Current baseline is EXP-038 at `best_test_acc=93.97%`; the +0.10 percentage-point rule requires EXP-040 to reach at least 94.07%.
- The current anchor is `STAGE_WIDTHS = (28, 56, 112)`, reflected `RandomCrop`, `label_smoothing=0.05`, `WEIGHT_DECAY = 2e-4`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `LR_MILESTONES = [21000, 64000]`, FP32, channels-last, cuDNN benchmark, and `torch.compile`.
- The regularization bracket is now asymmetric: `5e-5` was too weak, `1e-4` was improved by smoothing, `2e-4` is validated, and `3e-4` over-regularizes. Further isolated weight-decay increases are low value.
- Adjacent label-smoothing and first-drop retunes repeatedly stayed inside the noise band, and second-drop schedule-only work is a high-importance failed family.
- A modest LR increase has not been tested on the `2e-4` anchor and has a plausible mechanism: stronger shrinkage may tolerate more high-LR exploration before the preserved 21k first drop.

## Candidate Ideas

### 1. Raise Initial LR to 0.12 on the 2e-4 Anchor
**Summary**: Preserve the current anchor and change only `LR` from `0.1` to `0.12`, keeping `WEIGHT_DECAY = 2e-4` and `LR_MILESTONES = [21000, 64000]`.

**Reasoning**: The successful `2e-4` anchor may support a slightly larger high-LR exploration phase without changing model size or epoch geometry. This tests optimizer dynamics rather than schedule placement, avoiding repeated failed first-drop and second-drop retunes.

**Sources**: `reports/exp-report-038.md`; `reports/exp-report-039.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; current `train.py`.

**Estimated Effort**: low

**Risk Assessment**: Higher LR could destabilize early training, produce noisier pre-drop progress, or interact poorly with the fixed 21k milestone. The failure mode should be a clean no-improvement unless loss diverges.

### 2. Reduce Weight Decay to 1.5e-4
**Summary**: Preserve the current anchor but set `WEIGHT_DECAY = 1.5e-4`, testing whether an intermediate value improves over both 1e-4 and 2e-4.

**Reasoning**: EXP-038 and EXP-039 bracket the useful stronger-decay region, and the optimum could sit below 2e-4. This would map the local regularization curve more precisely.

**Sources**: `reports/exp-report-038.md`; `reports/exp-report-039.md`; `reports/exp-report-023.md`.

**Estimated Effort**: low

**Risk Assessment**: Moving back toward 1e-4 is less likely to clear the high 94.07% threshold, because 2e-4 already produced the best result and 5e-5 failed badly.

### 3. Lower Momentum to 0.85 on the 2e-4 Anchor
**Summary**: Preserve the current anchor but change `MOMENTUM` from `0.9` to `0.85`.

**Reasoning**: Higher momentum 0.95 failed under an older anchor, so the lower side could reduce overshoot with stronger weight decay. This tests optimizer damping while preserving schedule and throughput.

**Sources**: `reports/exp-report-026.md`; `reports/exp-report-038.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: Momentum is sensitive and prior Nesterov / 0.95 changes failed. Lower momentum may slow convergence under the fixed time budget, making it less attractive than a small LR increase.

## Idea Evaluation

The best next experiment is `LR = 0.12` on the `2e-4` anchor. It is a distinct optimizer-dynamics probe after isolated weight decay has been bounded, and it has a clear mechanism: stronger shrinkage may make a slightly larger high-LR phase productive without changing architecture, augmentation, batch geometry, or schedule reachability.

The 1.5e-4 weight-decay bracket would improve understanding but moves back toward weaker shrinkage and likely has lower chance of exceeding 94.07%. Lower momentum is a possible optimizer bracket, but local history has multiple momentum-related failures, and lower momentum risks slowing convergence under the fixed budget.

EXP-040 should therefore change only `LR` from `0.1` to `0.12`, preserving all other EXP-038 anchor choices and verifying the same schedule, batch, parameter, and metric conditions.

## Chosen Idea
**Selected**: Raise Initial LR to 0.12 on the 2e-4 Anchor

**Why this idea**:
It avoids the newly bounded weight-decay direction while testing a plausible interaction with the successful stronger regularization. It is also a one-scalar, no-throughput change with a clean verification path.

**Hypothesis**:
Raising `LR` from `0.1` to `0.12` while keeping `WEIGHT_DECAY = 2e-4` will improve pre-drop exploration and late refinement enough to raise `best_test_acc` from 93.97% to at least 94.07%.
