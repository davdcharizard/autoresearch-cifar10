# Brainstorm EXP-039
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **EXP-038 report** (`reports/exp-report-038.md`)
  Raising `WEIGHT_DECAY` from `1e-4` to `2e-4` improved the label-smoothed reflection anchor from 93.70% to 93.97% without changing throughput geometry, establishing stronger shrinkage as the newest validated lever.
- **Goal learnings** (`goal-learnings/maximize-cifar10-best-test-accuracy.md`)
  Current patterns now identify `2e-4` as the regularization anchor and recommend bracketing nearby stronger values before changing architecture.
- **Experiment index** (`experiment-indices/maximize-cifar10-best-test-accuracy.tsv`)
  The active baseline is 93.97%, so the goal's +0.10 point rule requires EXP-039 to reach at least 94.07%.

## Experimental History Review

- Current best is EXP-038 at `best_test_acc=93.97%` with `STAGE_WIDTHS = (28, 56, 112)`, reflected `RandomCrop`, `label_smoothing=0.05`, `WEIGHT_DECAY = 2e-4`, `BATCH_SIZE = 128`, `LR_MILESTONES = [21000, 64000]`, FP32, channels-last, cuDNN benchmark, and `torch.compile`.
- The strongest recent evidence is directional: weaker weight decay (`5e-5`, EXP-023) failed badly, baseline `1e-4` reached 93.70 after smoothing, and `2e-4` reached 93.97. This makes nearby stronger decay the cleanest next bracket.
- Adjacent label-smoothing probes around 0.05 were no-improvements: 0.03 and 0.08 both stayed inside the prior noise band, so smoothing changes should not be the next isolated move.
- Smaller batches are a medium-importance failed family, and schedule-only second drops are a high-importance failed family. Isolated first-drop retunes around the label-smoothed anchor also repeatedly stayed below the tightened threshold.
- Width increases beyond 28/56/112 are a high-importance recurring failure. The next idea should preserve architecture and target regularization or optimizer dynamics.

## Candidate Ideas

### 1. Increase Weight Decay to 3e-4
**Summary**: Preserve the current EXP-038 anchor and change only `WEIGHT_DECAY` from `2e-4` to `3e-4`.

**Reasoning**: EXP-038 showed that stronger shrinkage is beneficial on the label-smoothed reflection anchor, while prior weaker decay failed. A `3e-4` probe is the nearest stronger-side bracket and keeps the successful architecture, augmentation, smoothing, schedule, batch size, optimizer class, and throughput path unchanged. If 2e-4 was not yet the optimum, this can push the late post-drop plateau beyond 94.07%.

**Sources**: `reports/exp-report-038.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Patterns and Failed Approaches; current `train.py`.

**Estimated Effort**: low

**Risk Assessment**: The main risk is over-regularization: stronger decay may reduce late plateau accuracy or final calibration. The failure mode should be a clean no-improvement because this is one scalar and should not affect throughput or constraints.

### 2. Raise Initial LR to 0.12 on the 2e-4 Anchor
**Summary**: Preserve the EXP-038 anchor but change `LR` from `0.1` to `0.12`, keeping `WEIGHT_DECAY = 2e-4` and the existing milestones.

**Reasoning**: Stronger weight decay may allow slightly more aggressive high-LR exploration before the proven 21k first drop. This tests optimizer dynamics rather than adjacent schedule timing, which has repeatedly underperformed.

**Sources**: `reports/exp-report-038.md`; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Failed Approaches.

**Estimated Effort**: low

**Risk Assessment**: Higher LR could destabilize early training or interact poorly with the fixed 21k first drop. Its mechanism is less directly validated than the weight-decay bracket.

### 3. Reduce Weight Decay to 1.5e-4
**Summary**: Preserve the EXP-038 anchor but set `WEIGHT_DECAY = 1.5e-4`, testing whether the optimum lies between the former 1e-4 and new 2e-4 anchors.

**Reasoning**: EXP-038 proves 2e-4 beats 1e-4, but the optimum could be an intermediate value. This is a low-risk bracket that may improve calibration if 2e-4 is slightly too strong late.

**Sources**: `reports/exp-report-038.md`; `reports/exp-report-023.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: Because 2e-4 already improved substantially, moving back toward 1e-4 may reduce the useful shrinkage signal and is less likely to clear the new 94.07% threshold in one step.

## Idea Evaluation

The best next experiment is the `3e-4` weight-decay bracket. It directly extends the newest validated mechanism and keeps the experiment as an isolated scalar change with a clean failure mode. It also follows the goal-learning implication to bracket nearby stronger values before moving to architecture or more complex late-stability mechanisms.

The LR 0.12 idea is plausible, especially with stronger shrinkage now in place, but it has less direct local evidence and could conflate exploration with schedule calibration. The 1.5e-4 bracket is useful for mapping the optimum, but it moves back toward the weaker side and therefore has lower expected chance of clearing the now-higher 94.07% threshold.

EXP-039 should therefore change only `WEIGHT_DECAY` from `2e-4` to `3e-4`, preserving all other EXP-038 anchor choices and verifying the same schedule, batch, parameter, and metric conditions.

## Chosen Idea
**Selected**: Increase Weight Decay to 3e-4

**Why this idea**:
It is the most direct exploitation of the latest successful experiment, tests a clear stronger-side regularization bracket, and avoids known failed spaces such as width expansion, schedule-only retunes, smaller batches, and adjacent smoothing changes.

**Hypothesis**:
Increasing `WEIGHT_DECAY` from `2e-4` to `3e-4` will further improve the late post-drop plateau of the label-smoothed reflection anchor enough to raise `best_test_acc` from 93.97% to at least 94.07%.
