# Brainstorm EXP-021
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **PyTorch EMA Weight Averaging** (`knowledge/references/pytorch-ema-averaging.md`)
  `AveragedModel` can maintain averaged weights without changing the optimizer or evaluator. Prior use in this project warns that update frequency and BatchNorm-buffer handling are the key implementation risks.

- **Recent local reports** (`reports/exp-report-004.md`, `reports/exp-report-016.md`, `reports/exp-report-020.md`)
  EXP-004 showed per-step EMA was too expensive for too little gain, EXP-016 showed the current-best 28/56/112 recipe has a best/final gap, and EXP-020 pushed further capacity scaling into a recurring failure mode.

No new external search was needed. The next decision is dominated by the local trajectory and the existing PyTorch averaging reference.

## Experimental History Review

- Current baseline is EXP-016 at `best_test_acc=93.23%`; with the goal's +0.10 percentage-point rule, EXP-021 must reach at least `93.33%`.
- The successful anchor is `STAGE_WIDTHS = (28, 56, 112)` with `LR_MILESTONES = [21000, 64000]`.
- Capacity scaling above the anchor is now a high-priority recurring failure: 30/60/120, 29/58/116, and final-stage-only 28/56/128 all failed.
- Projection shortcuts preserved throughput but hurt accuracy, so transition-capacity changes are deprioritized.
- Cutout, Nesterov, BF16, TF32, and several schedule changes have failed under this fixed budget.
- EXP-016 had a useful best/final gap: 93.23% best and 93.03% final. That suggests the low-LR trajectory may contain better-than-final weights worth smoothing.
- EXP-004's per-step EMA reached only 91.98% and lost about 6.8k steps versus its baseline, so any averaging retry must avoid per-step overhead.

## Candidate Ideas

### 1. Sparse Post-Drop Weight Averaging on 28/56/112
**Summary**: Keep the current best 28/56/112 architecture and 21k first LR drop. After the first LR drop, maintain a sparse averaged copy of model weights updated once per epoch, then evaluate the averaged weights once per epoch instead of adding an extra validation pass.

**Reasoning**: This targets the current-best recipe's best/final gap without adding model capacity. It directly addresses EXP-004's failure mechanism by avoiding per-step EMA overhead. Because validation time is excluded but validation cadence is constrained, evaluating only one model per epoch is important: the average should replace the normal evaluation after the averaging window starts rather than doubling evaluation.

**Sources**: `knowledge/references/pytorch-ema-averaging.md`; `reports/exp-report-004.md`; `reports/exp-report-016.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py` epoch-level evaluation structure.

**Estimated Effort**: medium

**Risk Assessment**: The main risks are implementation complexity with `torch.compile`, BatchNorm buffer handling, and accidentally reducing training throughput through large state copies. Worst case is a valid no-improvement or an invalid result if evaluation cadence is mishandled; the plan must keep exactly one evaluation per epoch.

### 2. Schedule-Only 20k First Drop on 28/56/112
**Summary**: Keep the 28/56/112 architecture and all other settings unchanged, but move `LR_MILESTONES` from `[21000, 64000]` to `[20000, 64000]`.

**Reasoning**: EXP-016 improved over the 22k schedule by moving to 21k, and EXP-015 showed 23k was too late. A 20k bracket cleanly tests whether the same local trend continues without confounding architecture changes.

**Sources**: `reports/exp-report-014.md`; `reports/exp-report-015.md`; `reports/exp-report-016.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: The expected effect may be too small to clear the +0.10 threshold, and EXP-020's 20k schedule coupled with final-stage widening peaked far below baseline. This is a clean control but likely lower upside than a new mechanism.

### 3. Lower Weight Decay on the 28/56/112 Anchor
**Summary**: Keep the current architecture and schedule, but reduce `WEIGHT_DECAY` from `1e-4` to a smaller value such as `5e-5` to test whether the widened anchor is slightly over-regularized.

**Reasoning**: Several explicit regularization additions hurt under the fixed budget, and the current best uses a larger model than the original baseline. A mild weight-decay reduction is a low-overhead way to alter regularization without changing throughput-critical architecture.

**Sources**: `goal-learnings/maximize-cifar10-best-test-accuracy.md`; experiment index entries for cutout, Nesterov, and width scaling; `train.py` optimizer configuration.

**Estimated Effort**: low

**Risk Assessment**: Lower weight decay could overfit or increase final loss without improving best accuracy. There is less direct evidence than for sparse averaging, so it is better kept as a later micro-tuning option.

## Idea Evaluation

Sparse post-drop averaging has the clearest new mechanism after the capacity path stalled: it tries to improve which weights are evaluated rather than making the model larger. It is also grounded in an observed property of the current best run, where best accuracy exceeded final accuracy by 0.20 points. EXP-004's EMA failure does not rule this out because that experiment updated every step and paid enough overhead to miss the threshold; EXP-021 can make the update sparse and post-drop only.

The 20k schedule-only bracket is cleaner and easier, but it is an incremental local search move with a likely small effect. It may be worth trying, but needing +0.10 over an already tuned 21k anchor makes its expected upside modest.

Lower weight decay is plausible because regularization-heavy attempts have failed, but the evidence is indirect. It has low implementation cost, yet less reason to expect a threshold-clearing improvement than averaging the late low-LR trajectory.

## Chosen Idea
**Selected**: Sparse Post-Drop Weight Averaging on 28/56/112

**Why this idea**:
It avoids the now-recurring failed width-above-anchor path and targets a concrete weakness in the current best recipe: late-training weights fluctuate enough that best accuracy is meaningfully higher than final accuracy. Sparse epoch-level averaging is a distinct implementation from the failed per-step EMA experiment and can preserve training throughput if it is only updated after the first LR drop.

**Hypothesis**:
Keeping `STAGE_WIDTHS = (28, 56, 112)` and `LR_MILESTONES = [21000, 64000]`, then evaluating a sparse post-drop averaged model once per epoch, will smooth late low-LR weight noise and raise `best_test_acc` to at least 93.33% without violating the fixed-budget or validation-cadence constraints.
