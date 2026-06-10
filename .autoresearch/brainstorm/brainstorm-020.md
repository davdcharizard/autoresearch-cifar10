# Brainstorm EXP-020
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Width remains a validated CIFAR residual-network axis, but the project evidence now says proportional widening beyond 28/56/112 is too costly or poorly calibrated under the fixed 300s budget.

- **PyTorch EMA Weight Averaging** (`knowledge/references/pytorch-ema-averaging.md`)
  Averaged weights remain a plausible low-overhead route if updates are sparse, but prior per-step EMA overhead makes implementation discipline more important than another direct EMA retry.

- **Recent local experiment reports** (`reports/exp-report-016.md`, `reports/exp-report-017.md`, `reports/exp-report-019.md`)
  The current best is 28/56/112 with a 21k first LR drop. Both 30/60/120 and 29/58/116 failed, so the next capacity experiment should avoid broad proportional widening.

No new external search was needed. The existing knowledge base plus the latest local reports provide sufficient direction for a tightly scoped next experiment.

## Experimental History Review

- Current baseline is EXP-016 at `best_test_acc=93.23%`; the current goal requires at least `93.33%` for EXP-020 to count as an improvement.
- Width scaling produced the main gains through EXP-016: 20/40/80, 24/48/96, and 28/56/112 each improved when paired with schedule calibration.
- Proportional widening beyond 28/56/112 has now failed twice: 30/60/120 peaked at 93.16% with only 27,400 steps, and 29/58/116 peaked at 92.59% despite 36,139 steps.
- Projection shortcuts at the current width preserved throughput but reduced accuracy to 92.97%, so transition shortcut capacity is not the next best lever.
- The 28/56/112, 21k recipe remains the anchor. It reached 34,208 steps and 93.23%, with a best/final gap that keeps sparse late averaging plausible.
- Untried gaps: final-stage-only widening, lower-overhead late averaging, and other targeted capacity changes that avoid broad early-stage FLOPs.

## Candidate Ideas

### 1. Final-Stage-Only Widening to 28/56/128 with 20k First Drop
**Summary**: Keep stage 1 and stage 2 at the current successful widths, but widen only the final 8x8 stage from 112 to 128 channels by setting `STAGE_WIDTHS = (28, 56, 128)`. Move the first LR milestone from 21k to 20k to compensate for expected throughput loss while keeping the second milestone unreachable at 64k.

**Reasoning**: The final stage is spatially cheapest, so it can add representational capacity with less broad compute growth than proportional widening. EXP-017 and EXP-019 suggest all-stage widening is no longer useful, but they do not rule out targeted late-stage capacity. A 20k first drop is a conservative schedule response: if final-stage widening reduces the step budget, the model still gets enough LR 0.01 refinement time.

**Sources**: `train.py` stage structure; `reports/exp-report-016.md`; `reports/exp-report-017.md`; `reports/exp-report-019.md`; `knowledge/papers/wide-residual-networks.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: The nonuniform stage ratio may hurt more than it helps, and the final stage may still add enough compute to reduce step count. Worst case is a valid no-improvement that clarifies whether targeted capacity is viable.

### 2. Sparse Post-Drop Weight Averaging on 28/56/112
**Summary**: Keep the current 28/56/112 architecture and 21k schedule, but maintain a sparse averaged copy of the model after the first LR drop, such as updating once per epoch and evaluating the averaged weights once per epoch.

**Reasoning**: The current best has a meaningful best/final gap, suggesting late low-LR weights fluctuate around useful solutions. Prior per-step EMA missed the threshold because update overhead reduced steps, but sparse post-drop averaging could smooth evaluation weights with much less cost.

**Sources**: `reports/exp-report-004.md`; `reports/exp-report-016.md`; `reports/exp-report-019.md`; `knowledge/references/pytorch-ema-averaging.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: medium

**Risk Assessment**: The implementation is more complex than a constant change. Averaging compiled models or BatchNorm buffers incorrectly could make the result invalid or slow the loop enough to repeat the EMA failure.

### 3. Schedule-Only 20k First Drop on 28/56/112
**Summary**: Keep `STAGE_WIDTHS = (28, 56, 112)` and move the first LR drop from 21k to 20k, testing whether even more LR 0.01 refinement improves the current best.

**Reasoning**: EXP-016 improved by moving from 22k to 21k, so a direct local bracket at 20k is the cleanest schedule-only continuation. It has a simple causal mechanism and minimal implementation risk.

**Sources**: `reports/exp-report-014.md`; `reports/exp-report-015.md`; `reports/exp-report-016.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: Schedule-only effects may be exhausted, and 20k may start LR 0.01 refinement too early. The expected gain is probably small, so clearing the +0.10 threshold may be difficult.

## Idea Evaluation

Final-stage-only widening has the best balance of novelty, mechanism clarity, and feasibility. It directly addresses the failure mode of EXP-017 and EXP-019 by avoiding wider early stages, while still testing whether capacity can move the metric beyond the 28/56/112 ceiling. The causal mechanism is clear: increase high-level representation capacity where CIFAR feature maps are already 8x8, and compensate with a slightly earlier LR drop.

Sparse post-drop weight averaging is also plausible, but it has higher implementation risk and can easily become a throughput or validity problem if model averaging interacts poorly with `torch.compile`, BatchNorm buffers, or validation cadence. It is worth trying soon, but not before the simpler targeted-capacity gap is tested.

The 20k schedule-only bracket is the safest code change but the lowest expected impact. EXP-016 already found a strong local schedule point, and a pure 1k earlier drop would need a surprisingly large effect to clear the 93.33% threshold.

## Chosen Idea
**Selected**: Final-Stage-Only Widening to 28/56/128 with 20k First Drop

**Why this idea**:
It is the most targeted remaining capacity test after proportional widening failed. The change is isolated, respects all constraints, and answers whether capacity can still help if it is added only in the cheapest stage rather than across the full network.

**Hypothesis**:
Changing to `STAGE_WIDTHS = (28, 56, 128)` and `LR_MILESTONES = [20000, 64000]` will preserve more useful step budget than proportional widening while adding final-stage capacity, allowing `best_test_acc` to reach at least `93.33%`.
