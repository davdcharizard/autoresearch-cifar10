# Plan EXP-006: Schedule Optimization (0.35, 0.55) for AMP
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-006.md

Hypothesis: shifting LR drops from (0.5, 0.75) to (0.35, 0.55) gives the stable LR=0.001 phase 45% of budget instead of 25%, raising best_test_acc to 94.8-95.5%. Threshold: >= 94.54%.

## Milestones

### Milestone 1: Code changes
- [ ] Create branch, change two thresholds in lambda, ruff check

### Milestone 2: Run to completion
### Milestone 3: Verification

## Code Changes

**train.py** — `_wall_clock_fractional_step_decay` function: change `0.5` to `0.35` and `0.75` to `0.55` in the threshold comparisons.

## Configuration Changes
- Schedule first drop: 0.5 → 0.35 (fires at ~37 epochs instead of ~53)
- Schedule second drop: 0.75 → 0.55 (fires at ~58 epochs instead of ~80)

## Execution Environment
- Same as EXP-005 (AMP, local, H20 GPU)
- Estimated runtime: ~400s total

## Abort Criteria
- Same as EXP-005 plus: if LR=0.01 phase still shows severe oscillation despite being shorter, note but don't abort — the hypothesis is about the extended 0.001 phase

## Verification Protocol
Baseline: 94.44%. Threshold: 94.54%.
- Condition 1: best_test_acc > 94.54%
- Condition 2: Summary block complete
- Condition 3: eval_count <= num_epochs
