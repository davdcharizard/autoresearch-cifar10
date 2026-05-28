# Plan EXP-007: Width-4x (WIDTH_MULT=4) with AMP
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-007.md

Hypothesis: WIDTH_MULT=4 raises best_test_acc to 94.8-95.3% through ~4x capacity increase. Threshold: >= 94.54%.

## Milestones
### Milestone 1: Code change, ruff pass
### Milestone 2: Run to completion
### Milestone 3: Verification

## Code Changes
**train.py line 19**: `WIDTH_MULT = 2` → `WIDTH_MULT = 4`. One constant. ~4.3M params, channels {64, 128, 256}.

## Configuration Changes
- WIDTH_MULT: 2 → 4 (channel widths quadrupled from He-2015 baseline)

## Execution Environment
- Same as EXP-005 (AMP, local, H20 GPU)
- Estimated ~1 GB VRAM, ~35-50 epochs in 300s
- Estimated runtime: ~400-450s total

## Abort Criteria
Same as EXP-005. Additionally: if epoch count drops below 25, the model likely can't converge — but let it run to see.

## Verification Protocol
Baseline: 94.44%. Threshold: 94.54%.
- Condition 1: best_test_acc > 94.54%
- Condition 2: Summary block complete
- Condition 3: eval_count <= num_epochs
