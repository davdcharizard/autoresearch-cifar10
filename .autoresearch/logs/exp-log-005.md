# EXP-005: k=6 + Pre-activation Blocks

## Execution
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-005
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed

## Implementation Notes
### Summary
k=6 with pre-activation blocks, CutMix, T_max=35.
### Surprises & Discoveries
Only 32 epochs — k=6 is too slow per epoch for the 300s budget.
### Decisions
None.

## Run Log
### Run 1
- **Status**: completed
- best_test_acc: 94.52%, final: 94.47%, 32 epochs, 9.7M params, 735MB VRAM

## Verification Results
### Conditions Checked
94.52% < 95.35% — FAIL

## Errors & Dead Ends
### 2026-05-28 — k=6 capacity exceeds convergence budget
- Error: 94.52% < 95.25% baseline despite 2.25x more params
- Root cause: 32 epochs insufficient for 9.7M model; k=4 with 58 epochs is better
- Do NOT retry: k>=6 without proportionally increasing training budget or using techniques to accelerate convergence

## Human Notes
