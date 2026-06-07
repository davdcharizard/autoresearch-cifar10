# EXP-004: k=4 Width

## Execution
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md
- **Plan**: plans/plan-004.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-004
- **Commit**: 7a5ee65
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes
### Summary
Two-line change: WIDTH_MULT 3→4, COSINE_T_MAX 57→49.
### Surprises & Discoveries
Got 58 epochs (predicted 54). T_max=49 was slightly low but best/final gap only 0.09%.
### Decisions
None — straightforward config change.

## Run Log
### Run 1
Metadata:
- **Status**: completed
- **Log file(s)**: run.log

Key Metrics:
- best_test_acc: 95.25%, final_test_acc: 95.16%, num_epochs: 58, num_params: 4,327,754, peak_vram_mb: 537.8

## Verification Results
### Conditions Checked
All 4 passed. 95.25% >= 94.90%.
### Informational Metrics
final_test_loss: 0.2867, training_seconds: 300.0, total_seconds: 386.2

## Errors & Dead Ends
## Human Notes
