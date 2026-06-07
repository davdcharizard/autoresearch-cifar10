# EXP-026: Width k=5 with calibrated T_max

## Execution

Overall Status & Info:
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-026.md
- **Plan**: plans/plan-026.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-026
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed

## Implementation Notes

### Summary
Changed WIDTH_MULT 4→5 for ~6.7M params. Added dynamic T_max calibration after epoch 2: measures epoch time, estimates total epochs, creates new cosine scheduler with calibrated T_max.

### Surprises & Discoveries
None yet.

### Decisions
Calibrate from average of epochs 1-2 (total_training_time/2) rather than epoch 2 alone to reduce noise.

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-29
- **Ended**: 2026-05-29

Description:
- Width k=5 (6.76M params) with dynamic T_max calibration from epoch 2.

Observations:
- Catastrophic: 89.94% best, 67.78% final
- T_max calibrated to 12 from ep2_time=17.22s (inflated by compile overhead) — est 17 epochs but actual 31
- T_max=12 caused LR to decay too fast (0→peak in just 12 cosine epochs) then restart
- 31 epochs is far too few for 6.7M params
- Confirms k=4 is the capacity sweet spot

Key Metrics:
- best_test_acc: 89.94% (source: run.log)
- final_test_acc: 67.78% (source: run.log)
- num_epochs: 31, num_params: 6,758,810 (source: run.log)

## Verification Results

### Conditions Checked

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
