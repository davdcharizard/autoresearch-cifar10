# EXP-006: Schedule Optimization (0.35, 0.55) for AMP

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-006.md
- **Plan**: plans/plan-006.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-006
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: running

## Implementation Notes

### Summary
Changed two threshold constants in `_wall_clock_fractional_step_decay`: 0.5→0.35 and 0.75→0.55. This shifts the LR drops earlier to minimize time in the FP16-unstable LR=0.01 regime and maximize time in the productive LR=0.001 regime.

### Surprises & Discoveries
None.

### Decisions
No deviations.

## Experimental Adjustments
(none)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: running
- **Started**: 2026-05-27T16:02:00Z
- **Ended**: (pending)

Description:
- Running width-2x ResNet-20 with full recipe + AMP + optimized schedule (0.35, 0.55). Expected: ~37 epochs at LR=0.1, ~21 at LR=0.01, ~48 at LR=0.001. Target: 94.8-95.5%.

Observations:
Key Metrics:

## Verification Results
### Conditions Checked
### Informational Metrics

## Errors & Dead Ends

## Human Notes
> (autopilot)
