# EXP-007: Width-4x (WIDTH_MULT=4) with AMP

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-007.md
- **Plan**: plans/plan-007.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-007
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: running

## Implementation Notes
### Summary
Changed WIDTH_MULT from 2 to 4. Channels now {64, 128, 256}, ~4.3M params. All other settings unchanged from EXP-005.
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
- **Started**: 2026-05-27T16:10:00Z
- **Ended**: (pending)

Description:
- Width-4x ResNet-20 (~4.3M params) with AMP + aug + WD=5e-4 + (0.5, 0.75) schedule. Expected ~35-50 epochs, target 94.8-95.3%.

Observations:
Key Metrics:

## Verification Results
### Conditions Checked
### Informational Metrics

## Errors & Dead Ends

## Human Notes
> (autopilot)
