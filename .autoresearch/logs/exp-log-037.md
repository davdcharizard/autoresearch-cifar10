# EXP-037: Epoch-level sync + channels_last + T_max=49 + LR clamp

## Execution
- **Created**: 2026-06-04
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-037
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: (pending)

## Implementation Notes
### Summary
Removed per-step torch.cuda.synchronize() (19K barriers/run). Time tracking moved to epoch boundaries. Added channels_last, kept T_max=49, LR clamp after cosine.

## Run Log
### Run 1
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **Status**: running

## Verification Results
### Conditions Checked
## Errors & Dead Ends
## Human Notes
