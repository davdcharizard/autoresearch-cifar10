# EXP-039: Intrinsically Bounded Average-plus-RMS Readout

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-039
- **Commit**: (pending — committed on loop success)
- **Outcome**: (pending)

## Implementation Notes

### Summary

Replaced only the accepted final adaptive-average/flatten sequence with direct spatial average, RMS via `torch.linalg.vector_norm/8`, and fixed `1/64` interpolation before the unchanged classifier. No parameters, modules, hooks, phases, or evaluator changes were added.

### Surprises & Discoveries

The installed PyTorch2.9.1 zero-map FP32/FP64 `vector_norm` backward was directly verified finite with exact zero gradient, resolving the plan review's principal concern without adding epsilon semantics.

### Decisions

Kept coefficient exactly `1/64` and used `vector_norm` rather than square-root-of-mean-square so zero subgradient behavior is defined locally. Evaluation count remains informational under the unchanged once-per-epoch evaluator.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local preflight; production PID pending
- **Log file(s)**: ignored EXP039 controller logs/JSON; root `run.log` only after all gates
- **WandB**: N/A
- **Status**: implementing preflight
- **Started**: 2026-08-06 22:52 UTC
- **Ended**: pending

Description:
- Qualify exact bounded pooling math, unlabeled activity, registered-corpus trajectory safety, and paired timing before one conditional seed42 run. Expected benefit is a small dense salience bias for localized CutMix evidence without sparse max gradients.

Observations:

Key Metrics:

## Verification Results

### Conditions Checked

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> Autopilot; no human intervention.
