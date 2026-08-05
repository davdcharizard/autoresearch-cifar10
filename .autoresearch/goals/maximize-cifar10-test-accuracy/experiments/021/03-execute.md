# EXP-021: Final-Ten-Percent SAM

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-021
- **Commit**: (pending - no scored success)
- **PR**: N/A - local-only run
- **Outcome**: failed - preregistered throughput feasibility gate failed before scoring

## Implementation Notes

### Summary

Implemented rho-0.05 SAM for progress at or above 90%, preserving accepted SGD earlier. The production helper clones and restores parameters with `copy_`, preserves only the first forward's BatchNorm update, and leaves the existing optimizer to apply second-pass gradients once.

### Surprises & Discoveries

SAM step time was 2.15x normal, slightly worse than the ideal 2x assumption. Whole-run projected retention was 89.67%, narrowly below the preregistered 90% feasibility floor even though projected exposure remained 127.24 passes.

### Decisions

No scored run was launched. Lowering the retention threshold after observing preflight would invalidate the preregistration; the exact final-10% rho-0.05 treatment is closed as infeasible under its plan.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A - aborted before scoring
- **Log file(s)**: N/A - no scored run
- **WandB**: N/A
- **Status**: failed preflight
- **Started**: 2026-07-26
- **Ended**: 2026-07-26

Description:
- Evaluator-free semantic and throughput preflights tested the planned final-10% SAM implementation before spending the sole scored run.

Observations:
- Semantic preflight passed strict 90% boundary, 691,674 parameters, exact weight restoration, one persistent BatchNorm update, and finite second-pass gradients. (source: semantic preflight stdout)
- Timing measured normal/SAM steps at 12.447067/26.789581 ms with CV 0.000226/0.000866. Whole-run retention was 0.896678 and projected exposure 127.238548 passes. (source: throughput preflight stdout)
- The `retention >=0.90` condition failed; scoring was aborted without creating `run.log`. (source: throughput preflight assertion)

Key Metrics:
- best_test_acc: unavailable - no scored run.
- whole-run projected retention/passes: 0.896678 / 127.238548. (source: throughput preflight stdout)

## Verification Results

### Conditions Checked

- Verification not run because execution failed the preregistered feasibility gate before scoring.

### Informational Metrics

- No scored metrics available.

## Errors & Dead Ends

### 2026-07-26 - Final-window SAM misses throughput retention floor
- Error: `AssertionError: whole_run_retention 0.896678 < 0.90`
- Root cause: SAM steps cost 2.15x normal due to the second forward/backward plus exact parameter and BN cloning/restoration.
- Source: throughput preflight stdout
- Do NOT retry: do not lower the floor or rerun this exact rho-0.05 final-10% design; a future method needs periodic/fewer perturbation steps or a distinct cost rationale.

## Human Notes

> Autopilot local-only execution; no intervention requested.
