# EXP-023: Selective Width with Full Two-Gate SE

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-023
- **Commit**: (pending - no scored success)
- **PR**: N/A - local-only run
- **Outcome**: failed - preregistered exposure feasibility gate failed before scoring

## Implementation Notes

### Summary

Implemented explicit `[32,64,160]` widths and attached two diagnostic-free exact-neutral `160->10->160` SE gates to the final-stage residual branches. Gate construction uses seed 23017 inside a restored CPU RNG fork after width-model initialization.

### Surprises & Discoveries

The initial semantic plan incorrectly required accepted and width-model early tensors to be equal. The existing model-wide initializer runs after all shape-dependent constructors, so EXP-010-style later width changes alter the RNG position before early tensors are reinitialized. The valid isolation is exact composed-versus-width-only common state and RNG; that corrected check passed.

### Decisions

Kept EXP-010 width initialization semantics rather than redesigning initialization and confounding the composition. The plan and preflight were corrected before any score. The tightened 127-pass threshold remained unchanged after timing measured 126.206224.

## Experimental Adjustments

- **Corrected semantic comparison**: require exact composed-versus-width-only state/RNG and accepted early topology, because exact accepted-versus-width tensor equality is not part of EXP-010's shape-dependent initialization. (ref: semantic preflight attempt 1)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A - aborted before scoring
- **Log file(s)**: N/A - no scored `run.log`
- **WandB**: N/A
- **Status**: failed preflight
- **Started**: 2026-07-26
- **Ended**: 2026-07-26

Description:
- Evaluator-free semantic and production-path timing preflights tested the planned 160-channel/two-gate composition. A scored run was permitted only if the implementation passed semantic isolation and projected at least 127 effective passes.

Observations:
- Semantic preflight passed counts 691,674/961,562/968,302, two exact-neutral `160->10->160` gates, seed 23017, composed/width-only state and RNG preservation, optimizer grouping, and two-step gradient opening. (source: semantic preflight stdout)
- Mixup medians were 13.204973/14.849407 ms and hard-label medians 12.994256/14.605652 ms for accepted/candidate; every window CV was below 0.72%. (source: throughput preflight stdout)
- Weighted retention was 0.889403 and projected exposure 126.206224 passes, failing the fixed 127-pass gate. No score or `run.log` was launched. (source: throughput preflight stdout)

Key Metrics:
- best_test_acc: unavailable - no scored run.
- preflight retention/projected passes: 0.889403/126.206224. (source: throughput preflight stdout)

## Verification Results

### Conditions Checked

- Verification not run because the experiment failed its preregistered exposure gate before scoring.

### Informational Metrics

## Errors & Dead Ends

### 2026-07-26 - Composed architecture misses exposure floor
- Error: `AssertionError: projected_passes 126.206224 < 127`
- Root cause: combined selective-width and two-gate overhead retained only 88.94% of accepted production-path throughput.
- Source: throughput preflight stdout
- Do NOT retry: do not lower the floor or score this exact `[32,64,160]` plus two ratio-16 gate composition.

## Human Notes

> Autopilot local-only execution; no intervention requested.
