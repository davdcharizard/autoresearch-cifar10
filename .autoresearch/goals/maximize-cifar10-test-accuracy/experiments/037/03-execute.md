# EXP-037: Exclude Only Terminal Classifier Weight From Decay

## Execution

Overall Status & Info:
- **Created**: 2026-07-27
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-037
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - offline/local-only session
- **Outcome**: failed - valid result below margin

## Implementation Notes

### Summary

Changed only the two optimizer parameter comprehensions so exactly `fc.weight`
moves from matrix decay to the existing zero-decay group. Every convolution
and both successful pooled-head matrices remain continuously decayed.

### Surprises & Discoveries

None during production implementation.

### Decisions

The semantic verifier uses fresh common model snapshots for first-step and
preseeded-momentum cases. It permits only the direct `fc.weight` difference in
the first fixture and does not claim later trajectory identity.

## Experimental Adjustments

None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 24710
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-07-27 02:35 UTC
- **Ended**: 2026-07-27 02:42 UTC

Description:
- The sole fixed-seed score runs only after exact group/update semantics and
  H20 exposure gates pass. It tests whether removing coupled shrinkage from
  only the terminal class vectors improves the accepted nonlinear pooled
  representation. Success requires 94.58% best accuracy.

Observations:

- Semantic preflight passed with exact 999,856/3,626 element groups and
  first/preseeded classifier update deltas `1.454e-5`/`1.457e-5`; all oracle,
  common-state, RNG, and source checks passed (source: local stdout, 02:34 UTC).
- Timing passed with retention 1.001687, projected 130.52384 passes, maximum CV
  0.003050, and 610.2 MiB candidate peak (source: local stdout, 02:35 UTC).
- Sole score completed without runtime error. Mixup/RandAugment disabled at
  steps 16,507/16,575 and 195.0/195.8 s; 27 evaluation epochs were unique
  through final epoch 132 (source: `run.log` lines 38-62).

Key Metrics:

- Best/final accuracy 94.41%/94.38%; best delta -0.07 from baseline and -0.17
  from threshold; final delta -0.07 from accepted (source: lines 64-65).
- Final loss 0.2786, +0.0330 versus accepted 0.2456; best-final gap 0.03
  (source: lines 64-66).
- 25,728 steps, 132 epochs, 131.72736 passes, 300.0 counted / 345.9 total /
  1.1 startup seconds; 1,096.4 MiB and 1,003,482 parameters (source: lines 67-73).

## Verification Results

### Conditions Checked

- **Run integrity**: PASS - exit 0, finite summary, one H20, valid budget,
  131.72736 passes, exact source/topology/transitions, and 27 unique evaluations.
- **Primary metric**: FAIL - 94.41% is below baseline 94.48 and threshold 94.58.
- **Corroboration**: FAIL - final 94.38% <94.45 and loss 0.2786 >0.2456.

### Informational Metrics

- Formal success-only collection skipped; all values are preserved above.

## Errors & Dead Ends

## Human Notes

> Autopilot session; no execution-time intervention requested.
