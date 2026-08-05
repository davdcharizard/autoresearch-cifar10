# EXP-020: Extend Mixup to 75 Percent

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-020
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - local-only run
- **Outcome**: failed - valid run scored below baseline and acceptance threshold

## Implementation Notes

### Summary

Changed only `MIXUP_END_FRACTION` from 0.65 to 0.75. Model, alpha, coefficient sharing, optimizer, time-based LR, data transforms, seed, evaluation, and all other settings remain accepted.

### Surprises & Discoveries

None during implementation; the treatment is a single constant change already exposed by the accepted code.

### Decisions

Transition validation uses a one-second tolerance band because the strict time boundary is observed after a completed optimizer step. The fixed single-run protocol remains authoritative despite inherent wall-clock step jitter.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 7413
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed (research no-improvement)
- **Started**: 2026-07-26 18:24:56 UTC
- **Ended**: 2026-07-26 18:32:41 UTC

Description:
- One fixed-seed local H20 run tests whether extending accepted batch-shared alpha-0.2 mixup by 30 counted seconds improves generalization while retaining a 75-second hard-label tail. Success requires at least 94.17%. A complete valid run will not be rerun or tuned.

Observations:
- Semantic preflight passed the exact one-line diff, all other hyperparameters, strict 0.75 boundary, one simulated transition, and identical learning-rate values. (source: preflight stdout)
- Scope/compile audit passed with only `MIXUP_END_FRACTION = 0.65 -> 0.75` in tracked production code. (source: git/compile audit stdout)
- The scored run completed once with exit 0, exact 691,674 parameters, 29 unique evaluations, and no error signature. (source: `run.log` startup/final summary)
- Mixup disabled exactly once at epoch 106, step 20,569 and logged 225.0 seconds, within the preregistered transition band. (source: `run.log` transition line)
- The candidate realized 27,655 steps = 141.5936 passes, so the regression is not explained by exposure loss. (source: `run.log` final summary)

Key Metrics:
- best_test_acc: 93.82% at final epoch 142, -0.25 from baseline and -0.35 below threshold. (source: `run.log` final evaluation/summary)
- final_test_acc/loss: 93.82% / 0.2612. (source: `run.log` final summary)
- timing/exposure: 300.0 training seconds, 340.5 total seconds, 27,655 steps = 141.5936 passes. (source: `run.log` final summary)
- resources/model: 1,094.0 MiB peak VRAM and 691,674 parameters. (source: `run.log` final summary)

## Verification Results

### Conditions Checked

- **Completion and integrity**: PASS - exit 0, one H20, correct count, 300.0 counted and 340.5 total seconds, one 225.0-second transition, 29 unique evaluations, finite summary, and no errors. (source: `run.log`)
- **Primary metric >=94.17%**: FAIL - best 93.82%, 0.25 below baseline and 0.35 below threshold. Verification stopped on this necessary-condition failure. (source: `run.log` final summary and results index)
- **Remaining conditions**: skipped after metric failure; pre-score source, semantic, scope, and diff checks passed.

### Informational Metrics

- Skipped under the verification guard; values are preserved under Run 1 Key Metrics.

## Errors & Dead Ends

## Human Notes

> Autopilot local-only execution; no intervention requested.
