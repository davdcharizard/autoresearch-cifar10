# EXP-004: Earlier 50% Mixup Cutoff

## Execution

Overall Status & Info:
- **Created**: 2026-07-24
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-004
- **Commit**: (pending — committed on loop success)
- **PR**: N/A — local-only workflow requested
- **Outcome**: failed

## Implementation Notes

### Summary

Changed only `MIXUP_END_FRACTION` from 0.65 to 0.50 in `train.py`. This moves the existing mixup-to-hard-label transition from 195 to 150 counted seconds while preserving alpha 0.2, WRN-16-2, all optimizer and LR settings, seed and RNG code, loader, evaluation cadence, finite-loss guard, and output schema.

### Surprises & Discoveries

The planned 50% transition occurs at LR 0.109175 under the unchanged time-based cosine schedule, so the added hard-label phase begins at a moderate LR rather than merely extending the terminal low-LR endpoint.

### Decisions

No implementation decisions or deviations were required. Lint, compile, diff, constant assertion, parameter count, evaluator call-site, scope, and one-H20 checks all passed.

## Experimental Adjustments

None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local timeout PID 1137863
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-24 13:00:03 UTC
- **Ended**: 2026-07-24 13:06:30 UTC

Description:
- Run the accepted WRN-16-2 and alpha-0.2 mixup configuration with mixup disabled at 50% rather than 65% counted time. This tests whether 45 additional seconds of unchanged hard-label cosine refinement improves the 94.07% baseline. Success requires `best_test_acc >= 94.17%`, a complete 300-second training summary, and total runtime below 600 seconds.

Observations:

- Process launched successfully and `run.log` confirmed CUDA, WRN-16-2 with 691,674 parameters, a 300-second budget, and 195 batches per epoch. (source: local PID 1137863; `run.log` L1-L4)
- Mixup disabled exactly once at 150.0 counted seconds (50.0%), epoch 71, step 13,674, LR 0.1092. (source: `run.log` L34)
- Accuracy peaked at 93.91% at epoch 135 and finished at 93.90%, below the 94.07% baseline despite 143.1 realized passes. (source: `run.log` L60-L67)

Key Metrics:

- `best_test_acc`: 93.91% @ epoch 135 (source: `run.log` L60, L66)
- `final_test_acc`: 93.90%; `final_test_loss`: 0.2708 (source: `run.log` L67-L68)
- `training_seconds`: 300.0; `total_seconds`: 340.7 (source: `run.log` L69-L70)
- `num_steps`: 27,954; `num_epochs`: 144; realized dataset passes: 143.1 (source: `run.log` L73-L74)
- `peak_vram_mb`: 1,094.0; `num_params`: 691,674 (source: `run.log` L72, L75)

## Verification Results

### Conditions Checked

- **Run completes without crashing and within 10 minutes**: PASS. Exit code 0, complete summary, one H20, `training_seconds=300.0`, `total_seconds=340.7`, unique every-fifth-plus-final evaluation records, and an exact one-line `train.py` diff. (source: process exit; `run.log` L6-L75; git diff)
- **`best_test_acc` exceeds baseline by at least 0.1 percentage points**: FAIL. 93.91% is 0.16 points below the 94.07% baseline and 0.26 points below the 94.17% threshold. Verification stopped on this necessary-condition failure. (source: `04-results.tsv`; `run.log` L66)

### Informational Metrics

Skipped under the verification protocol after the necessary metric condition failed; the observed values are retained in Run 1 Key Metrics for analysis.

## Errors & Dead Ends

None.

## Human Notes

> The user requested a fully offline, local workflow with no GitHub CLI or remote PR operations.
