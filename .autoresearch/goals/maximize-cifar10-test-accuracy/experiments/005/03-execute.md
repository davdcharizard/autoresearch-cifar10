# EXP-005: Stronger Alpha-0.4 Mixup

## Execution

Overall Status & Info:
- **Created**: 2026-07-24
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-005
- **Commit**: (pending — committed on loop success)
- **PR**: N/A — local-only workflow requested
- **Outcome**: failed

## Implementation Notes

### Summary

Changed only `MIXUP_ALPHA` from 0.2 to 0.4 in `train.py`, preserving the validated 65% cutoff and every architecture, optimizer, schedule, seed, loader, evaluator, guard, logging, and output setting.

### Surprises & Discoveries

A 100,000-sample CUDA check confirmed the intended distribution shift: alpha 0.2 had mean 0.4991 with 21.34% of lambdas in `[0.2,0.8]`, while alpha 0.4 had mean 0.4993 with 35.53% central lambdas.

### Decisions

No code deviation was needed. Because Beta rejection sampling consumes alpha-dependent CUDA RNG draws, the fixed seed does not preserve bit-identical later permutations; the experiment measures the alpha-0.4 stochastic process as a whole.

## Experimental Adjustments

None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local timeout PID 1141983
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-24 13:21:07 UTC
- **Ended**: 2026-07-24 13:27:43 UTC

Description:
- Run the accepted WRN-16-2 and 65/35 mixup schedule with alpha 0.4 instead of 0.2. This tests whether increasing materially mixed batches improves generalization while preserving the proven timing and exposure. Success requires `best_test_acc >= 94.17%`; final test loss below 0.2432 is a supporting mechanism signal only.

Observations:

- Process launched successfully and `run.log` confirmed CUDA, WRN-16-2 with 691,674 parameters, and the 300-second budget. (source: local PID 1141983; `run.log` L1-L4)
- Mixup disabled once at 195.0 seconds (65.0%), epoch 92, step 17,872, LR 0.0612. (source: `run.log` L42)
- Accuracy finished at its 93.57% best with test loss 0.2737, below the 94.07%/0.2432 alpha-0.2 baseline despite normal exposure. (source: `run.log` L64-L68)

Key Metrics:

- `best_test_acc`: 93.57% @ final epoch 143 (source: `run.log` L64-L67)
- `final_test_acc`: 93.57%; `final_test_loss`: 0.2737 (source: `run.log` L67-L68)
- `training_seconds`: 300.0; `total_seconds`: 339.8 (source: `run.log` L69-L70)
- `num_steps`: 27,875; `num_epochs`: 143; realized passes: 142.7 (source: `run.log` L73-L74)
- `peak_vram_mb`: 1,094.0; `num_params`: 691,674 (source: `run.log` L72, L75)

## Verification Results

### Conditions Checked

- **Run completes without crashing and within 10 minutes**: PASS. Exit code 0, complete summary, one H20, `training_seconds=300.0`, `total_seconds=339.8`, unique evaluation epochs, and an exact one-line in-scope diff. (source: process exit; `run.log` L6-L75; git diff)
- **`best_test_acc` exceeds baseline by at least 0.1 percentage points**: FAIL. 93.57% is 0.50 points below the 94.07% baseline and 0.60 below the 94.17% threshold. Verification stopped. (source: results index; `run.log` L66)

### Informational Metrics

Skipped after the necessary metric condition failed; values are retained in Run 1 Key Metrics.

## Errors & Dead Ends

None.

## Human Notes

> The user requested a fully offline, local workflow with no GitHub CLI or remote PR operations.
