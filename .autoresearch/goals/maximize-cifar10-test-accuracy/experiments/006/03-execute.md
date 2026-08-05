# EXP-006: Early p=0.10 WRN Block Dropout

## Execution

Overall Status & Info:
- **Created**: 2026-07-24
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-006
- **Commit**: pending; committed only on loop success
- **PR**: N/A; this workflow is explicitly local-only
- **Outcome**: failed

## Implementation Notes

### Summary

Added p=0.10 dropout to the learned residual branch of each of the six `PreActBlock` modules, between the second BN/ReLU and `conv2`. The probability is threaded explicitly at construction, and the existing 65% mixup transition calls `WideResNet.set_block_dropout(0.0)` once so the accepted clean hard-label tail has neither masks nor dropout RNG overhead. No parameter, optimizer, schedule, input-mixup, loader, evaluator, or dependency behavior changed.

### Surprises & Discoveries

Importing `train.py` normally constructs the frozen evaluator at module scope, which would unnecessarily construct the test dataset during semantic and timing preflights. Both checks inserted an inert synthetic `prepare` module before importing the actual implementation, preserving code fidelity without loading or inspecting test data. The order-balanced H20 timing found only 1.53% dropout overhead, well below the preregistered 5% allowance.

### Decisions

`set_block_dropout` traverses the six block modules only at the one-shot transition; no traversal occurs per optimizer step. A `dropout_p > 0.0` guard bypasses `F.dropout` entirely after the transition, which the CUDA RNG-state check confirmed. The timing harness used identical initialized weights, optimizer structure, fixed synthetic inputs, condition-specific warmup, and alternating measurement order as hardened by the plan review.

## Experimental Adjustments

- **Stubbed `prepare` during unscored preflights**: Avoided module-scope evaluator construction while testing the actual `train.py` model implementation; no test examples or metrics were accessed. (ref: `02-plan-review.md` concern 1)
- **Order-balanced throughput gate**: Three 50-step windows per condition yielded p=0 medians of 12.567 ms and p=0.10 medians of 12.763 ms, a 98.47% throughput ratio and 139.73 calibrated passes. (ref: preflight command output before Run 1)

## Run Log

### Run 1

Metadata:
- **Job ID**: PID 1145799 (execution session 39434)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-24 13:54:40 UTC
- **Ended**: 2026-07-24 14:00:22 UTC

Description:
- One fixed-seed local run of the accepted WRN-16-2 recipe with p=0.10 residual-branch dropout during the first 65% of counted training. Mixup and dropout will be disabled together for the final 35% hard-label tail. The expected result is a valid 300-second run retaining at least 26,329 steps and reaching at least 94.17% best test accuracy.

Observations:
- Preflight passed on one NVIDIA H20: six blocks, 691,674 parameters, correct train/eval/zero semantics, 98.47% relative throughput, and 139.73 calibrated passes.
- The redirected log initialized normally on CUDA and reached step 350 without an error, at about 1.6% counted progress and 11 ms/step. (source: `run.log` initial bounded extract)
- Mixup and block dropout disabled exactly once at epoch 90, step 17,422, 195.0 counted seconds (65.0%), with LR 0.0612. The clean tail then ran at about 24k images/s. (source: `run.log` L40)
- The run completed normally at epoch 141 after 27,361 steps. Accuracy improved through the clean tail but ended below the accepted baseline. (source: `run.log` L42-L75)

Key Metrics:
- `best_test_acc`: 93.52% at epoch 141; baseline 94.07%, delta -0.55 points. (source: `run.log` L64-L66)
- `final_test_acc`: 93.52%; `final_test_loss`: 0.2718. (source: `run.log` L64-L68)
- `training_seconds`: 300.0; `total_seconds`: 341.9; `startup_seconds`: 1.1. (source: `run.log` L69-L71)
- `num_steps`: 27,361; realized exposure: 140.09 passes; `num_epochs`: 141. (source: `run.log` L73-L74)
- `peak_vram_mb`: 1,214.0; `num_params`: 691,674. (source: `run.log` L72-L75)

## Verification Results

### Conditions Checked

- **Run completion and 10-minute limit — PASS**: complete summary, finite loss, 300.0 counted seconds, and 341.9 total seconds on one NVIDIA H20. The run used 29 unique evaluations, every fifth epoch plus final epoch 141, with no duplicate epoch. (source: `run.log` L6-L75; verification command output)
- **Improve baseline by at least 0.10 points — FAIL**: baseline 94.07% requires at least 94.17%; actual `best_test_acc` was 93.52%, 0.55 points below baseline and 0.65 points below threshold. Verification stopped after this necessary-condition failure. (source: results index baseline output; `run.log` L66)
- **Scope/integrity checks — recorded before metric verdict**: exactly one 65.0% transition, 27,361 steps exceeds the 26,329-step exposure floor, 691,674 parameters, one `train.py` diff, lint/compile/diff checks pass, and no evaluator/seed/dependency change. These support mechanism attribution but do not rescue the failed primary condition.

### Informational Metrics

- Skipped as formal verification metrics after the primary necessary condition failed; all run values were preserved inline in `Run 1 > Key Metrics` before `run.log` cleanup.

## Errors & Dead Ends

None.

## Human Notes

No user intervention; autopilot remained local and offline.
