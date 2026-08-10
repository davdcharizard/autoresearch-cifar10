# EXP-009: Exclude BN and Bias from Weight Decay

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-009
- **Commit**: (pending — committed on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Replaced the single all-parameter SGD input with two explicit parameter groups. Trainable parameters are materialized once in model traversal order; 20 matrix/kernel tensors (1,071,200 elements) retain the accepted `1e-4` decay, while 39 one-dimensional BN-affine/bias tensors (2,762 elements) use zero decay. Construction-time assertions verify optimizer tensor count, uniqueness, and total-element coverage against the 1,073,962-parameter model.

### Surprises & Discoveries

The one-dimensional group is only 0.257% of parameters, but repeated coupled shrinkage can still materially alter functional BN scales over roughly 27,000 steps. A read-only reconstruction verified the pre-registered group counts and decay values exactly; no implementation surprise required changing the protocol.

### Decisions

The mandatory Claude plan critic identified that the initial reviewed refinement (`2e-4` kernels plus zero one-dimensional decay) both contradicted the fit-limited diagnosis and confounded scalar strength with targeting. The final implementation keeps kernel decay at the baseline `1e-4` and isolates only the exclusion rule. The preflight ran as an external `uv run python -c` expression, so no diagnostic logging was added to `train.py` and no `run.log` was created.

## Experimental Adjustments

- **Preserved baseline kernel scalar**: Accepted the external plan critic's attribution and fit-limit concerns; only one-dimensional decay differs from EXP-007. (ref: `02-plan-review.md` concerns 1-2)

## Run Log

### Run 1

Metadata:
- **Job ID**: PID 2207529 (timeout supervisor 2207528; tool session 13851)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 14:34 UTC
- **Ended**: 2026-08-05 14:39 UTC

Description:
- One fixed-seed width-2 CIFAR-10 run tests whether excluding BN affine and bias parameters from coupled decay improves finite-horizon fitting and best test accuracy while preserving accepted `1e-4` kernel regularization. All accepted EXP-007 architecture, RandAugment lifecycle, schedule, timing, seed, and evaluation mechanics remain fixed. The expected positive mechanism is an 80% strong-view checkpoint near or above 90.08% followed by a best accuracy of at least 93.65%; a valid lower result will not be rerun.

Observations:
- Preflight found one idle NVIDIA H20 with 97,871 MiB, zero reported memory use, and no compute processes. Static checks, Ruff, formatting, pre-commit, diff scope, and runtime group reconstruction all passed before launch.
- Startup succeeded on CUDA with 1,073,962 parameters; early finite loss fell from about 2.51 at step 50 to 1.30 at step 550 with approximately 11 ms synchronized steps. (source: `run.log`, startup and initial progress records)
- Checkpoints rose 84.68% -> 86.95% -> 89.43% through 60%, then ended the strong phase at 88.26%. The 80.0% switch stopped all eight workers, and the weak tail jumped to 92.71% before peaking at 93.52% on epoch 65. (source: `run.log` L6-L35)
- The process exited 0 after 300.0 counted seconds and 333.2 total seconds. No traceback, OOM, `nan`, or `inf` pattern was present, and no GPU compute process remained after exit. (source: `run.log` L45-L54; process exit status)

Key Metrics:
- best_test_acc: 93.52% @ epoch 65 (source: `run.log` L33-L35, L45)
- final_test_acc: 93.50% @ epoch 70 (source: `run.log` L43, L46)
- final_test_loss: 0.2340 @ epoch 70 (source: `run.log` L43, L47)
- final train-loss EMA: 0.0303 @ step 27,150 (source: `run.log` final progress record)
- training/total/startup: 300.0s / 333.2s / 1.0s (source: `run.log` L48-L50)
- peak_vram_mb: 598.7 MB (source: `run.log` L51)
- exposure: 70 epochs / 27,172 steps / 1,073,962 parameters (source: `run.log` L52-L54)

## Verification Results

### Conditions Checked

- **Primary accuracy improvement**: failed. Actual `best_test_acc=93.52%`; moving baseline 93.55%, required threshold 93.65%. The result is -0.03 percentage points versus baseline and -0.13 below the gate. (source: results-index baseline query; `run.log` L45)
- **Completion and numeric summary**: passed as a run-validity check completed before verdict. Process exit 0; all ten expected numeric keys were present and finite. (source: process exit status; `run.log` L45-L54)
- **Fixed budget and total timeout**: passed as a run-validity check. Counted training was 300.0s and total was 333.2s, below 600s. (source: `run.log` L48-L50)
- **Hard-constraint integrity**: passed. Exactly one idle H20 was confirmed; only `train.py` changed; seed/evaluator/timer remained fixed; 19 evaluations occurred on 19 unique epochs; one switch occurred at 80.0% with eight workers stopped. (source: preflight outputs; git diff; `run.log` L6-L43)
- Remaining success evaluation stopped at the failed primary gate; the valid result must not be rerun.

### Informational Metrics

- Skipped as formal success-only metrics because the primary necessary condition failed. Run metrics needed for no-improvement analysis are preserved above under Run 1 Key Metrics.

## Errors & Dead Ends

## Human Notes

> Autopilot session; no execution-phase intervention.
