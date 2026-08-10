# EXP-001: Time-Aware Pre-Activation WRN-16-4

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-001
- **Base Node**: BASE
- **Commit**: 1feed19
- **Outcome**: completed

## Implementation Notes

### Summary

Replaced the baseline post-activation ResNet-20 with a six-block pre-activation WRN-16-4 and implemented per-example expectation-preserving drop path. Training now uses batch 256, BF16 autocast, channels-last tensors, Nesterov SGD, and a piecewise LR schedule driven entirely by charged training-time progress. The frozen evaluator, seed, augmentation, input scaling, charged timer, and final metric keys remain unchanged.

### Surprises & Discoveries

The implemented network has exactly 2,748,890 parameters, matching the proposal estimate. On GPU 0, the inline preflight measured a median synthetic training step of 0.011597 seconds and a full frozen evaluation of 0.869836 seconds. This projects approximately 133 epochs and 446 total seconds, so every-epoch evaluation fits comfortably under the plan's 570-second preflight guard.

### Decisions

The shape-changing shortcut consumes the pre-activated input, while identity shortcuts preserve the raw input, matching the intended pre-activation residual design. Drop-path strength is passed explicitly into `forward` rather than stored as mutable module state; the evaluator's unmodified `model(inputs)` call therefore disables drop path by default. `EVAL_EVERY` remains 1 because the runtime-only projection passed; no configuration was selected from accuracy observations.

## Experimental Adjustments

- **Kept every-epoch evaluation**: Preflight projected about 446 total seconds from 11.597 ms median step and 0.870 s evaluation latency, below the 570-second guard. (ref: preflight output recorded in Implementation Notes)

## Run Log

### Run 1

Metadata:
- **Job ID**: local timeout PID 3723310 (training PID 3723311; exec session 47722)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 07:39:45 UTC
- **Ended**: 2026-08-05 07:47:46 UTC

Description:
- Run the reviewed PreAct WRN-16-4 package once on physical GPU 0 under the frozen 300-second charged training budget and 600-second outer timeout. The run tests whether increased residual capacity plus time-normalized optimization converts the H20's unused compute envelope into at least a 0.10-point accuracy gain over BASE. Expected `best_test_acc` is at least 91.61%, with a hypothesis target above 93%.

Observations:
- `run.log` reached 4,854 bytes after approximately 62 seconds with no bounded error-pattern matches. (source: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log`, checked 2026-08-05 07:40:47 UTC)
- The process exited 0 with no NaN/Inf, traceback, CUDA, or memory-error matches. Warmup reached LR 0.2000 at 5.0% progress; the late schedule reached LR 0.0020 and drop-path maximum 0.000 by 99.9%. (source: `run.log` progress records and L300)
- The best evaluation was 94.62% at epoch 147; the mandatory final epoch-148 evaluation was 94.52%. (source: `run.log` L299-L301)

Key Metrics:
- `best_test_acc`: 94.62% at epoch 147, +3.11 percentage points vs BASE (source: `run.log` L299, L303)
- `final_test_acc`: 94.52% at epoch 148 (source: `run.log` L301, L304)
- `training_seconds`: 300.0; `total_seconds`: 471.9 (source: `run.log` L306-L307)
- `num_steps`: 28,790 across 148 epochs; `num_params`: 2,748,890 (source: `run.log` L310-L312)
- `peak_vram_mb`: 1,178.9 MiB (source: `run.log` L309)

## Verification Results

### Conditions Checked

- **Parent-relative accuracy threshold - PASS**: BASE is 91.51%, so the required threshold is 91.61%; EXP-001 reached 94.62%, a +3.11-point gain. (source: `tree.sh show ... BASE`; `run.log` L303)
- **Clean completion and fixed budget - PASS**: launch exited 0, all ten summary keys are present, charged training was 300.0 seconds, total runtime was 471.9 seconds, and no bounded error pattern matched. (source: local exec session 47722; `run.log` L302-L312)
- **Planned model and schedules - PASS**: startup config reports PreActWideResNet, 2,748,890 parameters, peak LR 0.2, warmup fraction 0.05, maximum drop path 0.08, and evaluation cadence 1. LR reached 0.2000 at 5.0% and ended at 0.0020; effective maximum drop path ended at 0.000. (source: `run.log` L3, progress records through L300)
- **Validation cadence - PASS**: 148 `eval ep` records for 148 completed epochs, including final epoch 148. (source: `grep -c 'eval ep' run.log`; `run.log` L301, L310)

### Informational Metrics

- `final_test_acc`: 94.52% (source: `run.log` L304)
- `final_test_loss`: 0.2302 (source: `run.log` L305)
- `training_seconds`: 300.0 s (source: `run.log` L306)
- `total_seconds`: 471.9 s (source: `run.log` L307)
- `startup_seconds`: 1.1 s (source: `run.log` L308)
- `peak_vram_mb`: 1,178.9 MiB (source: `run.log` L309)
- `num_epochs`: 148 (source: `run.log` L310)
- `num_steps`: 28,790 (source: `run.log` L311)
- `num_params`: 2,748,890 (source: `run.log` L312)

## Errors & Dead Ends

## Human Notes

> Autopilot session; no execution-phase human intervention.
