# EXP-001: Time-Aligned Pre-Activation WRN-16-2

## Execution

Overall Status & Info:
- **Created**: 2026-07-24
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-001
- **Commit**: 600764e
- **PR**: N/A — local-only workflow requested
- **Outcome**: completed

## Implementation Notes

### Summary

Replaced the thin post-activation ResNet-20 in `train.py` with a pre-activation WRN-16-2, explicit module initialization, selective optimizer decay, Nesterov SGD, and a five-percent warmup followed by cosine decay keyed to counted training seconds. Enabled deterministic cuDNN benchmarking, efficient gradient clearing, and evaluation every fifth epoch plus the final epoch. Ruff and a CUDA forward/backward throughput benchmark pass; the timed smoke test projected 122.2 dataset-equivalent passes in 300 seconds.

### Surprises & Discoveries

The first smoke test exposed a stale reference to the removed stem `bn1` in `WideResNet.forward`. The pre-activation architecture must feed the raw stem convolution output to its first block. After correcting that line, the synthetic benchmark reached 79.53 steps/s, substantially exceeding the 40-pass feasibility gate.

### Decisions

The plan review identified the baseline's 592.7-second total runtime as a material timeout risk. Evaluation cadence was reduced from every epoch to every fifth epoch plus the final epoch, which is within the user constraint and preserves final low-LR measurement. `cudnn.benchmark` is paired with `cudnn.deterministic` to retain fixed-seed reproducibility.

## Experimental Adjustments

- **Corrected the WRN stem forward path**: Removed the stale `bn1` call and passed the stem convolution output directly to the first pre-activation block. (ref: Milestone 2 first smoke test)
- **Reduced evaluation cadence to every fifth epoch plus final**: Protects the hard 10-minute wall limit after the baseline consumed 592.7 seconds with per-epoch evaluation. (ref: `02-plan-review.md` concern 1)
- **Enabled persistent DataLoader workers for Run 2**: Run 1 timed out after 133 epochs because eight workers were recreated at every epoch boundary; persistence removes that excluded startup cost without changing the experiment hypothesis. (ref: Run 1 timeout)

## Run Log

### Run 1

Metadata:
- **Job ID**: local timeout PID 1117827
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: failed
- **Started**: 2026-07-24 11:22:07 UTC
- **Ended**: 2026-07-24 11:32:07 UTC

Description:
- Run the reviewed pre-activation WRN-16-2 configuration on one NVIDIA H20 for the fixed 300-second counted training budget. The time-aligned schedule should complete warmup and cosine refinement regardless of achieved step count. The expected result is `best_test_acc >= 91.64%`, with a target of at least 92.0%, while total wall time remains below 600 seconds.

Observations:
- Process launched successfully and `run.log` was created. (source: local PID 1117827)
- Warmup reached LR 0.2 at 5% of counted time with stable finite loss and approximately 24.4k images/s. (source: `run.log` L5-L9)
- `best_test_acc` reached 84.43% at epoch 30; an epoch 20-25 dip recovered and did not meet abort criteria. (source: `run.log` L10-L16)
- Accuracy reached a provisional 93.18% at epoch 130, but the outer timeout terminated training at 91% of the counted budget before a final summary. (source: `run.log` L54-L56)

Key Metrics:
- provisional `best_test_acc`: 93.18% @ epoch 130; invalid because the run timed out (source: `run.log` L56)
- last progress: 91.0% counted training @ step 25,900/epoch 133 (source: final progress line in `run.log`)

### Run 2

Metadata:
- **Job ID**: local timeout PID 1124798
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-24 11:34:04 UTC
- **Ended**: 2026-07-24 11:40:23 UTC

Description:
- Retry the identical WRN-16-2 training intervention with `persistent_workers=True` in the existing training DataLoader. Run 1 demonstrated strong research performance but failed the hard wall-time condition because per-epoch worker startup was excluded from the 300-second training accumulator. The retry should preserve the same optimization trajectory while completing the full counted budget and final summary below 600 seconds.

Observations:
- Process launched successfully with persistent training workers enabled. (source: local PID 1124798)
- Persistent workers eliminated the wall-time failure: the complete 300-second training budget plus periodic evaluation finished in 342.5 seconds total. (source: `run.log` L69-L70)
- Accuracy crossed baseline at epoch 120 and peaked at 93.38% at epoch 145; the final epoch retained 93.34%. (source: `run.log` L52-L64)

Key Metrics:
- `best_test_acc`: 93.38% @ epoch 145 (source: `run.log` L62, L66)
- `final_test_acc`: 93.34% @ epoch 147 (source: `run.log` L64, L67)
- `training_seconds`: 300.0; `total_seconds`: 342.5 (source: `run.log` L69-L70)
- `num_steps`: 28,540; `num_epochs`: 147 (source: `run.log` L73-L74)
- `peak_vram_mb`: 1,092.0; `num_params`: 691,674 (source: `run.log` L72, L75)

## Verification Results

### Conditions Checked

- **Run completes without crashing and within 10 minutes**: PASS. Run 2 exited 0 with a complete summary; `total_seconds=342.5 <= 600.0` and `training_seconds=300.0`. (source: local process exit; `run.log` L66-L75)
- **`best_test_acc` exceeds baseline by at least 0.1 percentage points**: PASS. Baseline is 91.54%; actual is 93.38%, a +1.84-point improvement and above the 91.64% threshold. (source: `04-results.tsv` baseline; `run.log` L66)
- **Evaluation cadence and scope**: PASS. Evaluation occurred at epochs divisible by five plus final epoch 147, never more than once per epoch; `git status --short` reports only modified `train.py`. (source: `run.log` L6-L64; git status)
- **Hardware and fixed budget**: PASS. One NVIDIA H20 with 97,871 MiB was used; `prepare.py` remained unchanged and the summary reports 300.0 training seconds. (source: `nvidia-smi`; `run.log` L69)

### Informational Metrics

- `peak_vram_mb`: 1092.0 (source: `run.log` L72)
- `final_test_acc`: 93.34% (source: `run.log` L67)
- `final_test_loss`: 0.2836 (source: `run.log` L68)
- `training_seconds`: 300.0 (source: `run.log` L69)
- `total_seconds`: 342.5 (source: `run.log` L70)
- `num_epochs`: 147 (source: `run.log` L73)
- `num_steps`: 28540 (source: `run.log` L74)
- `num_params`: 691674 (source: `run.log` L75)

## Errors & Dead Ends

### 2026-07-24 — Stale stem normalization reference in smoke test
- Error: `AttributeError: 'WideResNet' object has no attribute 'bn1'`
- Root cause: The model refactor removed stem BN but left the old `F.relu(self.bn1(self.conv1(x)))` expression in `forward`.
- Source: Milestone 2 first CUDA smoke-test traceback at `train.py:96`
- Do NOT retry: Do not add a new stem BN; pre-activation WRN should pass `self.conv1(x)` directly into the first pre-activation block.

### 2026-07-24 — Run 1 exceeded the total wall-time limit
- Error: `timeout 600s` exited 124 at 91% counted training; no final summary was emitted.
- Root cause: The 8-worker training DataLoader recreated worker processes on every one of 133 epochs, adding enough excluded wall overhead to exceed 600 seconds even with validation every fifth epoch.
- Source: Run 1 final `run.log` progress and leaked-semaphore shutdown warning
- Do NOT retry: Do not run high-epoch local training with `num_workers > 0` and the default non-persistent workers under this wall-time constraint.

## Human Notes

> The user requested a fully offline, local workflow with no GitHub CLI or remote PR operations.
