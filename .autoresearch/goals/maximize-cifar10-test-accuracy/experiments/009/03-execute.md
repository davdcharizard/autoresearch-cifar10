# EXP-009: Isolated BF16 Autocast at Batch 256

## Execution

Overall Status & Info:
- **Created**: 2026-07-24
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-009
- **Commit**: (pending — committed on loop success)
- **PR**: N/A — user required fully local/offline execution
- **Outcome**: failed — valid research result below necessary accuracy threshold

## Implementation Notes

### Summary

Starting from accepted commit `eb08811`, the implementation adds a named `torch.bfloat16` training dtype, logs it once, keeps FP32 mixup interpolation, and wraps only model forward plus cross entropy in CUDA-enabled autocast. Backward, Nesterov SGD, FP32 master parameters/state, schedule, model, batch 256, regularization, and the frozen evaluator remain unchanged.

### Surprises & Discoveries

The plan critic correctly identified that the proposal-development CUDA-resident benchmark omitted production's pinned host copies and fixed per-step work. The hardened line-for-line preflight still reproduced the earlier 1.147x ratio after including copies, LR/group updates, RNG operations, finite checking, backward, optimizer step, and synchronization.

### Decisions

The preflight used two models cloned from identical initialization, independent initially identical RNG streams, exact continuing optimizer state per path, and the preregistered window order. This avoids timing-order selection and makes BF16 enabled/disabled the only treatment. The experiment is interpreted jointly as BF16 numerics plus denser counted-time updates, never as pure exposure.

## Experimental Adjustments

None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 89571
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-24 15:20 UTC
- **Ended**: 2026-07-24 15:27 UTC

Description:
- One fixed-seed local H20 run will test BF16 autocast at the accepted batch 256 with FP32 master state and evaluation. The joint treatment is BF16 training numerics plus denser updates on the unchanged counted-time schedule. Expected exposure is about 162.8 passes versus 141.9 accepted, with the same 65% mixup transition and continuous 0.002 LR floor / matrix decay. Success requires at least 94.17%; there will be no reroll, batch/LR rescue, or precision fallback.

Observations:
- Static audit passed: `train.py`-only allowlisted diff, byte-identical `prepare.py`, local CIFAR data, one H20, PyTorch 2.9.1+cu128, and CUDA BF16 support.
- Fail-closed production-path preflight passed. Windows in preregistered order were FP32 11.050076, BF16 9.482360, BF16 9.563150, FP32 10.758033, FP32 10.877337, BF16 9.472570 ms/step. FP32/BF16 medians were 10.877337/9.482360 ms, population CVs 1.1004%/0.4270%, ratio 1.147113x, projected exposure 162.7753 passes, peak allocation 1099.3 MiB, and all dtype/evaluator checks passed. (source: local preflight stdout)
- Scored startup is healthy on `Device: cuda`, WRN-16-2 with 691,674 parameters, batch 256, and logged `torch.bfloat16` autocast. Loss remained finite through step 1,050 and instantaneous throughput was roughly 25-27k images/s with no error signature. (source: `run.log` startup through epoch 6)
- Mixup disabled exactly once at epoch 102, step 19,814, 195.0 counted seconds (65.0%) with accepted LR 0.0612. The treatment completed about 11% more pre-transition updates than recent FP32 pacing, and hard-label throughput reached roughly 27.5k images/s with finite loss. (source: `run.log` transition and steps 23,150-23,900)
- The process exited 0 with a complete summary after 300.0 counted / 344.3 total seconds. Best accuracy was 93.81% at epoch 150 and final accuracy was 93.78%; the 0.03-point best/final gap is small and there was no runtime or cadence anomaly. (source: `run.log` epoch 150 and final summary)

Key Metrics:

- best_test_acc: 93.81% at epoch 150, delta -0.26 points versus 94.07% accepted (source: `run.log` epoch 150 and final summary)
- final_test_acc: 93.78%; final_test_loss: 0.2634 at epoch 160 (source: `run.log` final evaluation and summary)
- exposure: 31,069 steps = 159.07328 passes across 160 epochs, 12.1% above accepted 141.9 passes and 2.3% below the 162.8 projection (source: `run.log` final summary)
- total_seconds: 344.3; startup_seconds: 1.1; peak_vram_mb: 540.0; num_params: 691,674; evaluations: 32 unique epochs (source: `run.log` final summary and cadence audit)

## Verification Results

### Conditions Checked

- **Run completion and protocol**: PASS. Exit 0; one H20; logged CUDA/BF16 treatment; 300.0 counted and 344.3 total seconds; 31,069 steps below `MAX_STEPS`; finite loss; 32 evaluations at 32 unique accepted-cadence epochs; one correct 65% transition; complete summary. (source: preflight output and `run.log`)
- **Primary metric improvement**: FAIL. `best_test_acc=93.81%`, below baseline 94.07% and required 94.17%. Verification stopped on this necessary-condition failure. (source: `run.log` final summary)

### Informational Metrics

Skipped by protocol because the primary necessary condition failed; descriptive run and preflight metrics are preserved under Run 1.

## Errors & Dead Ends

None.

## Human Notes

> Autopilot run; no execution-phase intervention requested.
