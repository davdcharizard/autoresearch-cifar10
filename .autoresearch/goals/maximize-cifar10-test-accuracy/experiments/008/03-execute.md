# EXP-008: Decoupled Cosine-to-Zero Floor

## Execution

Overall Status & Info:
- **Created**: 2026-07-24
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-008
- **Commit**: (pending — committed on loop success)
- **PR**: N/A — user required fully local/offline execution
- **Outcome**: failed — valid research result below necessary accuracy threshold

## Implementation Notes

### Summary

Starting from accepted commit `eb08811`, the implementation added a distinct `WARMUP_START_LR = 0.002`, changed only the post-warmup floor to `MIN_LR = 0.0`, used the warmup-start constant in the linear warmup, and initialized SGD from that same value. This is the exact four-edit allowlist from the reviewed plan; architecture, data, RNG operations, mixup, optimizer family, momentum, both decay groups, and evaluation cadence remain accepted.

### Surprises & Discoveries

No implementation surprise occurred. Direct production-function assertions confirmed the schedule at 0%, 2.5%, 4.9%, 5%, 65%, 90%, 95%, and 100%; checking interior warmup points was added during plan review to prove the full accepted warmup path rather than only its endpoints.

### Decisions

No deviation from the reviewed plan was required. No additional telemetry was added because it would exceed the exact diff allowlist; transition LR and final progress logging already provide sufficient schedule evidence.

## Experimental Adjustments

None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 26841
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-24 14:45 UTC
- **Ended**: 2026-07-24 14:52 UTC

Description:
- One fixed-seed local H20 run will test whether preserving the accepted warmup while annealing the post-warmup cosine to zero improves late hard-label settling. The run retains continuous matrix decay and all accepted stochastic behavior. Expected exposure is near 27,735 steps / 141.9 passes, with the mixup transition near 195 seconds at LR 0.0598. Success requires at least 94.17% best test accuracy; there will be no reroll or result-conditioned retry.

Observations:
- Preflight passed on one NVIDIA H20: syntax, exact eight-point production schedule, complete four-edit diff allowlist, accepted base, and byte-identical `prepare.py` all verified before launch.
- Startup is healthy on `Device: cuda` with 691,674 parameters and the frozen 300-second budget; finite loss fell from 2.0215 at step 50 to 1.3698 at step 350 with about 11 ms/step and no error signature. (source: `run.log` startup and progress records)
- Mixup disabled exactly once at epoch 92, step 17,867, 195.0 counted seconds (65.0%) with LR 0.0598; subsequent hard-label loss remained finite and throughput rose to about 24.3k images/s. (source: `run.log` transition and steps 18,200-19,050)
- The process exited 0 with a complete summary after 300.0 counted / 341.0 total seconds. Best accuracy was 93.80% at epoch 135 and final accuracy was 93.78%; no runtime, exposure, or cadence anomaly occurred. (source: `run.log` eval epoch 135 and final summary)

Key Metrics:

- best_test_acc: 93.80% at epoch 135, delta -0.27 points versus 94.07% accepted (source: `run.log` eval epoch 135 and final summary)
- final_test_acc: 93.78%; final_test_loss: 0.2629 at epoch 143 (source: `run.log` final evaluation and summary)
- exposure: 27,833 steps = 142.50496 passes across 143 epochs; 300.0 counted seconds (source: `run.log` final summary)
- total_seconds: 341.0; startup_seconds: 1.1; peak_vram_mb: 1094.0; num_params: 691,674 (source: `run.log` final summary)

## Verification Results

### Conditions Checked

- **Run completion and protocol**: PASS. Exit 0; `Device: cuda` on one visible NVIDIA H20; 300.0 counted seconds; 341.0 total seconds; 27,833 steps below `MAX_STEPS`; finite final loss; 29 evaluations at 29 unique epochs; one correct 65% transition; complete summary. (source: preflight output and `run.log`)
- **Primary metric improvement**: FAIL. `best_test_acc=93.80%`, below both baseline 94.07% and required 94.17%. Verification stopped on this necessary-condition failure. (source: `run.log` final summary)

### Informational Metrics

Skipped by protocol because the primary necessary condition failed; descriptive run metrics are preserved under Run 1 Key Metrics.

## Errors & Dead Ends

None.

## Human Notes

> Autopilot run; no execution-phase intervention requested.
