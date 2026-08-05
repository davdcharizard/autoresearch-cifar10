# EXP-010: Selective 160-Channel Final Stage

## Execution

Overall Status & Info:
- **Created**: 2026-07-24
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-010
- **Commit**: (pending — committed on loop success)
- **PR**: N/A — user required fully local/offline execution
- **Outcome**: failed — valid near miss below required improvement margin

## Implementation Notes

### Summary

Starting from `eb08811`, the implementation replaces uniform `WIDEN_FACTOR` with explicit `STAGE_WIDTHS=(32,64,160)`, strictly validates the three positive integer widths, uses them for all stages/final BN/classifier, and logs the exact topology. The residual blocks, FP32 training loop, accepted optimizer/schedule/mixup, seed, data, and evaluator are unchanged.

### Surprises & Discoveries

The full production-path preflight retained 92.34% throughput, better than the proposal's 90.3% affine prior, and projected 131.03 passes. Separating mixup and hard regimes confirmed the candidate overhead was consistent rather than hidden by width-independent mixup work.

### Decisions

Validation uses `type(width) is int` so Boolean values cannot pass as integers. The preflight weights regime medians 65/35 by counted time, matching the scored recipe; only this preregistered aggregate controls feasibility.

## Experimental Adjustments

None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 65660
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-24 15:46 UTC
- **Ended**: 2026-07-24 15:53 UTC

Description:
- One fixed-seed FP32 H20 run will test stage widths `[32,64,160]` with every accepted optimization and regularization choice unchanged. The treatment adds 39% parameters at a measured 7.7% production-throughput cost, targeting abstract 8x8 capacity. Expected exposure is 131.03 passes, above the 120 mechanism gate. Success requires at least 94.17%; there will be no width, LR, batch, precision, or seed rescue.

Observations:
- Static and semantic checks passed: `train.py`-only diff, byte-identical `prepare.py`, fail-closed evaluator, malformed-width rejection, full block/stride/projection topology, exact counts 691,674/961,562, and FP32 optimizer smoke.
- Matched preflight passed. Mixup windows: accepted 10.779223/10.730801/10.785198 and candidate 11.680026/11.713796/11.643109 ms; hard windows: accepted 10.515037/10.667171/10.473852 and candidate 11.298547/11.533313/11.376386 ms. Regime CVs were 0.2263-0.8563%; weighted aggregates 10.686758/11.573752 ms gave retention 0.923362 and 131.0250 projected passes; peak allocation 1179.3 MiB. (source: local preflight stdout)
- Scored startup is healthy on `Device: cuda` with exact `[32,64,160]` and 961,562 parameters. Loss remained finite through step 1,950 and instantaneous throughput was roughly 21.7-22.1k images/s, consistent with the passed retention projection. (source: `run.log` startup through epoch 10)
- Mixup disabled exactly once at epoch 85, step 16,565, 195.0 counted seconds (65.0%) with LR 0.0612. Hard-label throughput was about 22.6k images/s and the trajectory remained near 130 projected passes. (source: `run.log` transition and steps 18,250-19,000)
- The process exited 0 after 300.0 counted / 339.2 total seconds. Best accuracy was 94.11% at epoch 130 with loss 0.2435; final accuracy was 94.06% with loss 0.2457. (source: `run.log` epochs 130/133 and summary)

Key Metrics:

- best_test_acc: 94.11% at epoch 130, delta +0.04 points versus 94.07% baseline but 0.06 below required 94.17% (source: `run.log` epoch 130 and summary)
- final_test_acc: 94.06%; final_test_loss: 0.2457; best-epoch loss: 0.2435 (source: `run.log` epochs 130/133 and summary)
- exposure: 25,812 steps = 132.15744 passes across 133 epochs, above the 120 mechanism gate and near 131.025 projection (source: `run.log` summary)
- total_seconds: 339.2; startup_seconds: 1.1; peak_vram_mb: 1171.4; num_params: 961,562; evaluations: 27 unique epochs (source: `run.log` summary/cadence audit)

## Verification Results

### Conditions Checked

- **Run completion and protocol**: PASS. Exit 0; one H20; exact `[32,64,160]` / 961,562 parameters; 300.0 counted and 339.2 total seconds; 132.16 passes; finite loss; 27 unique accepted-cadence evaluations; one correct transition; complete summary. (source: preflight and `run.log`)
- **Primary metric improvement**: FAIL. Best 94.11% is only +0.04 over baseline and below required 94.17%. Verification stopped; the near miss is not rerun. (source: `run.log` summary)

### Informational Metrics

Skipped by protocol after the primary margin failure; descriptive metrics are retained under Run 1.

## Errors & Dead Ends

None.

## Human Notes

> Autopilot run; no execution-phase intervention requested.
