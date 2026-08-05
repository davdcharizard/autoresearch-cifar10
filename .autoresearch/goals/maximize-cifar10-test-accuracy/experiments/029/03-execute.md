# EXP-029: Batch 128 With a Fully Scaled LR Curve

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-029
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - offline/local-only session
- **Outcome**: failed

## Implementation Notes

### Summary

Changed exactly four constants: batch 128, peak/floor LR 0.1/0.001, and a doubled nonbinding step cap. All accepted model, transform, optimizer-family, temporal schedule, RNG, loader, timing, and evaluator code remains unchanged.

### Surprises & Discoveries

The same 50,000-image dataset drops exactly 80 images at both batch sizes, so full epochs remain 49,920 images despite doubling from 195 to 390 batches. This preserves example-domain epoch/evaluation meaning while intentionally changing optimizer, BN, mixup, momentum, and decay update frequency.

### Decisions

GPU timing reconstructs every arm/window from fresh deterministic fixtures and includes pinned-host copies inside the scored region. Loader timing reruns the GPU measurement locally to obtain consumer pacing without writing cross-command state, then projects excluded stall separately from the fixed 300-second counted budget.

## Experimental Adjustments

- None after plan review.

## Run Log

### Run 1

Metadata:
- **Job ID**: pending
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: failed pre-score feasibility gate; scored run not launched
- **Started**: N/A - preflights only
- **Ended**: 2026-07-26 22:26:58 UTC

Description:
- One offline local H20 score of the exact batch-128 / fully scaled `0.1 ->0.001` LR operating point on accepted EXP-027. It tests whether more full-model optimizer/BN/mixup decisions improve boundary quality while retaining at least 120 passes. It launches only after independent semantics, complete-body image-rate, update-count, and real-loader wall gates pass.

Observations:
- Semantic preflight passed against independent `git show 67c8e98:train.py`: exact four constants, byte-equal initialization/RNG, 987,098 parameters, half-LR curve, 390 batches / 49,920 images, finite update, and paired batch-128 tail replay.
- Balanced full-body GPU timing had all regime/arm CVs <=5%, then failed the first retention assertion: `retention <0.9022`, implying `projected_passes <119.99924019` and `projected_updates <46874.703` versus fixed 120 / 46,875 gates. Loader timing and scoring were aborted without retry.

Key Metrics:
- No scored accuracy metric. Stable timing bound: retention `<0.9022`, projected passes `<119.99924019`, projected updates `<46874.703`.

## Verification Results

### Conditions Checked

- PASS: exact four-line scope, independent accepted oracle, batch/LR semantics, finite update, and paired batch-128 worker clean-tail replay.
- FAIL: stable complete-body image-rate retention was `<0.9022`, forcing projected passes `<119.99924019` and updates `<46874.703` below preregistered floors.
- SKIPPED after prior failure: loader wall timing, scored run, completion, transitions, cadence, and accuracy conditions.

### Informational Metrics

- No scored metrics; `run.log` was never created.

## Errors & Dead Ends

### 2026-07-26 - Batch-128 image-rate feasibility miss
- Error: `AssertionError` at `retention >= 0.9022` after all timing CV assertions passed.
- Root cause: batch 128 did not halve complete scored-body step time enough to retain the fixed image-exposure/update regime on the H20. The verifier asserted before printing its payload, so exact windows were not emitted; the failed inequality still proves the fixed pass/update floors were missed.
- Source: EXP-029 throughput preflight traceback at `preflight.py:340`; threshold-derived upper bounds recorded above.
- Do NOT retry: do not rerun stable timing, lower the exposure/update floors, or repair the LR/floor/momentum; future timing harnesses must print measurements before assertions.

## Human Notes

> Autopilot run; no intervention requested.
