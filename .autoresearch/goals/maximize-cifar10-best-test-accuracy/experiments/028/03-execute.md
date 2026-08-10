# EXP-028: Signal-Scale-Matched Positive-Negative Momentum

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-028
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Implemented the reviewed optimizer intervention in tracked `train.py` only. `ScaleMatchedPNM` retains two zero-initialized alternating buffers per parameter, applies all-parameter coupled decay before a `rho=0.81` recurrence, and analytically scales the paper's `2*current-previous` direction to match accepted PyTorch momentum under a constant direction at every step. The production call site, elapsed LR assignment, timer boundary, model/data curriculum, evaluator, seed, precision, and worker lifecycle remain unchanged; only three scalar optimizer-integrity diagnostics were appended after the standard summary.

### Surprises & Discoveries

The complete PNM update can use four multi-tensor operations: one out-of-place decay direction, one in-place active-stream recurrence, one out-of-place raw direction, and one in-place parameter update. This avoids a Python parameter-update loop while retaining exact per-parameter state and the approved recurrence. The scale sequence computed by the implementation begins `5.884389, 22.360680, 12.173050, 22.360680` as derived.

### Decisions

- Store `pnm_step` in the sole optimizer parameter group so ordinary `state_dict()` serialization preserves global parity alongside per-parameter odd/even buffers.
- Enforce missing/sparse/non-FP32 gradient contracts without production device synchronization. Finite scans remain in preflight; timing and production both execute the same unsynchronized optimizer recurrence.
- Keep the review-added 1.30 median gate on total parameter-update norm. It does not constrain the analytically larger coefficient on the newest gradient component directly.

## Experimental Adjustments

- **Goal-aligned production verdict**: Actual steps below the hypothesized 26,091 floor are reported as an exposure-hypothesis miss but do not invalidate an otherwise fixed-budget accuracy improvement. Fresh paired timing ratio governs the pre-production cost gate. (ref: `02-plan-review.md` concerns 2 and 4)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local preflight)
- **Log file(s)**: `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/028/preflight.log`, `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/028/preflight-report.json`
- **WandB**: N/A
- **Status**: failed
- **Started**: 2026-08-06 18:09:19 UTC
- **Ended**: 2026-08-06 18:10:14 UTC

Description:
- Algebraic and immutable-corpus H20 preflight of signal-scale-matched PNM against installed PyTorch momentum SGD. The controller will verify recurrence/state semantics and run aligned models over the registered 200 strong batches plus one persisted 64-batch weak corpus. Timing and production are authorized only if every registered safety gate passes.

Observations:

- The immutable strong corpus matched the registered SHA-256 `e04dc2f...8946`; the newly persisted 64-batch hard weak corpus has SHA-256 `ffefe980...5032`. All 264 aligned model steps completed with finite state, valid BN counters, unchanged gradients/RNG/corpora, and exact PNM global step count. (source: `preflight-report.json` fields `strong_file_sha256`, `weak_file_sha256`, `corpora_unchanged`, and `trajectory` integrity fields)
- The installed foreach recurrence matched the manual changing-gradient implementation exactly at the parameter level (`0.0` max error; buffer error `7.45e-09`), first-step parameters matched SGD within `2.98e-08`, and all four registered optimizer state roundtrips passed. The constant-direction delta had `9.54e-07` maximum absolute error but `4.19e-06` relative error, narrowly failing the predeclared relative tolerance because the compared FP32 delta was small. (source: `preflight-report.json` fields `constant_direction_oracle` and `changing_direction_oracle`)
- PNM produced 157 candidate-only concentration events. The first occurred at step 3 with 100% candidate versus 86.72% control class share; the last persisted at step 264 with 97.66% candidate versus 15.62% control. (source: `preflight-report.json` field `trajectory.concentration_failures`)
- The maximum candidate/control update ratio was 12.3456 at step 5: candidate update norm 12.9813 versus 1.05149 control, candidate loss 41.0337 versus 10.0437, and candidate class share 100% versus 55.47%. Median update ratio was only 0.7128, showing that the failure came from severe alternating spikes rather than uniformly enlarged updates. (source: `preflight-report.json` field `trajectory.records`, step 5, and update-ratio summary fields)
- The controller exited on the serialized registered gates; paired timing and the scored production run were not launched. (source: `preflight.log` L1-L7)

Key Metrics:

- candidate-only concentration events: 157 / 264 aligned steps; first at step 3 (source: `preflight-report.json`)
- maximum candidate/control update ratio: 12.345637 at step 5 (source: `preflight-report.json`)
- median / p95 update ratio: 0.712803 / 6.697578 (source: `preflight-report.json`)
- strong loss-EMA candidate/control ratio: 1.138453; weak ratio: 1.347240 (source: `preflight-report.json`)
- scored `best_test_acc`: not measured — production blocked by pre-registered safety gates.

## Verification Results

### Conditions Checked

- Scored verification skipped — execution failed at mandatory immutable-corpus safety conditions before timing or production.

### Informational Metrics

## Errors & Dead Ends

### 2026-08-06 — Scale-matched PNM caused severe alternating class collapse
- Error: `preflight gates failed: ['constant-direction relative error', 'candidate-only class concentration', 'paired update spike']`
- Root cause: Matching coherent constant-gradient scale did not control PNM's changing-gradient geometry. Alternating history generated a 12.35x paired update at step 5 and repeated one-class predictions through both strong and weak batches despite correct first-step/state recurrence.
- Source: `preflight.log` L1-L7; `preflight-report.json` fields `trajectory.concentration_failures`, `trajectory.update_ratio_max`, and `trajectory.records`.
- Do NOT retry: Retire beta0=1 scale-matched `+2/-1` PNM at global LR 0.1; do not weaken concentration/spike gates, reroll the corpus/seed, clip or norm-match as an in-experiment rescue, or launch production.

## Human Notes

> Autopilot session; no human intervention requested.
