# EXP-017: Neutral Stage-3 Squeeze-and-Excitation

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-017
- **Commit**: (pending - committed on success)
- **PR**: N/A - local-only run
- **Outcome**: failed - valid run missed the accuracy threshold by 0.01 points

## Implementation Notes

### Summary

Added exact-identity ratio-16 SE gates to both accepted stage-3 residual branches after accepted model initialization. Gate creation uses CPU-only seed 17017 inside a restored RNG fork. Nine non-persistent scalar buffers per gate accumulate preregistered training-only diagnostics with constant memory.

### Surprises & Discoveries

The first semantic harness used object equality between generic `cuda` and concrete `cuda:0`, which fails despite correct placement; the production module was already on GPU.

### Decisions

Diagnostics pool all scored training forwards and never update during evaluation. They are measured in production throughput and synchronized only for terminal printing.

## Experimental Adjustments

- **Correct device assertion**: require CUDA device type and FP32 rather than equality to a generic device object. (ref: preflight error before scoring)

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 58458
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed (research no-improvement)
- **Started**: 2026-07-26 17:11:13 UTC
- **Ended**: 2026-07-26 17:17:01 UTC

Description:
- One fixed-seed H20 run tests whether exact-identity stage-3 channel gates improve top-1 without changing accepted representation or exposure. Success requires at least 94.17%; gate diagnostics are observational only.

Observations:
- Semantic preflight passed exact gate count/placement, 696,042 parameters, seed-17017 oracle, common state and CPU/CUDA RNG, identity logits, two-step opening, optimizer groups, and diagnostic formulas. (source: semantic preflight stdout)
- Instrumented timing passed: accepted/candidate weighted 13.044400/13.679807 ms, retention 0.953551, projected 135.308954 passes, all CVs below 0.0043. (source: throughput preflight stdout)
- Scored log capture started on CUDA with exact 696,042 parameter count and 195 batches per epoch. (source: `run.log` startup)
- Mixup disabled once at epoch 86, step 16,744, 195.0 seconds; the run exited 0 with 27 unique evaluation epochs and no error signature. (source: `run.log` L6-L73)
- Realized 133.63712 passes, slightly below the 134.8 preflight projection but with valid timing/integrity; accuracy remains authoritative. (source: `run.log` L69-L70)

Key Metrics:
- best_test_acc: 94.16% at epoch 125, +0.09 over baseline but 0.01 below the 94.17 threshold. (source: `run.log` L56, L62)
- final_test_acc/loss: 94.12% / 0.2321 at epoch 134. (source: `run.log` L60, L63-L64)
- timing/exposure: 300.0 training seconds, 338.5 total seconds, 26,101 steps = 133.63712 passes. (source: `run.log` L65-L70)
- resources/model: 1,094.1 MiB peak VRAM, 696,042 parameters. (source: `run.log` L68, L71)
- gate 0: mean 0.646774, variance 0.03427926, across-example variance 0.003116285, saturation 8.278e-7, feature/bias RMS 0.764061/0.198260. (source: `run.log` L72)
- gate 1: mean 0.869488, variance 0.06345350, across-example variance 0.02430556, saturation 0.0037346, feature/bias RMS 0.807165/0.209579. (source: `run.log` L73)

## Verification Results

### Conditions Checked

- **Completion and integrity**: PASS - exit 0, one H20, correct model/count, 300.0 counted and 338.5 total seconds, one transition, 27 unique evaluation epochs, finite diagnostics, no errors. (source: `run.log` L6-L73)
- **Primary metric >=94.17%**: FAIL - best 94.16%, one hundredth below the required threshold. Verification stopped on this necessary-condition failure. (source: `run.log` L62 and results index)
- **Remaining conditions**: skipped after metric failure; pre-score scope/semantics/diff checks had passed.

### Informational Metrics

- Skipped under the verification guard; values are preserved under Run 1 Key Metrics for analysis.

## Errors & Dead Ends

### 2026-07-26 - Generic CUDA device equality assertion failed
- Error: `AssertionError` at gate parameter device/dtype check.
- Root cause: concrete parameter device `cuda:0` is not object-equal to generic `torch.device("cuda")`; production placement was correct.
- Source: evaluator-free semantic preflight before any scored run.
- Do NOT retry: compare device type/index semantics rather than generic/concrete device object equality.

## Human Notes

> Autopilot local-only execution; no intervention requested.
