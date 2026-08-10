# EXP-025: Identity-Initialized Final-Stage ECA

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-025
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Added three length-5 ECA gates to the accepted width-2 ResNet-20, only on `layer3` residual branches immediately before shortcut addition. Each zero-start kernel produces an exact unit `2*sigmoid` gate. The candidate has 1,073,977 parameters and preserves shared state, RNG, stage geometry, Option-A ratios, and initial logits against a persisted pre-edit `7c1e7d8` oracle.

### Surprises & Discoveries

Constructing `Conv1d` inside CPU `fork_rng` is sufficient to preserve the post-construction RNG stream because the model-wide initializer touches only `Conv2d` and `Linear`; explicit zeroing then defines the gate start.

### Decisions

- Make ECA an explicit `ResNet(..., use_eca=True)` option so controllers can construct the edited-path control with `use_eca=False` while production opts in.
- Preserve the paper-motivated length-5 local channel interaction and ordinary optimizer group; no special gate LR, decay, warmup, or clamp.

## Experimental Adjustments

- **Tight FP32 shared-gradient tolerance**: Two cold full-controller checks found bitwise-equal state/RNG/logits but `conv1.weight` gradient max-absolute difference `7.4505806e-08` and relative norm `3.74015372e-07`; an independent critic approved fixed per-tensor bounds `1e-7` and `1e-6` plus exact BN-buffer equality. (ref: `02-plan-identity-review.md`)
- **No recruitment rescue**: The reviewed tolerance enabled the intended 200-step test, which then failed the pre-registered gate range/mean bounds. Timing and production were skipped without changing bounds, gate count, kernel, scale, or optimizer. (ref: `preflight-report.json`)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A — production not launched
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/025/preflight-report.json`
- **WandB**: N/A
- **Status**: failed before timing/production
- **Started**: 2026-08-06 15:52 UTC
- **Ended**: 2026-08-06 15:54 UTC

Description:
- One seed-42 H20 run of three identity-scale final-stage ECA gates on the accepted CIFAR-10 recipe, conditional on exact-corpus recruitment and paired timing gates. Formal success requires `best_test_acc >=94.25%`; hard/soft gate statistics and the switch/tail trajectory diagnose the mechanism.

Observations:

- Pre-edit oracle and static checks passed: 1,073,977 parameters, three zero length-5 kernels, exact shared state/RNG/logits and BN changes, with worst hard shared-gradient difference `9.6858e-08` absolute and `3.5529e-07` relative, inside independently reviewed bounds. (source: `baseline-identity-*`; `preflight-report.json`)
- First hard/soft updates passed: maximum ECA weight changes were 0.007176/0.005448 and every gate remained close to one. (source: `preflight-report.json` first_updates)
- Recruitment then saturated rapidly. By hard step 19 the three blocks reached global gate range `[0,2]`; the last recorded soft step 200 had maxima 1.7092/1.9135/1.9997 and block means 1.0398/1.1516/1.3604. (source: `preflight-report.json` gate_records)
- No candidate-only class concentration occurred and all state remained finite. Candidate/control terminal loss-EMA ratio was 1.083684, but these facts cannot override repeated gate range and mean vetoes. (source: `preflight-report.json`)
- Timing and production were not launched; no `run.log` was created. (source: execution status)

Key Metrics:

- preflight status: failed on gate recruitment bounds (source: `preflight-report.json`)
- hard gate global range through step 19: `[0.0,2.0]`; maximum block-mean deviation: 0.13307 (source: `preflight-report.json`)
- soft step-200 block means: 1.03978/1.15162/1.36038; global range `[0.00498,1.99974]` (source: `preflight-report.json`)
- candidate/control terminal loss-EMA ratio: 1.083684; concentration events: 0 (source: `preflight-report.json`)
- best_test_acc: unavailable; production not launched (source: execution status)

## Verification Results

### Conditions Checked

- **Primary metric improvement — skipped**: mandatory ECA recruitment safety failed before production.
- **Completion/numeric summary — skipped**: production was not launched.
- **Fixed budget and <10-minute runtime — skipped**: production was not launched.

### Informational Metrics

## Errors & Dead Ends

### 2026-08-06 — Cold CUDA backward was not bitwise-identical
- Error: `hard shared gradient mismatch: conv1.weight; max_abs=7.4505806e-08; relative_norm=3.74015372e-07`
- Root cause: The mathematically zero ECA derivative path changes cold CUDA backward accumulation order at sub-epsilon scale despite exact forward/state/RNG identity; a separately warmed deterministic diagnostic was bitwise exact.
- Source: preflight attempts 1-2; `02-plan-identity-review.md`
- Do NOT retry: Do not require cross-graph bitwise gradient equality or relax beyond the reviewed `max_abs<=1e-7` AND relative-norm `<=1e-6` bounds.

### 2026-08-06 — Identity-scale ECA gates saturated during recruitment
- Error: `trajectory gate range/mean failed from steps 4-5 onward`
- Root cause: Ordinary LR-0.1 SGD rapidly drove the three zero-start channel kernels from near-unit gates toward sigmoid endpoints; hard gates reached `[0,2]` and late soft block means rose as high as 1.3604.
- Source: `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/025/preflight-report.json`
- Do NOT retry: Do not relax bounds or rescue EXP-025 with fewer gates, special gate LR/decay, another kernel/scale, clipping, or warmup.

## Human Notes

> Autopilot session; no human intervention requested.
