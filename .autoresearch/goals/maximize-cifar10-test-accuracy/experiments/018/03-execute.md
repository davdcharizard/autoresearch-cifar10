# EXP-018: Final-Block-Only Neutral SE

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-018
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - local-only run
- **Outcome**: failed - valid run scored below baseline and acceptance threshold

## Implementation Notes

### Summary

Added one ratio-16 squeeze-and-excitation gate to the residual branch of `layer3[1]` only. The accepted WRN is fully initialized first; gate construction then uses the fixed project seed 42 inside a CPU-only restored RNG fork, with an exact-zero second projection making the initial scale exactly one. Production contains no gate diagnostic state or extra summary work.

### Surprises & Discoveries

The independent plan review correctly identified a separate experiment-derived gate seed as an unnecessary trajectory knob. Reusing the fixed training seed 42 removes that ambiguity while retaining deterministic, isolated initialization.

### Decisions

Gate behavior is verified through evaluator-free initial-function and two-step-gradient oracles rather than runtime instrumentation. This separates implementation validation from the treatment and maximizes scored exposure.

## Experimental Adjustments

- **Use fixed seed 42 for gate initialization**: removes an independent seed knob and satisfies the no-reroll constraint. (ref: `02-plan-review.md` concern 1)

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 88076
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed (research no-improvement)
- **Started**: 2026-07-26 17:38:58 UTC
- **Ended**: 2026-07-26 17:45:21 UTC

Description:
- One fixed-seed local H20 run will test whether retaining only EXP-017's more conditional final stage-3 SE gate preserves its accuracy signal while recovering exposure. Success requires at least 94.17% with the accepted data, optimizer, schedule, evaluation, and timing protocol unchanged. A complete valid run will not be rerun or tuned.

Observations:
- Production scope/compile audit passed before semantic preflight: `train.py` is the sole changed tracked production file and the diff is 24 insertions. (source: local audit stdout)
- Semantic preflight passed exact single-gate placement, 693,858 parameters, fixed-seed-42 oracle, common state/logits, CPU/CUDA RNG, shortcut, optimizer, two-step opening, and absence of diagnostics. (source: semantic preflight stdout)
- Matched timing passed: accepted/candidate weighted steps were 12.478465/12.658247 ms, retention 0.985797, worst CV 0.000783. The absolute synthetic projection was 121.34 passes and is informational; normalized retention over accepted production exposure implies about 139.9 passes. (source: throughput preflight stdout)
- The scored run started on one H20 with the exact 693,858 parameter count and completed once with exit 0. (source: `run.log` startup and process status)
- Mixup disabled exactly once at epoch 89, step 17,245 and 195.0 seconds; all 28 evaluated epochs were unique and no error signature appeared. (source: `run.log` transition/evaluation lines)
- The candidate realized 26,920 steps = 137.8304 dataset-equivalent passes, satisfying the exposure hypothesis but not the accuracy hypothesis. (source: `run.log` final summary)

Key Metrics:
- best_test_acc: 93.67% at final epoch 139, -0.40 points from baseline and -0.50 below threshold. (source: `run.log` final evaluation/summary)
- final_test_acc/loss: 93.67% / 0.2468. (source: `run.log` final summary)
- timing/exposure: 300.0 training seconds, 340.9 total seconds, 26,920 steps = 137.8304 passes. (source: `run.log` final summary)
- resources/model: 1,094.0 MiB peak VRAM and 693,858 parameters. (source: `run.log` final summary)

## Verification Results

### Conditions Checked

- **Completion and integrity**: PASS - exit 0, one H20, correct model/count, 300.0 counted and 340.9 total seconds, one transition, 28 unique evaluation epochs, finite summary, and no errors. (source: `run.log`)
- **Primary metric >=94.17%**: FAIL - best 93.67%, 0.40 below the accepted 94.07 baseline and 0.50 below threshold. Verification stopped on this necessary-condition failure. (source: `run.log` final summary and results index)
- **Remaining conditions**: skipped after metric failure; pre-score scope, semantics, and diff checks had passed.

### Informational Metrics

- Skipped under the verification guard; values are preserved under Run 1 Key Metrics for analysis.

## Errors & Dead Ends

## Human Notes

> Autopilot local-only execution; no intervention requested.
