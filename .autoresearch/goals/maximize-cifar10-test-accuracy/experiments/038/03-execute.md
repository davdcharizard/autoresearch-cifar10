# EXP-038: Double Only Terminal Classifier Decay

## Execution

Overall Status & Info:
- **Created**: 2026-07-27
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-038
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - offline/local-only run
- **Outcome**: failed - valid normal-exposure metric miss

## Implementation Notes

### Summary

Added `CLASSIFIER_WEIGHT_DECAY = 1e-3` and split the existing matrix-decay group into a 999,856-element representation group at accepted `5e-4` plus the exact 1,280-element `fc.weight` group at `1e-3`. The 2,346 rank-below-2 elements remain at zero decay. No graph, initialization, data, RNG, schedule, loss, evaluator, or training-loop behavior was changed.

### Surprises & Discoveries

The production allocation mapped directly to the planned three named-parameter comprehensions; no implementation dependency or scope expansion emerged. The host has no standalone `python` command, so syntax and preflight commands use the repository's existing `uv run python` environment.

### Decisions

The ignored preflight is derived from EXP037's already validated two-arm harness, but independently loads accepted `a7c42dc`. Its update oracle asserts that only the coupled decay contribution doubles; it does not incorrectly claim that gradient-plus-momentum shrinkage doubles.

## Experimental Adjustments

- None.

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local foreground process)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-27 02:55 UTC
- **Ended**: 2026-07-27 03:01 UTC

Description:
- Sole fixed-seed scored run of classifier-specific `1e-3` coupled decay against the accepted 94.48% baseline. It will launch only after independent semantic and full-body timing gates pass. Success requires best test accuracy at least 94.58%; final accuracy and loss are corroborating metrics, while realized exposure determines the strength of any mechanism conclusion.

Observations:
- Semantic preflight passed with exact common state/RNG/forward/backward identity, exact three-group membership/counts, and independent first/preseeded Nesterov oracles. The maximum accepted/candidate `fc.weight` deltas were `1.4543533e-05` for mixup and `1.4603138e-05` for the preseeded hard step (source: semantic preflight stdout, 2026-07-27).
- Counterbalanced timing passed: retention `0.998113`, projected exposure `130.058` passes, maximum CV `0.005064`, and candidate peak `610.16 MiB` (source: timing preflight stdout, 2026-07-27).
- Launch output confirmed CUDA, 1,003,482 parameters, a 300-second budget, and 195 batches per epoch (source: `run.log` L1-L4).
- The temporal transitions were valid: mixup stopped at step 16,285 and 195.0 seconds; RandAugment stopped only after the epoch-84 iterator exhausted at step 16,380 and 196.1 seconds (source: `run.log` L42-L43).
- The run produced 27 unique evaluations at every fifth epoch plus the final epoch 131, with no duplicate final evaluation (source: `run.log` L5-L62).

Key Metrics:
- `best_test_acc`: `93.82%` versus `94.58%` threshold and `94.48%` baseline (source: `run.log` L64).
- `final_test_acc`: `93.74%`; `final_test_loss`: `0.2598` (source: `run.log` L65-L66).
- Exposure: `25,399` steps = `130.04288` CIFAR-10 passes across 131 epochs (source: `run.log` L71-L72).
- Counted/wall time: `300.0/346.3s`; peak VRAM: `1096.4 MiB`; parameters: `1,003,482` (source: `run.log` L67-L73).

## Verification Results

### Conditions Checked

- **Completion/resource contract - PASS**: exit code 0; CUDA H20; finite summary; `300.0s` counted and `346.3s` wall (<600); one evaluation per epoch maximum; correct temporal transitions; 1,003,482 parameters; 130.04288 passes (source: `run.log` L1-L73).
- **Primary metric improvement - FAIL**: best `93.82%` is `0.66` points below baseline `94.48%` and `0.76` below required `94.58%` (source: `run.log` L64).
- **Corroboration - skipped after necessary metric failure**: observed final `93.74%` and loss `0.2598` are retained in the run metrics but are not separately certified (source: `run.log` L65-L66).

### Informational Metrics

- Skipped under the verification procedure after the primary-metric necessary condition failed; raw values remain inline in Run 1.

## Errors & Dead Ends

- None.

## Human Notes

> User requested uninterrupted autopilot and offline/local-only execution.
