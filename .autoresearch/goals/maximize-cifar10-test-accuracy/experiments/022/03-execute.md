# EXP-022: Alternating Final-Ten-Percent SAM

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-022
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - local-only run
- **Outcome**: failed - scored accuracy missed the acceptance threshold

## Implementation Notes

### Summary

Added a pure final-window/parity predicate and a rho-0.05 SAM second-backward helper to `train.py`. Alternating SAM begins only on an even optimizer step after 90% counted progress; the helper restores parameters and post-first-forward BatchNorm buffers before the existing single Nesterov step.

### Surprises & Discoveries

The independent gradient oracle initially failed because `deepcopy` does not preserve parameter `.grad` buffers; explicitly snapshotting first-pass gradients fixed the harness. Separately allocated deterministic CUDA models differed by up to 4.29e-5 in a gradient element, so the oracle uses a 5e-5 absolute bound that remains far below an accumulated first-pass gradient.

### Decisions

The preflight times the actual alternating normal/SAM pattern rather than combining dense SAM and normal timings. It measured 12.5140 ms normal versus 20.2549 ms alternating average step time, projecting 94.1746% whole-run exposure retention and 133.6337 passes.

## Experimental Adjustments

- **Explicit oracle gradient snapshot**: fixed the preflight-only `deepcopy` behavior without changing production code. (ref: semantic preflight attempt 1)

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 53605
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-26 19:08:08 UTC
- **Ended**: 2026-07-26 19:14:14 UTC

Description:
- One fixed-seed local score of alternating rho-0.05 SAM during the final 10% counted-time window. The accepted WRN/mixup trajectory is unchanged before that boundary. Expected outcome is at least 94.17% best accuracy while retaining most fixed-time exposure.

Observations:
- Semantic preflight passed exact restoration, one persistent BatchNorm update, pure second-pass gradients within 4.29e-5 of an independent oracle, the strict window/parity predicate, unchanged optimizer groups, and 691,674 parameters. (source: semantic preflight stdout)
- Alternating-pattern timing passed with 0.941746 projected retention and 133.633698 passes. (source: throughput preflight stdout)
- Scored log initialized on one CUDA device with WRN-16-2, 691,674 parameters, a 300-second budget, and 195 batches per epoch. (source: `run.log` initial lines)
- Mixup disabled once at 195.0 counted seconds; alternating SAM activated once at 270.0 seconds on even step 24,970. (source: `run.log` lines 42 and 58)
- The run completed without errors in 340.4 seconds wall. Post-SAM evaluations were 93.35%, 93.79%, and 93.76%, so the intervention did not recover the accepted top-1 trajectory. (source: `run.log` lines 60-75)

Key Metrics:
- Preflight normal/alternating step time: 12.513996/20.254894 ms, CV 0.000808/0.008596. (source: throughput preflight stdout)
- best_test_acc: 93.79% at epoch 135; final_test_acc: 93.76%; final_test_loss: 0.2329. (source: `run.log` lines 62-68)
- training/total time: 300.0/340.4 seconds; steps/epochs/passes: 26,755/138/136.9856. (source: `run.log` lines 69-74)
- peak_vram_mb: 1094.0; num_params: 691,674. (source: `run.log` lines 72 and 75)

## Verification Results

### Conditions Checked

- **Completion/runtime**: PASS - exit 0, finite summary, 300.0 counted seconds, 340.4 wall seconds, one H20, correct parameters/transitions, and no duplicate epoch evaluations. (source: `run.log` lines 58-75 and verification commands)
- **Primary metric**: FAIL - `best_test_acc=93.79%`, below both baseline 94.07% and required 94.17%. Verification stopped at this failed necessary condition. (source: `run.log` line 66)
- **Final post-metric diff audit**: skipped after the primary-metric failure; the pre-run scope audit had shown only planned `train.py` changes.

### Informational Metrics

## Errors & Dead Ends

### 2026-07-26 - Preflight oracle clone dropped gradients
- Error: `RuntimeError: stack expects a non-empty TensorList`
- Root cause: `copy.deepcopy(model)` does not retain parameter gradient buffers for the independent oracle.
- Source: semantic preflight attempt 1 stdout
- Do NOT retry: snapshot first-pass gradients explicitly before cloning an oracle model.

## Human Notes

> Autopilot local-only execution; no intervention requested.
