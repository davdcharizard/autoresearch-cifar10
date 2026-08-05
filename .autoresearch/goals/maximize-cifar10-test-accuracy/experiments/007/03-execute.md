# EXP-007: Disable Weight Decay for the Hard-Label Tail

## Execution

Overall Status & Info:
- **Created**: 2026-07-24
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-007
- **Commit**: pending; committed only on loop success
- **PR**: N/A; this workflow is explicitly local-only
- **Outcome**: failed

## Implementation Notes

### Summary

Added a production helper that validates the accepted optimizer's two live parameter groups, changes only matrix-parameter weight decay from `5e-4` to zero, and returns the actual group values for logging. The existing 65% mixup transition invokes the helper before the first hard-label update. A deterministic norm helper records matrix-weight L2 norms at the switch and before final runtime/VRAM snapshots; it does not mutate parameters or consume RNG.

### Surprises & Discoveries

PyTorch optimizers may copy input parameter-group dictionaries, so mutating a dictionary originally passed to SGD is not a reliable production switch. The implementation instead validates and mutates `optimizer.param_groups` directly through the shared helper. Semantic preflight proved the live transition `(0.0005, 0.0, 0.0)` and exact norm/RNG behavior.

### Decisions

The training branch and preflight call the same helper, avoiding drift between tested and scored paths. Norm telemetry is descriptive only because EXP-002 has no comparable norm control; accuracy, loss, exposure, and run integrity determine the result. The one transition norm is charged inside counted time, and the final norm is computed before `t_end` and peak-memory capture.

## Experimental Adjustments

- **Shared production switch helper**: Added after plan review showed a hand-reproduced preflight could miss a production-branch typo. The helper returns values read from the live optimizer groups. (ref: `02-plan-review.md` concern 1)
- **Descriptive-only norms with complete accounting**: Norms will not support causal claims, and final diagnostic cost is included in runtime/VRAM snapshots. (ref: `02-plan-review.md` concerns 2-3)

## Run Log

### Run 1

Metadata:
- **Job ID**: PID 1148927 (execution session 67976)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-24 14:23:02 UTC
- **Ended**: 2026-07-24 14:28:42 UTC

Description:
- One fixed-seed local run of the accepted WRN-16-2 and alpha-0.2 mixup recipe. Matrix-parameter weight decay remains `5e-4` through 65% counted progress, then changes to zero alongside mixup for the hard-label tail. The expected result is a valid 300-second run with at least 26,329 steps and `best_test_acc >= 94.17%`.

Observations:
- Preflight passed on one NVIDIA H20: live groups changed `[0.0005, 0.0] -> [0.0, 0.0]`, parameter count was 691,674, norm 13.0 was exact, and tensors/RNG were unchanged.
- The redirected run initialized normally on CUDA and reached step 900 at about 3.5% progress with finite loss and roughly 23.6k images/s. (source: `run.log` initial bounded extract)
- Mixup and matrix weight decay disabled exactly once at epoch 92, step 17,876, 195.0 seconds (65.0%); live groups reported `0.0005 -> 0` and `0`, with descriptive norm 25.8668. (source: `run.log` L42)
- The run completed normally at epoch 143 after 27,835 steps. Accuracy peaked at 93.74% at epoch 140 and finished at 93.70%. (source: `run.log` L60-L76)

Key Metrics:
- `best_test_acc`: 93.74% at epoch 140; baseline 94.07%, delta -0.33 points. (source: `run.log` L62-L66)
- `final_test_acc`: 93.70%; `final_test_loss`: 0.3244. (source: `run.log` L64-L68)
- `training_seconds`: 300.0; `total_seconds`: 340.7; `startup_seconds`: 1.1. (source: `run.log` L69-L71)
- `num_steps`: 27,835; realized exposure: 142.52 passes; `num_epochs`: 143. (source: `run.log` L73-L74)
- `peak_vram_mb`: 1,094.0; `num_params`: 691,674. (source: `run.log` L72-L75)
- Descriptive decayed-weight L2: 25.8668 at switch and 37.5522 final; no causal baseline norm exists. (source: `run.log` L42, L76)

## Verification Results

### Conditions Checked

- **Run completion and 10-minute limit — PASS**: complete finite summary, 300.0 counted seconds, and 340.7 total seconds on one NVIDIA H20. The run used 29 unique evaluations, every fifth epoch plus final epoch 143, without duplicates. (source: `run.log` L6-L76; verification output)
- **Improve baseline by at least 0.10 points — FAIL**: baseline 94.07% requires at least 94.17%; actual best was 93.74%, 0.33 points below baseline and 0.43 below threshold. Verification stopped after this necessary-condition failure. (source: results index baseline output; `run.log` L66)
- **Scope/integrity checks — recorded before metric verdict**: one correct 65.0% live-group transition, 27,835 steps exceeds the 26,329 floor, unchanged 691,674 parameters, one `train.py` diff, passing lint/compile/diff checks, and no evaluator/seed/dependency change. These support attribution but do not rescue the metric failure.

### Informational Metrics

- Skipped as formal verification metrics after the primary condition failed; all values were preserved inline in `Run 1 > Key Metrics`. The norm increase from 25.8668 to 37.5522 is descriptive only because no accepted-run norm control exists.

## Errors & Dead Ends

None.

## Human Notes

No user intervention; autopilot remained local and offline.
