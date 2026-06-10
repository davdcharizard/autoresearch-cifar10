# EXP-035: Combine Lower Smoothing with 22k First Drop

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-035.md
- **Plan**: plans/plan-035.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-035
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

Implemented EXP-035 exactly as planned by changing only two scalar values in `train.py`: `LR_MILESTONES` from `[21000, 64000]` to `[22000, 64000]`, and the training loss smoothing from `label_smoothing=0.05` to `label_smoothing=0.03`. The reflection-padding 28/56/112 ResNet-20 anchor, batch size, optimizer settings, FP32 channels-last compile path, seed, and once-per-epoch validation path were preserved. Preflight checks passed for Python syntax, ruff, diff scope, validation cadence, reflected padding, lower label smoothing, and the 22k schedule.

### Surprises & Discoveries

No implementation surprises. The change is the direct composition of the two recent 93.79 near-miss probes.

### Decisions

No deviations from the plan were needed. The second milestone remains at 64000 to avoid retrying the failed second-drop family.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local command session 64900
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 03:33 UTC
- **Ended**: 2026-06-09 03:40 UTC

Description:
- Run the reflection-padding 28/56/112 ResNet-20 anchor with both recent near-miss scalar edits: `label_smoothing=0.03` and first LR drop at step 22000. The experiment tests whether sharper class separation plus slightly more high-LR fitting can combine to clear the 93.80% threshold. The run must complete under the fixed 300s training budget and under 10 minutes total wall-clock time. It must reach `best_test_acc >= 93.80%` to count as an improvement over the current `93.70%` baseline.

Observations:
- Startup confirms CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. GPU 1 is active for the run. (source: run.log L1-L4)
- Early epoch evaluations are present, with best test accuracy reaching 83.97% by epoch 10 and no error/OOM/NaN/Inf signatures in the initial log. (source: `grep "eval ep" run.log | head -10`)
- The planned schedule behavior is confirmed: step 21000 remained at `lr: 0.1000`, and step 22000 dropped to `lr: 0.0100`. Post-drop accuracy reached 93.35% by epoch 70, still below the 93.80% threshold. (source: run.log schedule grep and eval tail)
- The run exited cleanly with `best_test_acc: 93.63%`, `final_test_acc: 93.36%`, `final_test_loss: 0.2340`, and `total_seconds: 403.4`. The combined near-miss probe underperformed the 93.70 baseline and is a clear no-improvement.

Key Metrics:
- `best_test_acc`: 93.63%
- `final_test_acc`: 93.36%
- `final_test_loss`: 0.2340
- `training_seconds`: 300.0
- `total_seconds`: 403.4
- `startup_seconds`: 2.4
- `peak_vram_mb`: 660.4
- `num_epochs`: 107
- `num_steps`: 41444
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Passed: Process exited with code 0 and completed before the 10-minute wall-clock limit (`total_seconds: 403.4`).
- Passed: The run reported a numeric `best_test_acc` of 93.63%.
- Failed: The current baseline is 93.70%, so the goal requires at least 93.80% for improvement; EXP-035 reached 93.63%, below baseline.
- Passed: Fixed training budget was preserved (`training_seconds: 300.0`).
- Passed: Architecture stayed unchanged (`num_params: 822,790`).
- Passed: Batch size stayed unchanged (`Batches per epoch: 390`).
- Passed: Step 21000 remained at `lr: 0.1000`.
- Passed: First LR drop occurred at step 22000 with `lr: 0.0100`.
- Passed: No second LR drop occurred; `step 64000` was absent and the run ended at 41,444 steps.
- Passed: The tracked source diff during the run was limited to the planned `train.py` LR milestone and label-smoothing scalar changes.
- Passed: No error, exception, CUDA OOM, NaN, or Inf signatures were found in `run.log`.

### Informational Metrics
- The interaction between lower smoothing and a later first drop was worse than either single-axis near-miss.
- Final accuracy was 0.27 points below the best accuracy, with late training never approaching the 93.80% threshold.

## Errors & Dead Ends

## Human Notes

> No human notes yet.
