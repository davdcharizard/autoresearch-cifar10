# EXP-032: Mild Isolated Label Smoothing

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-032.md
- **Plan**: plans/plan-032.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-032
- **Commit**: e736c35
- **PR**: skipped — no git remote configured
- **Outcome**: completed - improvement

## Implementation Notes

### Summary

Implemented EXP-032 exactly as planned by changing only the training loss call in `train.py` from `F.cross_entropy(outputs, targets)` to `F.cross_entropy(outputs, targets, label_smoothing=0.05)`. The reflection-padding 28/56/112 ResNet-20 anchor, batch size, optimizer settings, LR schedule, FP32 channels-last compile path, seed, and once-per-epoch validation path were preserved. Preflight checks passed for Python syntax, ruff, diff scope, installed PyTorch API support, validation cadence, reflected padding, label-smoothing presence, and schedule preservation.

### Surprises & Discoveries

No implementation surprises. The installed `torch.nn.functional.cross_entropy` signature includes `label_smoothing`, so no compatibility shim or dependency change was needed.

### Decisions

No deviations from the plan were needed. Kept the smoothing value at `0.05` to isolate a mild confidence regularizer rather than retesting the stronger combined regularization bundle from EXP-000.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local command session 5292
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 20:38 UTC
- **Ended**: 2026-06-08 20:47 UTC

Description:
- Run the EXP-029 reflection-padding 28/56/112 ResNet-20 anchor with only the cross-entropy loss changed to use `label_smoothing=0.05`. The experiment tests whether mild confidence regularization improves late post-drop stability without reducing step throughput. The run must complete under the fixed 300s training budget and under 10 minutes total wall-clock time. It must reach `best_test_acc >= 93.68%` to count as an improvement over the current `93.58%` baseline.

Observations:
- Startup confirms CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. Early epoch evaluations are present, and GPU 0 is active. (source: run.log L1-L8)
- Early training is healthy with no crash/OOM/NaN/Inf patterns; best test accuracy reached 87.08% by epoch 17 while still pre-LR-drop. (source: run.log L24-L38)
- The planned first LR drop fired at step 21000 with `lr: 0.0100`; immediate post-drop evaluation reached 92.00% at epoch 54. (source: run.log L111-L112)
- Post-drop accuracy climbed to 93.48% by epoch 60 and then flattened below the 93.68% improvement threshold through epoch 69. (source: run.log L124-L142)
- Late post-drop refinement crossed the improvement bar at epoch 87, reaching `best: 93.70%`, above the required 93.68% threshold. (source: run.log L184-L188)
- The run exited cleanly with `best_test_acc: 93.70%`, `final_test_acc: 93.65%`, `final_test_loss: 0.2446`, and `total_seconds: 398.1`. This is +0.12 percentage points over the 93.58% baseline and clears the explicit 93.68% improvement threshold.

Key Metrics:
- `best_test_acc`: 93.70%
- `final_test_acc`: 93.65%
- `final_test_loss`: 0.2446
- `training_seconds`: 300.0
- `total_seconds`: 398.1
- `startup_seconds`: 1.8
- `peak_vram_mb`: 660.4
- `num_epochs`: 108
- `num_steps`: 41823
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Passed: Baseline was 93.58% from EXP-029, so the current goal requires at least 93.68% for improvement; EXP-032 reached 93.70%.
- Passed: Process exited with code 0 and completed before the 10-minute wall-clock limit (`total_seconds: 398.1`).
- Passed: Fixed training budget was preserved (`training_seconds: 300.0`).
- Passed: Architecture stayed unchanged (`num_params: 822,790`).
- Passed: Batch size stayed unchanged (`Batches per epoch: 390`).
- Passed: First LR drop occurred at step 21000 with `lr: 0.0100`.
- Passed: No second LR drop occurred; `step 64000` was absent and the run ended at 41,823 steps.
- Passed: The tracked source diff during the run was limited to the planned `train.py` loss-call change.
- Passed: No error, exception, CUDA OOM, NaN, or Inf signatures were found in `run.log`.

### Informational Metrics
- Late best accuracy occurred at epoch 87 and remained the run best through completion.
- Final accuracy was close to the best accuracy (93.65% vs. 93.70%), suggesting label smoothing improved late stability rather than producing a single isolated spike.

## Errors & Dead Ends

## Human Notes

> No human notes yet.
