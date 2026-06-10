# EXP-040: Raise Initial LR to 0.12 on 2e-4 Anchor

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-040.md
- **Plan**: plans/plan-040.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-040
- **Commit**: (pending - committed on loop success)
- **PR**: (pending - created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

Implemented EXP-040 exactly as planned by changing only `LR` in `train.py` from `0.1` to `0.12`. The reflection-padding 28/56/112 ResNet-20 anchor, batch size 128, `LR_MILESTONES = [21000, 64000]`, optimizer class, momentum, `WEIGHT_DECAY = 2e-4`, `label_smoothing=0.05`, FP32 channels-last compile path, seed, fixed training budget, and once-per-epoch validation path were preserved. Preflight checks passed for Python syntax, ruff, diff scope, higher initial LR, anchor settings, and validation cadence.

### Surprises & Discoveries

No implementation surprises. LR is a single top-level constant passed directly into SGD.

### Decisions

No deviations from the plan were needed. The schedule milestones are intentionally unchanged so the first drop should produce `lr: 0.0120` at step 21000.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local command session 18331; shell PID 2528353
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 08:20 UTC
- **Ended**: 2026-06-09 08:27 UTC

Description:
- Run the reflection-padding 28/56/112 ResNet-20 anchor with only `LR` changed from `0.1` to `0.12`. The experiment tests whether the validated `WEIGHT_DECAY = 2e-4` anchor can benefit from slightly more aggressive high-LR exploration before the preserved 21k first drop. It must complete under the fixed 300s training budget and under 10 minutes total wall-clock time. It must reach `best_test_acc >= 94.07%` to count as an improvement under the goal's +0.10 percentage-point rule.

Observations:
- Preflight checks passed: `python3 -m py_compile train.py`, `uv run ruff check train.py`, diff scope, `LR = 0.12`, preserved anchor grep, and validation-cadence grep all succeeded before launch. (source: command outputs, 2026-06-09)
- Baseline check reported `baseline=93.97`, making the concrete improvement threshold `best_test_acc >= 94.07`; GPU 0 was occupied by an unrelated run, GPU 1 was idle, and CUDA isolation reported one visible `NVIDIA H20`. (source: exp-index, nvidia-smi, CUDA smoke test, 2026-06-09)
- Startup confirms CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, `Batches per epoch: 390`, and early step output with `lr: 0.1200`. (source: run.log L1-L5)
- The planned first LR drop fired at step 21000 with `lr: 0.0120`; early post-drop accuracy reached 93.70% by epoch 60, still below the 94.07% improvement threshold. (source: `grep "step 21000" run.log`; `grep "eval ep" run.log | tail -20`)
- The late plateau did not improve beyond 93.70%; the run exited cleanly with `best_test_acc: 93.70%`, `final_test_acc: 93.13%`, `final_test_loss: 0.2574`, and `total_seconds: 407.9`. This is below the 93.97% baseline and below the 94.07% improvement threshold. (source: final summary in run.log)

Key Metrics:
- `best_test_acc`: 93.70%
- `final_test_acc`: 93.13%
- `final_test_loss`: 0.2574
- `training_seconds`: 300.0
- `total_seconds`: 407.9
- `startup_seconds`: 4.6
- `peak_vram_mb`: 660.4
- `num_epochs`: 104
- `num_steps`: 40378
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Passed: Process exited with code 0 and completed before the 10-minute wall-clock limit (`total_seconds: 407.9`).
- Passed: The run reported a numeric `best_test_acc` of 93.70%.
- Failed: The current baseline is 93.97%, so the goal requires at least 94.07% for improvement; EXP-040 reached only 93.70%.
- Passed: Fixed training budget was preserved (`training_seconds: 300.0`).
- Passed: Architecture stayed unchanged (`num_params: 822,790`).
- Passed: Batch size stayed unchanged (`Batches per epoch: 390`).
- Passed: First LR drop occurred at step 21000 with `lr: 0.0120`.
- Passed: No second LR drop occurred; `step 64000` was absent and the run ended at 40,378 steps.
- Passed: The tracked source diff during the run was limited to the planned `train.py` LR scalar change.
- Passed: No error, exception, CUDA OOM, NaN, or Inf signatures were found in `run.log`.

### Informational Metrics
- Higher initial LR reached only 93.70%, matching the old EXP-032 baseline but below the current `2e-4` anchor.
- Final accuracy was 0.57 points below the peak, indicating the best value came from early post-drop refinement rather than final training.

## Errors & Dead Ends

## Human Notes

> No human notes yet.
