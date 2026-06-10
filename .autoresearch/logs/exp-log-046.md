# EXP-046: Time-Budget-Matched Cosine Schedule

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-046.md
- **Plan**: plans/plan-046.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-046
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Implemented the planned schedule-only experiment in `train.py`. The code now uses an elapsed-time cosine helper to set SGD learning rate once per training step based on `total_training_time / TIME_BUDGET_S`, replacing the prior step-milestone `MultiStepLR` path. Architecture, augmentation, batch size, optimizer family, initial LR, momentum, weight decay, label smoothing, compile, channels-last, and `MAX_STEPS` were intentionally left unchanged.

### Surprises & Discoveries

No implementation surprises. The existing training loop already tracks `total_training_time` before each step, which made time-fraction scheduling a localized change.

### Decisions

- Kept `LR=0.1` as the cosine peak instead of adding warmup or raising peak LR. This preserves the current anchor and avoids retrying the known failed scalar LR retunes from EXP-040 and EXP-043.
- Removed the active `LR_MILESTONES` constant instead of leaving it unused, so future readers do not mistake the experiment for a milestone schedule.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground process, Python PID 3005868
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 13:10 UTC
- **Ended**: 2026-06-09 13:17 UTC

Description:
- Run the EXP-046 time-budget-matched cosine schedule on a single local GPU with output captured to `run.log`. The test keeps the EXP-038 anchor recipe otherwise unchanged and evaluates whether smooth elapsed-time decay can clear the 94.07% improvement threshold. Expected behavior is LR gradually decaying from 0.1000 toward zero across the 300-second training budget, with one validation pass per epoch.

Observations:
- Launch GPU check showed both H20 GPUs busy from an external Protenix workload; GPU0 had lower memory usage (`19680 / 97871 MiB`) than GPU1 (`21736 / 97871 MiB`), so Run 1 used GPU0. This may reduce realized step count but remains a valid single-GPU, fixed-time run.
- Startup succeeded on CUDA with `822,790` parameters. LR decay is active: the log shows `lr: 0.0994` at step 50 and `lr: 0.0983` by step 550, and epoch 1 evaluation reached 54.05%. (source: `run.log` startup tail)
- Late cosine refinement plateaued below the active threshold: best improved from 92.05% at epoch 38 to 93.01% at epoch 49, then held at 93.01% through epoch 50 and finished at 92.82%. (source: `run.log` L80-L117)

Key Metrics:
- best_test_acc: 93.01% (source: `run.log` L108)
- final_test_acc: 92.82% (source: `run.log` L109)
- final_test_loss: 0.2514 (source: `run.log` L110)
- training_seconds: 300.0 (source: `run.log` L111)
- total_seconds: 366.9 (source: `run.log` L112)
- startup_seconds: 3.3 (source: `run.log` L113)
- peak_vram_mb: 660.4 (source: `run.log` L114)
- num_epochs: 51 (source: `run.log` L115)
- num_steps: 19,691 (source: `run.log` L116)
- num_params: 822,790 (source: `run.log` L117)

## Verification Results

### Conditions Checked
- Baseline query: passed. Current baseline is 93.97% at commit `755be2c`; improvement threshold is 94.07%.
- Scope check: passed. `git diff --name-only` reported only `train.py`.
- Compile check: passed. `python3 -m py_compile train.py` exited 0.
- Ruff check: passed. `uv run ruff check train.py` reported `All checks passed!`.
- Run completion: passed. `uv run train.py` exited 0 and produced numeric final metrics.
- Improvement condition: failed. `best_test_acc=93.01%`, which is below the required `94.07%` threshold. (source: `run.log` L108)
- Remaining success conditions: skipped — aborted verification after the improvement condition failed.

### Informational Metrics
- best_test_acc: 93.01%
- final_test_acc: 92.82%
- final_test_loss: 0.2514
- training_seconds: 300.0
- total_seconds: 366.9
- startup_seconds: 3.3
- peak_vram_mb: 660.4
- num_epochs: 51
- num_steps: 19,691
- num_params: 822,790

## Errors & Dead Ends

## Human Notes

> No human interventions during autopilot execution.
