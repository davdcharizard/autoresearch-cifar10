# EXP-065: Lower CutMix Probability to 0.25

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-065.md
- **Plan**: plans/plan-065.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-065
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned one-scalar CutMix probability bracket in `train.py`: `CUTMIX_PROB` changed from `0.5` to `0.25`. All other EXP-064 CutMix anchor settings and the model, optimizer, schedule, augmentation, compile path, and validation cadence remain unchanged.

### Surprises & Discoveries

No implementation surprises. The current EXP-064 baseline already exposes `CUTMIX_PROB` as a top-level hyperparameter, so the bracket is an isolated constant change.

### Decisions

- Kept `CUTMIX_ALPHA=1.0` and `CUTMIX_LABEL_SMOOTHING=0.05` unchanged to isolate probability as the only intervention.
- Preserved the same on-device CutMix helper and loss path from EXP-064 so the experiment measures frequency, not implementation mechanics.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 4022; shell PID 3646811; uv PID 3646812; main python PID 3646815
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 19:47:21 UTC
- **Ended**: 2026-06-09 19:53:58 UTC

Description:
- Local foreground run of EXP-065 on one selected GPU with output captured to `run.log`. This tests whether reducing CutMix application probability from 0.5 to 0.25 preserves the regional-mixing benefit while lowering mixed-label pressure. Expected startup markers are `ResNet-20`, `CutMix alpha: 1.0, prob: 0.25, label smoothing: 0.05`, unchanged `Batches per epoch: 390`, and final metrics classified against the 94.21% improvement threshold.

Observations:
- Preflight passed: tracked diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, `uv run ruff check train.py` reported `All checks passed!`, and the baseline check reported `baseline=94.11`, `baseline_commit=1119ff8`.
- 2026-06-09 19:46 UTC: GPU1 selected after `nvidia-smi` showed GPU1 at `0MiB` and `0%` utilization. GPU0 had unrelated activity and was not touched.
- 2026-06-09 19:47 UTC: Foreground run launched on GPU1. Process table showed shell PID 3646811, uv PID 3646812, and main Python PID 3646815; `/proc/3646815/cwd` verified this project root.
- Startup confirmed CUDA, `ResNet-20 | params: 822,790`, `CutMix alpha: 1.0, prob: 0.25, label smoothing: 0.05`, 300s budget, and `Batches per epoch: 390`.
- 2026-06-09 19:48 UTC: Early training healthy through epoch 16 with best test accuracy 83.56%, 7-10ms batch timings, no traceback/OOM/runtime-error/non-finite patterns, and GPU1 active.
- 2026-06-09 19:49 UTC: Mid-run pre-drop progress healthy through epoch 29. Best reached 87.75%; LR remained 0.1000, and the 21k first LR drop remained reachable.
- 2026-06-09 19:50 UTC: First LR drop confirmed at `step 21000 ep 54` with `lr: 0.0100`. Pre-drop best was 88.23%; post-drop refinement reached 93.06% by epoch 55.
- 2026-06-09 19:51 UTC: Post-drop refinement improved to 93.76% by epoch 65, still below the 94.21% improvement threshold.
- 2026-06-09 19:52 UTC: Late peak reached 94.09% at epoch 75, below both the 94.11% baseline and the 94.21% improvement threshold.
- 2026-06-09 19:53 UTC: Run exited cleanly with final summary metrics. Final `best_test_acc` remained 94.09%, so this is a valid no-improvement result.

Key Metrics:
- `best_test_acc`: 94.09%
- `final_test_acc`: 93.76%
- `final_test_loss`: 0.2572
- `training_seconds`: 300.0
- `total_seconds`: 398.4
- `startup_seconds`: 2.3
- `peak_vram_mb`: 660.4
- `num_epochs`: 105
- `num_steps`: 40,685
- `num_params`: 822,790
- Verdict for execution: valid no-improvement because 94.09% is below the 94.11% baseline and the 94.21% improvement threshold.

## Verification Results

### Conditions Checked
- Baseline check: `exp-index.sh baseline` reported `baseline=94.11`, `baseline_commit=1119ff8`, `total_experiments=66`, `improvements=10`; pass.
- Scoped diff check: `git diff --name-only` listed only `train.py`; pass.
- Compile check: `python3 -m py_compile train.py` exited 0; pass.
- Style check: `uv run ruff check train.py` reported `All checks passed!`; pass.
- Execution summary check: `run.log` contains numeric final metrics including `best_test_acc: 94.09%`; pass.
- Model-depth check: `run.log` reports `ResNet-20 | params: 822,790`; pass.
- CutMix settings check: `run.log` reports `CutMix alpha: 1.0, prob: 0.25, label smoothing: 0.05`; pass.
- Batch-geometry check: `run.log` reports `Batches per epoch: 390`; pass.
- LR-drop check: `run.log` contains `step 21000 ep 54 ... lr: 0.0100`; pass.
- Error scan: `rg -n "Traceback|CUDA out of memory|RuntimeError|\bnan\b|\binf\b" run.log` returned no matches; pass.
- Classification check: valid run but `94.09% < 94.21%`; classified as no-improvement.

### Informational Metrics
- The run completed 40,685 steps and 105 epochs, more than EXP-064's 39,493 steps and 102 epochs.
- Parameter count stayed at 822,790 and peak VRAM stayed at 660.4 MB, so the probability bracket did not change model size or memory footprint.
- Final accuracy was 93.76%, above EXP-064's final 93.02%, but the primary metric still fell below the new baseline.

## Errors & Dead Ends

## Human Notes

> Autopilot mode; no human approval or intervention requested during execution.
