# EXP-064: Probabilistic CutMix Regional Mixing

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-064.md
- **Plan**: plans/plan-064.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-064
- **Commit**: 1119ff8
- **PR**: not created — no git remote configured
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned probabilistic CutMix variant in `train.py`. The patch adds `CUTMIX_ALPHA = 1.0`, `CUTMIX_PROB = 0.5`, and `CUTMIX_LABEL_SMOOTHING = 0.05`, defines a `rand_bbox` helper for clipped rectangular patches, samples CutMix on-device after batch transfer, pastes permuted patches into cloned inputs, recomputes lambda from the actual patch area, and uses weighted endpoint cross entropy for CutMix batches. Non-CutMix batches keep the original label-smoothed cross entropy path, and all architecture, optimizer, schedule, augmentation, compile/channels-last, validation, and timing anchors remain unchanged.

### Surprises & Discoveries

No code-structure surprises. The existing training loop already centralizes device transfer and loss computation, so CutMix can be inserted without touching the dataset transform, model definition, optimizer, scheduler, or evaluation harness.

### Decisions

- Kept `label_smoothing=0.05` for CutMix endpoint losses to preserve the validated target-regularization anchor rather than turning this into another label-smoothing deviation.
- Used `CUTMIX_PROB = 0.5` rather than applying CutMix every batch, because direct mixup and Cutout have negative local evidence and the experiment should bound regularization strength.
- Cloned inputs only on CutMix batches, leaving non-CutMix batches on the original tensor path to reduce avoidable overhead.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 54486; shell PID 3623802; uv PID 3623803; main python PID 3623806
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 19:31:32 UTC
- **Ended**: 2026-06-09 19:38:33 UTC

Description:
- Local foreground run of EXP-064 on one selected GPU with output captured to `run.log`. This tests whether probabilistic CutMix regional image/label mixing can improve the current ResNet-20 anchor while preserving the 21k first LR drop and the fixed evaluation harness. Expected startup markers are `ResNet-20`, `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, unchanged `Batches per epoch: 390`, and final metrics classified against the 94.07% improvement threshold.

Observations:
- Preflight passed: tracked diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- Baseline for classification: `93.97%`; improvement threshold: `94.07%`.
- 2026-06-09 19:31 UTC: GPU0 selected after `nvidia-smi` showed GPU0 at `0MiB` and `0%` utilization. GPU1 had unrelated activity and was not touched.
- 2026-06-09 19:31 UTC: Foreground run launched on GPU0. Process table showed shell PID 3623802, uv PID 3623803, and main Python PID 3623806; `/proc/3623806/cwd` verified this project root.
- Startup confirmed CUDA, `ResNet-20 | params: 822,790`, `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, 300s budget, and `Batches per epoch: 390`.
- 2026-06-09 19:32 UTC: Early training healthy through epoch 7 with best test accuracy 76.94%, 6-9ms batch timings, no traceback/OOM/runtime-error/non-finite patterns, and GPU0 active.
- 2026-06-09 19:33 UTC: Mid-run pre-drop progress healthy through epoch 21. Best reached 85.97%; LR remained 0.1000, timings mostly 6-8ms with occasional 10-11ms batches, and the 21k first LR drop remained comfortably reachable.
- 2026-06-09 19:35 UTC: First LR drop confirmed at `step 21000 ep 54` with `lr: 0.0100`. Pre-drop best was 87.97%; post-drop refinement reached 93.54% by epoch 58 with about 125s training budget remaining.
- 2026-06-09 19:36 UTC: Post-drop refinement improved to 93.89% by epoch 71, still below the 94.07% improvement threshold. No error patterns were present, so the run continued for final summary metrics.
- 2026-06-09 19:38 UTC: Run crossed the improvement threshold at epoch 96 with `test_acc=94.11%`, then exited cleanly with final summary metrics. Final `best_test_acc` remained 94.11%, above the 93.97% baseline and 94.07% threshold.

Key Metrics:
- `best_test_acc`: 94.11%
- `final_test_acc`: 93.02%
- `final_test_loss`: 0.3064
- `training_seconds`: 300.0
- `total_seconds`: 395.4
- `startup_seconds`: 2.4
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,493
- `num_params`: 822,790
- Verdict for execution: valid improvement because 94.11% is above both the 93.97% baseline and the 94.07% improvement threshold.

## Verification Results

### Conditions Checked
- Baseline check: `exp-index.sh baseline` reported `baseline=93.97`, `baseline_commit=755be2c`, `total_experiments=65`, `improvements=9`; pass.
- Scoped diff check: `git diff --name-only` listed only `train.py`; pass.
- Compile check: `python3 -m py_compile train.py` exited 0; pass.
- Style check: `uv run ruff check train.py` reported `All checks passed!`; pass.
- Execution summary check: `run.log` contains numeric final metrics including `best_test_acc: 94.11%`; pass.
- Model-depth check: `run.log` reports `ResNet-20 | params: 822,790`; pass.
- CutMix settings check: `run.log` reports `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`; pass.
- Batch-geometry check: `run.log` reports `Batches per epoch: 390`; pass.
- LR-drop check: `run.log` contains `step 21000 ep 54 ... lr: 0.0100`; pass.
- Error scan: `rg -n "Traceback|CUDA out of memory|RuntimeError|\bnan\b|\binf\b" run.log` returned no matches; pass.
- Classification check: valid run and `94.11% >= 94.07%`; classified as improvement.

### Informational Metrics
- The run completed 39,493 steps and 102 epochs, comparable to direct mixup EXP-060's 41,074 steps and above EXP-055's 37,547 steps.
- Parameter count stayed at the 822,790-param anchor, so the gain came from training augmentation rather than model capacity.
- Final accuracy was 93.02%, below the best epoch, so the benefit appears as a peak checkpoint improvement rather than a monotonic final-checkpoint lift.

## Errors & Dead Ends

## Human Notes

> Autopilot mode; no human approval or intervention requested during execution.
