# EXP-017: ResNet-20 Width 30/60/120 with 20k First Drop

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-017.md
- **Plan**: plans/plan-017.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-017
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the planned modest width step and matching earlier first LR drop in `train.py`. `STAGE_WIDTHS` changed from `(28, 56, 112)` to `(30, 60, 120)`, and `LR_MILESTONES` changed from `[21000, 64000]` to `[20000, 64000]`.

### Surprises & Discoveries
No implementation surprises. Both target values are centralized hyperparameter constants near the top of `train.py`, and syntax/lint checks passed after the edit.

### Decisions
Kept the second LR milestone at 64000 so the run remains in LR 0.01 after the first drop. This isolates whether the modest width increase plus earlier first drop can improve the current 93.23% baseline without adding a reachable LR 0.001 phase.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local shell PID 835893
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 16:48 UTC
- **Ended**: 2026-06-08 16:55 UTC

Description:
- Run a ResNet-20 with `STAGE_WIDTHS = (30, 60, 120)` and first LR milestone at step 20000. This tests whether a cautious width increase beyond the EXP-016 baseline can clear the new 93.33% threshold when the schedule is moved earlier for the expected slower step budget. The run keeps the fixed 300s training budget, single-GPU execution, and once-per-epoch validation.

Observations:
- Physical GPU 0 was idle before launch, stale `run.log` was removed, and the single-GPU CUDA check reported one visible NVIDIA H20. EXP-017 launched on physical GPU 0 via `CUDA_VISIBLE_DEVICES=0`. (source: `nvidia-smi`, CUDA check)
- Startup log is being written and reports `Device: cuda`, `ResNet-20 | params: 944,200`, `Time budget: 300s`, and `Batches per epoch: 390`. (source: `run.log` startup lines)
- Early monitoring is clean through epoch 11: no traceback, CUDA OOM, NaN/Inf, or compile-failure patterns are present. Best early test accuracy is 82.96%, and the run is still before the planned step-20000 LR drop. (source: `run.log` early epoch output and error-pattern scan)
- The planned first LR drop occurred at step 20000 during epoch 52 with about 96s remaining. Post-drop accuracy reached 92.47% at epoch 52 and 93.01% by epoch 55, below the 93.33% threshold while the run continues. (source: `run.log` LR-drop and eval lines)
- The run completed normally with final summary metrics present. Best accuracy peaked at 93.16%, which is below the 93.23% baseline and below the 93.33% improvement threshold. (source: `run.log` final summary)

Key Metrics:
- best_test_acc: 93.16%
- final_test_acc: 92.97%
- final_test_loss: 0.2768
- training_seconds: 300.0
- total_seconds: 378.6
- startup_seconds: 3.1
- peak_vram_mb: 713.0
- num_epochs: 71
- num_steps: 27400
- num_params: 944,200

## Verification Results

### Conditions Checked
- Baseline and threshold: passed. `exp-index.sh baseline` reported baseline=93.23, so EXP-017 required `best_test_acc >= 93.33`.
- Single-GPU CUDA execution: passed. `CUDA_VISIBLE_DEVICES=0 uv run python ...` reported CUDA available, visible device count 1, and device name NVIDIA H20 before launch.
- Completion metric present: passed. `run.log` includes numeric `best_test_acc: 93.16%` and `peak_vram_mb: 713.0`.
- Primary metric condition: failed. 93.16 < 93.33, so the run is `no-improvement` under the tightened +0.10 point rule.
- Schedule/throughput sanity: passed. The first LR drop occurred at step 20000, the run reached 27,400 steps over 71 epochs, and the model had 944,200 parameters.
- Error-pattern scan: passed. `rg -n "Traceback|CUDA out of memory|RuntimeError|nan|inf|compile failure" run.log` returned no matches.
- Scope review: passed. `git diff -- train.py` only changes `STAGE_WIDTHS` from `(28, 56, 112)` to `(30, 60, 120)` and `LR_MILESTONES` from `[21000, 64000]` to `[20000, 64000]`.
- Validation cadence review: passed. `train.py` has one `evaluator = Eval()` construction and one `evaluator.evaluate(model, device)` call inside the epoch path; no extra validation loop was added.

### Informational Metrics
- final_test_acc: 92.97%
- final_test_loss: 0.2768
- training_seconds: 300.0
- total_seconds: 378.6
- startup_seconds: 3.1
- peak_vram_mb: 713.0
- num_epochs: 71
- num_steps: 27400
- num_params: 944,200

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
