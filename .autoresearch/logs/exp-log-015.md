# EXP-015: 28/56/112 First LR Drop at 23k

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-015.md
- **Plan**: plans/plan-015.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-015
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the planned isolated schedule retune on top of the EXP-014 baseline. `train.py` now keeps `STAGE_WIDTHS = (28, 56, 112)` and changes only `LR_MILESTONES` from `[22000, 64000]` to `[23000, 64000]`.

### Surprises & Discoveries
No implementation surprises. The scheduler remains centralized in the `LR_MILESTONES` constant, so the experiment is a one-line configuration change.

### Decisions
Kept the second LR milestone at 64000 so the run remains in LR 0.01 after the first drop. This isolates whether the current 28/56/112 model benefits from 1k more high-LR exploration without adding capacity or changing any throughput settings.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local shell PID 499869
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 16:19 UTC
- **Ended**: 2026-06-08 16:26 UTC

Description:
- Run the current 28/56/112 ResNet-20 with first LR milestone moved from step 22000 to 23000. This tests whether a little more high-LR exploration can improve the 93.09% baseline to at least 93.19% while retaining enough LR 0.01 refinement.

Observations:
- Physical GPU 0 was idle before launch, stale `run.log` was removed, and the single-GPU CUDA check reported one visible NVIDIA H20. EXP-015 will launch on physical GPU 0 via `CUDA_VISIBLE_DEVICES=0`. (source: `nvidia-smi`, CUDA check)
- Startup log is being written and reports `Device: cuda`, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. Physical GPU 0 shows 413 MiB allocated during startup. (source: `run.log` startup lines, `nvidia-smi`)
- Early monitoring is clean through epoch 22: no traceback, CUDA OOM, NaN/Inf, or compile failure patterns are present. Best early test accuracy is 87.14%, and the run is still before the planned step-23000 LR drop. (source: `run.log` early epoch output and error-pattern scan)
- The planned first LR drop occurred at step 23000 during epoch 59 with about 132s remaining. Post-drop accuracy reached 92.17% at epoch 60 and 92.79% by epoch 66, below the 93.19% threshold so far while the run continues. (source: `run.log` LR-drop and eval lines)
- The run completed normally with final summary metrics present. Best accuracy peaked at 92.88%, which is below the 93.09% baseline and below the 93.19% improvement threshold. (source: `run.log` lines 204-213)

Key Metrics:
- best_test_acc: 92.88%
- final_test_acc: 92.75%
- final_test_loss: 0.3130
- training_seconds: 300.0
- total_seconds: 386.4
- startup_seconds: 2.6
- peak_vram_mb: 660.4
- num_epochs: 99
- num_steps: 38274
- num_params: 822,790

## Verification Results

### Conditions Checked
- Baseline and threshold: passed. `exp-index.sh baseline` reported baseline=93.09, so EXP-015 required best_test_acc >= 93.19.
- Single-GPU CUDA execution: passed. `CUDA_VISIBLE_DEVICES=0 uv run python ...` reported CUDA available, visible device count 1, and device name NVIDIA H20 before launch.
- Completion metric present: passed. `run.log` includes numeric `best_test_acc: 92.88%` and `peak_vram_mb: 660.4`.
- Primary metric condition: failed. 92.88 < 93.19, so the run is `no-improvement` under the tightened +0.10 point rule.
- Schedule/throughput sanity: passed. The first LR drop occurred at step 23000, the run reached 38,274 steps over 99 epochs, and the model had 822,790 parameters.
- Error-pattern scan: passed. `grep -n "Traceback\|CUDA out of memory\|RuntimeError\|Error\|nan\|inf" run.log` returned no matches.
- Scope review: passed. `git diff -- train.py` only changes `LR_MILESTONES` from `[22000, 64000]` to `[23000, 64000]`.
- Validation cadence review: passed. `train.py` has one `evaluator = Eval()` construction and one `evaluator.evaluate(model, device)` call inside the epoch path; no extra validation loop was added.

### Informational Metrics
- final_test_acc: 92.75%
- final_test_loss: 0.3130
- training_seconds: 300.0
- total_seconds: 386.4
- startup_seconds: 2.6
- peak_vram_mb: 660.4
- num_epochs: 99
- num_steps: 38274
- num_params: 822,790

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
