# EXP-002: FP32 Throughput Without AMP

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md
- **Plan**: plans/plan-002.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-002
- **Commit**: 6743174
- **PR**: N/A — repo has no configured remote; TASK.md says remote unavailability is intentional for this benchmark.
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the FP32 throughput isolation experiment in `train.py`. Added cuDNN benchmark, channels-last model/input layout, and `torch.compile`, while leaving forward/loss arithmetic in ordinary FP32 code. The statistical training recipe remains unchanged: same architecture, augmentation, loss, optimizer settings, LR milestones, and evaluation cadence.

### Surprises & Discoveries

No static issues appeared. The diff is smaller than EXP-001 because all AMP/autocast code was intentionally omitted.

### Decisions

Kept `torch.compile` enabled because EXP-001 showed this code path runs on the H20. The experiment isolates BF16 removal rather than changing both compile and precision at once.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 86127
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 09:15 UTC
- **Ended**: 2026-06-08 09:21 UTC

Description:
- Run the baseline recipe with FP32-preserving throughput mechanisms only: cuDNN benchmark, channels-last layout, and `torch.compile`. This tests whether removing BF16 autocast from EXP-001 recovers the small 0.04 percentage-point gap while retaining some step-count improvement. Expected result is a clean final summary with a numeric `best_test_acc` and `num_steps` diagnostic.

Observations:
- Run 1 started cleanly with a non-fatal Inductor TF32 warning, then trained through epoch 18 without compile/OOM/runtime errors; step times were mostly 6-7 ms and early best accuracy was 83.80%. (source: run.log L5-L42)
- The run completed cleanly and improved `best_test_acc` to 91.95%, exceeding the 91.52 baseline by 0.43 points. It also reached 43,398 steps, higher than EXP-001's 39,558. (source: run.log L231-L241)

Key Metrics:
- best_test_acc: 91.95% (source: run.log L232)
- final_test_acc: 91.42% (source: run.log L233)
- final_test_loss: 0.3365 (source: run.log L234)
- training_seconds: 300.0 (source: run.log L235)
- total_seconds: 396.2 (source: run.log L236)
- startup_seconds: 2.7 (source: run.log L237)
- peak_vram_mb: 379.0 (source: run.log L238)
- num_epochs: 112 (source: run.log L239)
- num_steps: 43,398 (source: run.log L240)
- num_params: 269,722 (source: run.log L241)

## Verification Results

### Conditions Checked
- `uv run train.py` completes without crashing: passed. The run exited normally and emitted a final summary. (source: run.log L231-L241)
- The run reports a numeric `best_test_acc`: passed. `best_test_acc` was 91.95%. (source: run.log L232)
- `best_test_acc` improves over the current experiment-index baseline in the higher-is-better direction: passed. 91.95% is above the 91.52% baseline. (source: run.log L232; baseline: experiment-indices/maximize-cifar10-best-test-accuracy.tsv)
- The implementation respects all hard constraints: passed. The diff changes only `train.py`, leaves AMP/autocast absent, preserves augmentation/loss/optimizer/scheduler/model/evaluation cadence, and `evaluator.evaluate(model, device)` remains once per epoch. (source: train.py diff; train.py L219)

### Informational Metrics
- final_test_acc: 91.42% (source: run.log L233)
- final_test_loss: 0.3365 (source: run.log L234)
- training_seconds: 300.0 (source: run.log L235)
- total_seconds: 396.2 (source: run.log L236)
- peak_vram_mb: 379.0 (source: run.log L238)
- num_epochs: 112 (source: run.log L239)
- num_steps: 43,398 (source: run.log L240)
- num_params: 269,722 (source: run.log L241)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
