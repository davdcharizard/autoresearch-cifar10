# EXP-001: Baseline-Preserving Throughput Acceleration

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-001.md
- **Plan**: plans/plan-001.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-001
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Implemented the throughput-only experiment in `train.py`. The baseline model, augmentation, cross-entropy targets, SGD hyperparameters, LR milestones, and once-per-epoch evaluation remain unchanged. Added feature flags for cuDNN benchmarking, channels-last memory format, CUDA BF16 autocast, and `torch.compile`; ruff and Python compilation both passed.

### Surprises & Discoveries

No static issues appeared. The implementation deliberately computes `num_params` before wrapping the model with `torch.compile`, keeping the reported parameter count tied to the original ResNet module.

### Decisions

Used BF16 autocast instead of FP16 autocast to avoid introducing `GradScaler` and reduce the risk of loss-scaling dynamics becoming an extra experimental variable. Left compile enabled for the first run; if runtime fails specifically in the compiler path, the retry should disable only `USE_COMPILE` while retaining channels-last and BF16.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 80738
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 08:55 UTC
- **Ended**: 2026-06-08 09:01 UTC

Description:
- Run the baseline statistical training recipe with throughput mechanisms enabled. This tests whether channels-last layout, cuDNN benchmarking, BF16 autocast, and `torch.compile` can increase useful optimizer steps inside the fixed 300 second training budget. Expected result is a clean final summary with higher `num_steps` than EXP-000 and, if the baseline is step-limited, `best_test_acc` above 91.52%.

Observations:
- Run 1 started cleanly; by epoch 16 it had no compiler/autocast/OOM errors and was showing mostly 6-8 ms training steps, with best accuracy 83.60%. (source: run.log L19-L38)
- The run completed cleanly and increased throughput to 39,558 steps / 102 epochs, but the best accuracy was 91.48%, narrowly below the 91.52 baseline. (source: run.log L211-L221)

Key Metrics:
- best_test_acc: 91.48% (source: run.log L212)
- final_test_acc: 91.23% (source: run.log L213)
- final_test_loss: 0.3248 (source: run.log L214)
- training_seconds: 300.0 (source: run.log L215)
- total_seconds: 394.0 (source: run.log L216)
- startup_seconds: 4.1 (source: run.log L217)
- peak_vram_mb: 276.9 (source: run.log L218)
- num_epochs: 102 (source: run.log L219)
- num_steps: 39,558 (source: run.log L220)
- num_params: 269,722 (source: run.log L221)

## Verification Results

### Conditions Checked
- `uv run train.py` completes without crashing: passed. The run exited normally and emitted a final summary. (source: run.log L211-L221)
- The run reports a numeric `best_test_acc`: passed. `best_test_acc` was 91.48%. (source: run.log L212)
- `best_test_acc` improves over the current experiment-index baseline in the higher-is-better direction: failed. 91.48% is below the 91.52% baseline. (source: run.log L212; baseline: experiment-indices/maximize-cifar10-best-test-accuracy.tsv)
- The implementation respects all hard constraints: skipped — aborted verification after the failed primary metric condition.

### Informational Metrics
- Skipped — informational metrics are collected only when all necessary conditions pass.

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
