# EXP-000: Cutout, Label Smoothing, and Cosine LR

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-000.md
- **Plan**: plans/plan-000.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-000
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Implemented the planned low-risk training recipe upgrade in `train.py` only. The data pipeline now adds cutout-style tensor masking through `transforms.RandomErasing`; the optimizer uses Nesterov SGD; the learning-rate schedule uses `CosineAnnealingLR` over the existing `MAX_STEPS` horizon; and the loss uses modest label smoothing. `uv run ruff check train.py` and `python3 -m py_compile train.py` both passed before launch.

### Surprises & Discoveries

No unexpected code structure issues appeared. `RandomErasing` fits the existing torchvision transform pipeline because it runs after `ToTensor()`.

### Decisions

Used an exact 16x16 square erasing area by setting both RandomErasing scale bounds to `(16 / 32) ** 2` and ratio bounds to `1.0`. This follows the plan's cutout-size intent more closely than a variable area range while staying inside the planned `RandomErasing` implementation.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 22105
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 08:40 UTC
- **Ended**: 2026-06-08 08:47 UTC

Description:
- Run the EXP-000 training recipe on one visible GPU with output redirected to `run.log`. This run tests whether cutout-style masking, label smoothing, Nesterov SGD, and cosine LR can improve `best_test_acc` over the 91.52% baseline without changing the model architecture or evaluation harness. Expected outcome is a completed final summary containing `best_test_acc`, runtime, VRAM, epoch, step, and parameter metrics.

Observations:
- Early run is healthy through epoch 3: best accuracy reached 65.52%, and no traceback/OOM/runtime error signature was present in the first log check. (source: run.log L7-L11)
- The run completed normally with a final summary and no traceback/OOM/runtime error signatures. Peak accuracy reached 90.45%, below the 91.52 baseline. (source: run.log L188-L198)

Key Metrics:
- best_test_acc: 90.45% (source: run.log L189)
- final_test_acc: 89.60% (source: run.log L190)
- final_test_loss: 0.3247 (source: run.log L191)
- training_seconds: 300.0 (source: run.log L192)
- total_seconds: 360.7 (source: run.log L193)
- peak_vram_mb: 330.1 (source: run.log L195)
- num_epochs: 91 (source: run.log L196)
- num_steps: 35279 (source: run.log L197)
- num_params: 269,722 (source: run.log L198)

## Verification Results

### Conditions Checked
- `uv run train.py` completes without crashing: passed. The run exited normally and emitted the final summary. (source: run.log L188-L198)
- The run reports a numeric `best_test_acc`: passed. `best_test_acc` was 90.45%. (source: run.log L189)
- `best_test_acc` improves over the current experiment-index baseline in the higher-is-better direction: failed. 90.45% is below the 91.52% baseline. (source: run.log L189; baseline: experiment-indices/maximize-cifar10-best-test-accuracy.tsv)
- The implementation respects all hard constraints: skipped — aborted verification after the failed primary metric condition.

### Informational Metrics
- Skipped — informational metrics are collected only when all necessary conditions pass.

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
