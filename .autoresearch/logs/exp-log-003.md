# EXP-003: Earlier Second LR Drop

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md
- **Plan**: plans/plan-003.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-003
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Implemented the planned scheduler-only change on top of the current EXP-002 integration branch. The only source change is moving the second `MultiStepLR` milestone from 48,000 to 40,000 steps so the fixed-budget run can enter LR 0.001.

### Surprises & Discoveries

No implementation surprises. Static checks passed and the diff is exactly one scheduler line.

### Decisions

Kept the first milestone at 32,000 to preserve the successful EXP-002 transition into LR 0.01 and isolate only the second drop timing.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 39611
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 10:32 UTC
- **Ended**: 2026-06-08 10:39 UTC

Description:
- Run EXP-003 with the successful FP32 throughput setup from EXP-002 and only the second LR milestone changed to 40,000. This tests whether a reachable LR 0.001 refinement phase improves beyond the 91.95 baseline. Expected output is a clean final summary plus log evidence that LR reaches 0.001 after step 40,000.

Observations:
- The run completed cleanly and reached LR 0.001 after step 40,000, but best accuracy was 91.85%, below the 91.95 baseline. (source: run.log L209-L249)

Key Metrics:
- best_test_acc: 91.85% (source: run.log L240)
- final_test_acc: 91.66% (source: run.log L241)
- final_test_loss: 0.3139 (source: run.log L242)
- training_seconds: 300.0 (source: run.log L243)
- total_seconds: 404.6 (source: run.log L244)
- startup_seconds: 3.5 (source: run.log L245)
- peak_vram_mb: 379.0 (source: run.log L246)
- num_epochs: 117 (source: run.log L247)
- num_steps: 45,279 (source: run.log L248)
- num_params: 269,722 (source: run.log L249)

## Verification Results

### Conditions Checked
- `uv run train.py` completes without crashing: passed. The run exited normally and emitted a final summary. (source: run.log L239-L249)
- The run reports a numeric `best_test_acc`: passed. `best_test_acc` was 91.85%. (source: run.log L240)
- `best_test_acc` improves over the current experiment-index baseline in the higher-is-better direction: failed. 91.85% is below the 91.95% baseline. (source: run.log L240; baseline: experiment-indices/maximize-cifar10-best-test-accuracy.tsv)
- The implementation respects all hard constraints: skipped — aborted verification after the failed primary metric condition.

### Informational Metrics
- Skipped — informational metrics are collected only when all necessary conditions pass.

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
