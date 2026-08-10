# EXP-003: Modest Label Smoothing

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-003
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Added `LABEL_SMOOTHING = 0.05` and passed it to the existing training-only `F.cross_entropy` in `train.py`. Preserved the accepted EXP-002 model, transforms, loader, optimizer, elapsed-time schedule, evaluation cadence, seed, and fixed evaluator. PyTorch API, compilation, Ruff lint/format, pre-commit, diff, and scope checks all passed before launch.

### Surprises & Discoveries

No code-level surprises. Pre-commit mechanically collapsed the cross-entropy call to one line. External Claude plan review identified GPU contention as a fairness confound because counted training time includes synchronized step duration; preflight therefore confirmed the only visible H20 was idle with zero used memory and no compute process.

### Decisions

Adopted Claude's recommendation to require an idle GPU and loosen the counted-time sanity ceiling to `<310s`, avoiding a false failure from final-step overshoot while still checking the fixed 300-second budget. Rejected no substantive review concern; noted that a threshold result is protocol-valid but not alone strong causal evidence.

## Experimental Adjustments

- **Require idle selected H20**: Prevents contention from changing the number of optimizer steps completed within counted wall time. (ref: `02-plan-review.md` concern 1)
- **Use a non-brittle counted-time sanity band**: Allows final-step overshoot without weakening the fixed-budget check. (ref: `02-plan-review.md` concern 2)

## Run Log

### Run 1

Metadata:
- **Job ID**: local training PID 2121713 (timeout supervisor PID 2121709)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 09:06 UTC
- **Ended**: 2026-08-05 09:13 UTC

Description:
- Run the isolated `label_smoothing=0.05` intervention once on idle H20 GPU 0 under the unchanged EXP-002 schedule and 300-second counted training budget. The moving baseline is `91.83%` at commit `5016cc4`; improvement requires `best_test_acc >= 91.93%`. Expected step count, VRAM, and total runtime should remain close to EXP-002 because smoothing adds negligible work.

Observations:
- Preflight found exactly one visible idle NVIDIA H20: index 0, 97,871 MiB total, 0 MiB used, 0% utilization, and no compute process. (source: pre-launch `nvidia-smi` at 2026-08-05 09:06 UTC)
- Moving baseline confirmed as 91.83% at commit 5016cc4. (source: pre-launch `exp-index.sh baseline` at 2026-08-05 09:06 UTC)
- Run produced training steps normally within 11 seconds; step 250 reported finite smoothed loss 1.5821 at `lr=0.1000` and about 16,392 images/s. (source: `run.log` compact tail at 2026-08-05 09:06 UTC)
- Process exited `0` before the timeout and printed a complete summary. Best accuracy was 91.83%, exactly equal to the moving baseline and 0.10 points below the required 91.93%. (source: `run.log` L54-L63)
- Final test loss improved from EXP-002's 0.2843 to 0.2740, but final accuracy was 91.80%; this is the calibration/NLL-versus-top-1 divergence anticipated by external review. (source: `run.log` L54-L56; EXP-002 `04-analysis.md` § Results)
- The run completed 36,039 steps versus EXP-002's 38,629, a 6.7% reduction despite identical architecture and budget. The label-smoothing loss path therefore had a non-negligible fixed-time throughput cost. (source: `run.log` L57, L62; EXP-002 `04-analysis.md` § Results)

Key Metrics:
- best_test_acc: 91.83% (source: `run.log` L54)
- final_test_acc: 91.80% (source: `run.log` L55)
- final_test_loss: 0.2740 (source: `run.log` L56)
- training_seconds: 300.0s; total_seconds: 333.6s; startup_seconds: 1.0s (source: `run.log` L57-L59)
- peak_vram_mb: 330.1 MB (source: `run.log` L60)
- num_epochs: 93; num_steps: 36,039; num_params: 269,722 (source: `run.log` L61-L63)

## Verification Results

### Conditions Checked

- **Accuracy improvement**: FAIL — `91.83% < 91.93%`; delta `0.00` points from moving baseline `91.83%`. (source: `run.log` L54)
- **Clean completion and numeric summary**: skipped — verification stopped after the primary necessary condition failed, as required by protocol.
- **Fixed budget and wall limit**: skipped — verification stopped after the primary necessary condition failed, as required by protocol.

### Informational Metrics

- Not formally collected because the primary necessary condition failed. Inline run metrics above are retained as execution evidence for analysis.

## Errors & Dead Ends

## Human Notes

> The user requires external Claude adversarial reviews with no fallback; both EXP-003 idea and plan reviews completed successfully.
