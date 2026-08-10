# EXP-002: 80%-Hold Cosine with Standard Momentum

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-002
- **Commit**: 5016cc4
- **Outcome**: completed

## Implementation Notes

### Summary

Changed only `train.py`: retain standard SGD momentum, hold `lr=0.1` through 80% of counted training, step to `0.01`, and cosine-decay to `1e-4` over the final 20%. Reused validated persistent training workers, four early evaluation checkpoints, and every-epoch evaluation throughout the late tail. Removed both `MultiStepLR` construction and `scheduler.step()` while preserving model, transforms, loss, seed, weight decay, batch size, evaluator, and fixed budget.

### Surprises & Discoveries

No implementation surprises. The external plan review improved measurement symmetry by requiring dense final-tail evaluation and clarified that the baseline's flat `0.01` regime must be preserved rather than replaced by a cosine mostly above `0.01`.

### Decisions

Implemented the reviewed refinement as a discontinuous step `0.1 -> 0.01` at 80%, followed by cosine `0.01 -> 1e-4`. Initialized terminal metrics to `None`, made `training_done` an unconditional evaluation trigger, and assert non-null immediately before summary. Verification accepts a variable dense-tail evaluation count but requires uniqueness and terminal epoch coverage.

## Experimental Adjustments

- **Densify final-tail evaluation and anneal entirely at/below 0.01**: Addresses external plan-review concerns about missed peaks and a mechanism weaker than the baseline's low-LR phase. (ref: `02-plan-review.md` concerns 1 and 3)

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 2109607 (timeout supervisor PID 2109603)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 08:07:51 UTC
- **Ended**: 2026-08-05 08:14:15 UTC

Description:
- Run the reviewed 80%-hold schedule on one H20 under the fixed 300-second training budget and 600-second total cutoff. The schedule preserves baseline-like exploration, uses standard momentum, and spends the complete final 20% at or below `0.01`. Expected result is `best_test_acc >= 91.77%`, a valid terminal summary, and unique dense-tail evaluations ending at the final epoch.

Observations:

- Process exited `0` before the 600-second timeout with no fatal patterns during monitoring. GPU utilization was healthy and the final summary was complete. (source: local PID 2109607; `run.log` L58-L67)
- Accuracy improved from 83.26% at the first checkpoint to a best of 91.83%; the terminal epoch scored 91.82%, so dense late evaluation captured a one-quantum pre-terminal peak. (source: `run.log` L6, L56, L58-L59)
- The run used 26 unique evaluation epochs and ended with evaluation epoch 100 equal to summary `num_epochs=100`. (source: verification command; `run.log` L56, L65)

Key Metrics:

- best_test_acc: 91.83% (source: `run.log` L58)
- final_test_acc: 91.82% (source: `run.log` L59)
- final_test_loss: 0.2843 (source: `run.log` L60)
- training_seconds: 300.0s; total_seconds: 336.0s (source: `run.log` L61-L62)
- peak_vram_mb: 330.1 MB (source: `run.log` L64)
- num_epochs: 100; num_steps: 38,629; num_params: 269,722 (source: `run.log` L65-L67)

## Verification Results

### Conditions Checked

- **Accuracy improvement**: PASS — `91.83% >= 91.77%`, delta `+0.16` points over baseline `91.67%`. (source: `run.log` L58)
- **Valid numeric summary**: PASS — all ten required fields occurred exactly once with finite numeric values. (source: `run.log` L58-L67)
- **Fixed budget and wall limit**: PASS — `training_seconds=300.0`, `total_seconds=336.0 < 600.0`. (source: `run.log` L61-L62)
- **Evaluation integrity**: PASS — 26 evaluation lines, 26 unique epochs, terminal evaluation epoch 100 equals `num_epochs=100`. (source: verification command; `run.log` L56, L65)

### Informational Metrics

- final_test_acc: 91.82% (source: `run.log` L59)
- final_test_loss: 0.2843 (source: `run.log` L60)
- training_seconds: 300.0s (source: `run.log` L61)
- total_seconds: 336.0s (source: `run.log` L62)
- startup_seconds: 1.0s (source: `run.log` L63)
- peak_vram_mb: 330.1 MB (source: `run.log` L64)
- num_epochs: 100 (source: `run.log` L65)
- num_steps: 38,629 (source: `run.log` L66)
- num_params: 269,722 (source: `run.log` L67)

## Errors & Dead Ends

## Human Notes

> The user requires external Claude adversarial reviews with no fallback; EXP-002 idea and plan reviews completed successfully.
