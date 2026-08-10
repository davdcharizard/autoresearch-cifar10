# EXP-005: Early Weak-Phase Adaptation

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-005
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Added `AUG_SWITCH_FRACTION=0.75` and changed exactly the post-batch augmentation break and loader-switch predicates. Preserved LR scheduling and dense evaluation at `LR_HOLD_FRACTION=0.8`, producing a predeclared 75-80% interval of weak crop/flip training at `lr=0.1`. All other accepted EXP-004 behavior remains unchanged.

### Surprises & Discoveries

No implementation surprises. External Claude review emphasized the four occurrences of `LR_HOLD_FRACTION`; explicit `rg` verification confirmed only the two augmentation meanings moved, while LR and evaluation meanings stayed at 0.8.

### Decisions

Retained one fixed seed 42 and the user's official moving-baseline threshold rather than adding incomparable seed reruns. Deliberately did not add a 75% evaluation because that would give EXP-005 an extra best-checkpoint opportunity absent from EXP-004; phase wiring is verified from switch and LR logs.

## Experimental Adjustments

- **Line-semantic predicate guard**: Verified the two augmentation predicates moved and the two LR/evaluation predicates did not. (ref: `02-plan-review.md` concern 5)
- **Preserve evaluation cadence**: Avoided a new 75% model-selection checkpoint. (ref: `02-plan-review.md` concern 4, independently assessed)

## Run Log

### Run 1

Metadata:
- **Job ID**: local training PID 2155801 (timeout supervisor PID 2155797)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 11:29 UTC
- **Ended**: 2026-08-05 11:37 UTC

Description:
- Move only the strong-to-weak augmentation boundary from 80% to 75%, preserving the accepted magnitude-7 policy and optimizer schedule. This gives clean crop/flip and BatchNorm statistics 15 counted seconds at `lr=0.1` before annealing. Baseline is 92.30%; improvement requires at least 92.40%.

Observations:
- Preflight confirmed baseline 92.30% at commit 11f8469, no stale log, and one idle H20 with 97,871 MiB, 0 MiB used, 0% utilization, and no compute process. (source: pre-launch commands at 2026-08-05 11:29 UTC)
- Training emitted finite steps normally; step 1,750 reached 4.7% at `lr=0.1000` with about 16.6k images/s. (source: compact `run.log` tail)
- Switch occurred exactly once at epoch 74 and 75.0%, stopping all eight workers; resumed steps through 78.5% logged `lr=0.1000`. (source: `run.log` switch and step lines)
- Process exited 0 with best accuracy 92.12%, 0.18 points below baseline and 0.28 below the required threshold. (source: `run.log` L57-L66)

Key Metrics:
- best_test_acc: 92.12%; final_test_acc: 91.98%; final_test_loss: 0.2624. (source: `run.log` L57-L59)
- training_seconds: 300.0s; total_seconds: 339.6s; startup_seconds: 1.0s. (source: `run.log` L60-L62)
- peak_vram_mb: 330.1 MB; epochs: 99; steps: 38,234; params: 269,722. (source: `run.log` L63-L66)

## Verification Results

### Conditions Checked

- **Accuracy improvement**: FAIL — `92.12% < 92.40%`; delta `-0.18` points from baseline 92.30%. (source: `run.log` L57)
- **Clean completion and summary**: skipped after primary failure.
- **Budget/wall limit and integrity**: skipped after primary failure.

### Informational Metrics

- Not formally collected after the primary failure; inline run metrics are retained for analysis.

## Errors & Dead Ends

## Human Notes

> External Claude idea and plan reviews both completed successfully with no fallback.
