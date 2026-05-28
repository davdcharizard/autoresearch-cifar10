# Experiment Log: EXP-030

## Execution

- **Created**: 2026-05-28
- **Brainstorm**: brainstorm/brainstorm-030.md
- **Plan**: plans/plan-030.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-030
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Changed normalization std from (1,1,1) to proper CIFAR-10 per-channel std (0.2470, 0.2435, 0.2616). Removed the comment about original paper per-pixel mean since we're now using standard normalization.

### Surprises & Discoveries
None.

### Decisions
None.

## Experimental Adjustments

(none)

## Run Log

### Run 1
- **Description**: Full training with proper per-channel std normalization. Testing whether standard CIFAR-10 preprocessing improves accuracy by better matching Kaiming init assumptions. Expected ~96-98 epochs at 16ms/step.
- **Job ID**: local
- **Log file**: run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-28
- **Ended**: 2026-05-28
- **Observations**: Massive regression — 94.67% vs 96.46% baseline (-1.79pp). 99 epochs at 16ms/step (zero throughput cost). The 4x wider input distribution from proper std normalization completely disrupted optimization dynamics. The model's hyperparameters (LR, WD, cosine schedule) were all tuned for the std=(1,1,1) input scale. Changing the input scale is effectively changing the learning rate for the first layer by 4x.
- **Key Metrics**: best_test_acc=94.67%, final_test_acc=94.63%, num_epochs=99, training_seconds=300.0, peak_vram_mb=864.6

## Verification Results

### Conditions Checked

1. **best_test_acc > 96.56%**: FAILED. Actual: 94.67% (-1.79pp below baseline). Source: run.log.
2. **Clean completion**: PASSED.
3. **Max 1 eval per epoch**: PASSED. 99 evals for 99 epochs.

### Informational Metrics

## Errors & Dead Ends

(none)

## Human Notes

(autopilot — no human interaction)
