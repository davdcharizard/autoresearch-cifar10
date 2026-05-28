# Experiment Log: EXP-028

## Execution

- **Created**: 2026-05-28
- **Brainstorm**: brainstorm/brainstorm-028.md
- **Plan**: plans/plan-028.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-028
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Two changes: `NUM_BLOCKS = 4` (ResNet-26, ~5.7M params) and `ESTIMATED_EPOCHS = 80` (recalibrate cosine schedule for shorter epoch budget).

### Surprises & Discoveries
None.

### Decisions
Set ESTIMATED_EPOCHS=80 rather than 75 to provide a slight over-estimate, ensuring cosine decay reaches very low LR by the actual final epoch. If we get ~75 epochs, LR at step 75/80 = progress 0.933, multiplier ≈ 0.011 (LR ≈ 0.002). Close enough to zero.

## Experimental Adjustments

(none)

## Run Log

### Run 1
- **Description**: Full training with ResNet-26 (NUM_BLOCKS=4, ~5.7M params) and adjusted cosine schedule (ESTIMATED_EPOCHS=80). Testing capacity increase to break the ~96.5% ceiling. Expected ~72-80 epochs in 300s with ~20-22ms/step. This is the first capacity increase since EXP-007 (WIDTH_MULT=4).
- **Job ID**: local
- **Log file**: run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-28
- **Ended**: 2026-05-28
- **Observations**: ResNet-26 (5.8M params) at 20-21ms/step completed 75 epochs (vs ~96 for baseline). Convergence was slower early (78.64% at epoch 11 vs ~81% for baseline) due to more parameters needing training. Model converged well in the final phase (best=final at 96.31%) but the 22% epoch reduction was too costly. The deeper model was still improving at epoch 75 (best=final), confirming undertrained. VRAM increased from 865MB to 1103MB.
- **Key Metrics**: best_test_acc=96.31%, final_test_acc=96.31%, num_epochs=75, training_seconds=300.0, peak_vram_mb=1103.1, num_steps=14512, num_params=5,836,106

## Verification Results

### Conditions Checked

1. **best_test_acc > 96.56%**: FAILED. Actual: 96.31% (-0.15pp below baseline 96.46%). Source: run.log `best_test_acc: 96.31%`
2. **Clean completion**: PASSED.
3. **Max 1 eval per epoch**: PASSED. 75 evals for 75 epochs.

### Informational Metrics

## Errors & Dead Ends

(none)

## Human Notes

(autopilot — no human interaction)
