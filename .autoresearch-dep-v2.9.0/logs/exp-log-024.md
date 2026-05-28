# Experiment Log: EXP-024

## Execution

- **Created**: 2026-05-28
- **Brainstorm**: brainstorm/brainstorm-024.md
- **Plan**: plans/plan-024.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-024
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented BN bias parameter group separation in the optimizer construction. Iterated `model.named_parameters()` to identify BatchNorm bias parameters (name contains `bn` and ends with `.bias`), creating two parameter groups: `norm_biases` with `lr=LR*64.0` and `weight_decay=0`, and `other_params` with `lr=LR` and `weight_decay=WEIGHT_DECAY`. The existing `LambdaLR` cosine warmup+decay scheduler applies multiplicatively to all param groups, maintaining the 64x ratio throughout training.

### Surprises & Discoveries
None — the implementation was straightforward. The existing optimizer construction was a single `optim.SGD(model.parameters(), ...)` call, cleanly replaceable with parameter groups. The `LambdaLR` scheduler uses a single `lr_lambda` function applied to all param groups, so no scheduler changes were needed.

### Decisions
- Set `weight_decay=0` for BN bias group: biases should not be weight-decayed (they have no regularization benefit from L2 penalty), and the 64x LR already provides strong implicit regularization through parameter magnitude.
- Parameter identification via `"bn" in name and name.endswith(".bias")`: matches exactly the 19 BN bias parameters (18 in BasicBlocks + 1 in stem). This is specific enough to avoid false positives (no other parameters have "bn" in their name in this architecture).

## Human Notes

(autopilot — no human interaction)

## Run Log

### Run 1
- **Description**: Full training run with BN bias 64x LR multiplier. Expected ~99 epochs in 300s budget with zero throughput cost. Monitoring for NaN loss or training instability from the aggressive BN bias LR (effective 12.8 at peak).
- **Job ID**: local
- **Log file**: run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-28
- **Ended**: 2026-05-28
- **Observations**: Severe instability during warmup and high-LR phase. BN bias LR ramped to 12.8 during 5-epoch warmup, causing wild test accuracy oscillations (26-65% range through epoch 20). Model only stabilized after epoch 80 when cosine decay brought LR down sufficiently. Zero throughput cost confirmed (99 epochs, 16ms/step, identical to baseline).
- **Key Metrics**: best_test_acc=94.47%, final_test_acc=94.45%, num_epochs=99, training_seconds=300.0, peak_vram_mb=864.6

## Experimental Adjustments

(none)

## Verification Results

### Conditions Checked

1. **best_test_acc > 96.56%**: FAILED. Actual: 94.47% (1.99pp below baseline 96.46%, 2.09pp below threshold 96.56%). Source: run.log `best_test_acc: 94.47%`
2. **Clean completion**: PASSED. Summary block printed with all expected fields. Source: run.log final lines.
3. **Max 1 eval per epoch**: PASSED. 99 evals for 99 epochs. Source: `grep -c "eval ep" run.log` = 99.

## Errors & Dead Ends

(none)
