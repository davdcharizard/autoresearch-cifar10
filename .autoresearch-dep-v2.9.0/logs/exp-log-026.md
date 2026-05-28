# Experiment Log: EXP-026

## Execution

- **Created**: 2026-05-28
- **Brainstorm**: brainstorm/brainstorm-026.md
- **Plan**: plans/plan-026.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-026
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Single parameter change: added `nesterov=True` to `optim.SGD()`. No other code changes needed.

### Surprises & Discoveries
None — the change is a single keyword argument.

### Decisions
None — followed the plan exactly.

## Experimental Adjustments

(none)

## Run Log

### Run 1
- **Description**: Full training run with Nesterov momentum enabled. Expected ~99 epochs in 300s budget with zero throughput cost. Nesterov changes the momentum update formula (look-ahead gradients) without adding any computation. Monitoring for any throughput impact or training instability.
- **Job ID**: local
- **Log file**: run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-28
- **Ended**: 2026-05-28
- **Observations**: Training completed with 96 epochs (same as EXP-025, vs baseline's expected ~99). 16ms/step confirmed — Nesterov adds zero per-step overhead. Wider mid-training oscillations compared to baseline (dips to 73.47% at epoch 34, 86.39% at epoch 59). Model converged well in the final phase with consecutive new bests from epochs 82-93. Final convergence was very similar to EXP-025's trajectory. best=final epoch (96.52% at epoch 96).
- **Key Metrics**: best_test_acc=96.52%, final_test_acc=96.52%, num_epochs=96, training_seconds=300.0, peak_vram_mb=864.6, num_steps=18659

## Verification Results

### Conditions Checked

1. **best_test_acc > 96.56%**: FAILED. Actual: 96.52% (+0.06pp above baseline 96.46%, but 0.04pp below threshold 96.56%). Source: run.log `best_test_acc: 96.52%`
2. **Clean completion**: PASSED. Summary block printed with all expected fields.
3. **Max 1 eval per epoch**: PASSED. 96 evals for 96 epochs.

### Informational Metrics

## Errors & Dead Ends

(none)

## Human Notes

(autopilot — no human interaction)
