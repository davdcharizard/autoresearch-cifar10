# Experiment Log: EXP-025

## Execution

- **Created**: 2026-05-28
- **Brainstorm**: brainstorm/brainstorm-025.md
- **Plan**: plans/plan-025.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-025
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Added gradient centralization (GC) to the training loop. Inserted `scaler.unscale_(optimizer)` after the backward pass, followed by a loop over all model parameters that subtracts the per-output-channel mean from gradients with dim > 1 (conv and linear weights). BN parameters (1D) and biases (1D) are automatically excluded by the `dim() > 1` filter. The existing `scaler.step(optimizer)` then detects already-unscaled gradients and proceeds to `optimizer.step()` directly.

### Surprises & Discoveries
None — the implementation was straightforward. The `scaler.unscale_()` + manual gradient modification + `scaler.step()` pattern is well-documented in PyTorch AMP docs.

### Decisions
- Applied GC to all parameters with `grad.dim() > 1` rather than explicitly filtering by layer type. This catches both Conv2d weights (4D) and the final Linear weight (2D), while automatically excluding all 1D parameters (BN weight/bias, conv bias if any). This matches the GC paper's recommendation.
- Used `p.grad.data.sub_()` (in-place) to minimize memory allocation.

## Experimental Adjustments

(none)

## Run Log

### Run 1
- **Description**: Full training run with gradient centralization applied to all conv/linear weight gradients. Expected ~99 epochs in 300s budget with zero throughput cost. The GC loop adds negligible computation (mean subtraction per parameter tensor). Monitoring for any instability from modified gradient dynamics or unexpected throughput impact.
- **Job ID**: local
- **Log file**: run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-28
- **Ended**: 2026-05-28
- **Observations**: Training completed normally with 96 epochs (3 fewer than baseline's 99). Per-step time remained at 16ms but the GC gradient loop added ~0.5ms per step, accumulating to ~9s lost over 18500 steps — enough to cost 3 epochs. Wider test accuracy oscillations observed throughout training compared to baseline (e.g., 78.69% dip at epoch 34, 77.92% at epoch 22) suggesting GC modifies loss landscape geometry. Model converged steadily in final phase with consecutive new bests from epochs 88-94. Final convergence was slightly behind baseline's trajectory.
- **Key Metrics**: best_test_acc=96.49%, final_test_acc=96.39%, num_epochs=96, training_seconds=300.0, peak_vram_mb=864.6, num_steps=18529

## Verification Results

### Conditions Checked

1. **best_test_acc > 96.56%**: FAILED. Actual: 96.49% (+0.03pp above baseline 96.46%, but 0.07pp below threshold 96.56%). Source: run.log `best_test_acc: 96.49%`
2. **Clean completion**: PASSED. Summary block printed with all expected fields. Source: run.log final lines.
3. **Max 1 eval per epoch**: PASSED. 96 evals for 96 epochs. Source: `grep -c "eval ep" run.log` = 96.

### Informational Metrics

## Errors & Dead Ends

(none)

## Human Notes

(autopilot — no human interaction)
