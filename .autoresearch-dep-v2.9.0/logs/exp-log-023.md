# EXP-023: Cubic-Scheduled EMA (Lookahead)

## Execution

Overall Status & Info:
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-023.md
- **Plan**: plans/plan-023.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-023
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the LookaheadEMA class (~25 lines) after the model definition in train.py. The class maintains a shadow copy of the full model state_dict (including BN buffers), with `update()`, `swap()`, `restore()`, and `copy_to()` methods. Integrated into the training loop: every 5 steps after `scheduler.step()`, `ema.update(model, step, total_steps)` is called with cubic decay `alpha = 0.95^5 * (step/total_steps)^3`. Before per-epoch TTA evaluation, EMA weights are swapped in via `ema.swap(model)` and restored after via `ema.restore(model)`. On the final epoch (budget exhausted), `ema.copy_to(model)` is used instead so the final summary reflects EMA weights permanently.

### Surprises & Discoveries

- The initial `update()` method signature omitted the `model` parameter — it referenced `model` as a free variable from the enclosing scope. Fixed by adding `model` as the first positional argument. The call site was already correct.

### Decisions

- No deviations from the plan. All implementation follows plan-023.md exactly.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: (local run)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/exp-023-run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-28
- **Ended**: 2026-05-28

Description:
- Running `uv run python train.py` with output captured to exp-023-run.log. This trains the WIDTH_MULT=4 ResNet-20 on CIFAR-10 with the new cubic-scheduled EMA (Lookahead) for 300s on H20 GPU. We expect ~95-99 epochs to complete with minimal throughput degradation from the every-5-steps EMA update. The hypothesis is best_test_acc > 96.56% (baseline 96.46% + 0.1pp threshold).

Observations:
- Training completed 95 epochs in 300.0s (vs baseline ~99 epochs — ~4 epoch throughput cost from EMA overhead)
- Early-epoch EMA evaluations showed near-random accuracy (6.14% at epoch 1), indicating the shadow severely lags model weights during high-LR warmup phase
- EMA accuracy gradually caught up: ~88% by epoch 20, ~95.4% by epoch 85+, but never matched baseline performance
- best_test_acc peaked at 96.02% (epoch 90), well below baseline 96.46%
- The cubic schedule `alpha = 0.95^5 * (step/total_steps)^3` starts near-zero and grows slowly — this means early training steps barely update the shadow, causing the severe initial lag

Key Metrics:
- best_test_acc: 96.02%
- final_test_acc: 95.92%
- final_test_loss: 0.3001
- training_seconds: 300.0
- peak_vram_mb: 881.3
- num_epochs: 95
- num_steps: 18464
- num_params: 4,286,026

## Verification Results

### Conditions Checked

1. **Condition 1 — Accuracy threshold**: `grep "^best_test_acc:" exp-023-run.log` → `96.02%`. Required > 96.56%. **FAILED** (96.02% < 96.56%, delta = -0.44pp from baseline).

2. **Condition 2 — Clean completion**: `grep -c "^best_test_acc:" exp-023-run.log` → 1. Summary block printed with all fields. **PASSED**.

3. **Condition 3 — Validation frequency**: `grep -c "eval ep" exp-023-run.log` → 95. `num_epochs: 95`. Eval count matches epoch count (95 = 95). **PASSED**.

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> 
