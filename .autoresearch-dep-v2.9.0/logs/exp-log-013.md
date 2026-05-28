# EXP-013: EMA of Model Weights (Polyak Averaging)

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-013.md
- **Plan**: plans/plan-013.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-013
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented EMA of model weights in train.py with three localized changes following the plan exactly. (1) After model creation on line 150, initialized `ema_shadow` as a dictionary mapping parameter names to cloned tensors from the initial model state. (2) Inside the training loop after `scaler.update()`, added a `torch.no_grad()` block that updates each shadow parameter with the EMA formula: `ema[name] = 0.999 * ema[name] + 0.001 * param.data` using in-place `mul_` and `add_`. (3) Before evaluation, swap model parameters with EMA shadow for eval, then restore original params after. The swap uses `orig_params` backup dict → copy EMA in → evaluate → copy originals back.

### Surprises & Discoveries

None. The implementation was straightforward — train.py has clean separation between training steps and evaluation, making the swap-in/swap-out pattern easy to insert.

### Decisions

Used `torch.no_grad()` for the EMA update block to avoid unnecessary gradient tracking overhead, even though the operations are on `.data` tensors. This is a minor defensive measure with zero cost.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/exp-013.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27
- **Ended**: 2026-05-27

Description:
- Running EMA experiment with β=0.999 on the EXP-009 baseline (WIDTH_MULT=4, batch 256, AMP, all augmentation). Expecting ~98 epochs in 300s with negligible per-step overhead from EMA updates. The EMA-averaged weights should show improved test accuracy (target >95.49%) by smoothing SGD noise, particularly after LR drops where weight oscillations increase.

Observations:
- Training completed normally in 300.0s (93 epochs, 18064 steps) with no errors
- EMA eval accuracy showed two distinct phases due to BN running stats mismatch: (a) epochs 1-59: severe suppression at 80-90% because BN buffers (running_mean/running_var) are not included in EMA shadow and remain computed from non-EMA forward passes, (b) epochs 60-93: rapid recovery after second LR drop (75% mark, LR→0.002) as SGD weights stabilize and BN stats become more compatible with EMA weights, climbing from ~87% to peak 94.98%
- Best EMA eval accuracy: 94.98% at epoch 91 — 0.41pp below baseline (95.39%) and 0.51pp below threshold (95.49%)
- The naive parameter-only EMA implementation fundamentally cannot match baseline because it only averages nn.Parameter tensors, not nn.Module buffers (BatchNorm running_mean/running_var). The EMA-averaged conv/fc weights are evaluated with BN statistics computed from the original SGD trajectory, creating a mismatch
- Per-step time remained ~16-17ms throughout — zero overhead from EMA updates as expected

Key Metrics:
- best_test_acc: 94.98%
- final_test_acc: 94.82%
- final_test_loss: 0.1616
- training_seconds: 300.0
- total_seconds: 404.1
- startup_seconds: 1.2
- peak_vram_mb: 899.5
- num_epochs: 93
- num_steps: 18064
- num_params: 4,286,026

## Verification Results

### Conditions Checked

1. **Primary metric exceeds threshold**: FAILED
   - Command: `grep "^best_test_acc:" exp-013.log | awk '{print $2}' | tr -d '%'`
   - Result: 94.98
   - Required: > 95.49
   - Verdict: 94.98 ≤ 95.49 — condition failed

2. **Script completes with full summary block**: PASSED
   - Command: `grep -c "^best_test_acc:\|^final_test_acc:..." exp-013.log`
   - Result: 10 (all 10 summary lines present)
   - Required: count = 10

3. **Validation at most once per epoch**: PASSED
   - eval count: 93, num_epochs: 93
   - 93 ≤ 93 — condition passed

### Informational Metrics

- training_seconds: 300.0
- peak_vram_mb: 899.5
- final_test_acc: 94.82%
- final_test_loss: 0.1616
- num_epochs: 93
- num_steps: 18064
- num_params: 4,286,026

## Errors & Dead Ends

## Human Notes

> 
