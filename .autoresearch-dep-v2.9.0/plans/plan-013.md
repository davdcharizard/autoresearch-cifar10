# Plan EXP-013: EMA of Model Weights (Polyak Averaging)
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-013.md

## Milestones

### Milestone 1: Implement EMA in train.py
- [x] Add EMA shadow parameter initialization after model creation (deep copy of model state_dict)
- [x] Add per-step EMA update after `scaler.update()`: `ema_param = β * ema_param + (1-β) * param` with β=0.999
- [x] Add EMA weight swap-in before `evaluator.evaluate()` and swap-out after
- [x] Verify no syntax errors: `python -c "import ast; ast.parse(open('train.py').read())"`

### Milestone 2: Run experiment and collect results
- [x] Run `uv run train.py 2>&1 | tee exp-013.log`
- [x] Confirm training completes within 300s budget
- [x] Confirm full summary block is printed
- [x] Extract best_test_acc from output

### Milestone 3: Verify results against baseline
- [x] Check best_test_acc > 95.49% (baseline 95.39% + 0.1pp threshold) — FAILED (94.98%)
- [x] Check validation runs at most once per epoch — PASSED
- [x] Collect informational metrics — DONE

## Code Changes
- **train.py**: Add EMA of model weights with β=0.999. Three localized changes:
  1. **After model creation (after line 149)**: Create a shadow copy of model parameters as a dictionary mapping parameter names to cloned tensors. Initialize from model's initial state.
  2. **Inside the training loop, after `scaler.update()` (after line 223)**: Update EMA shadow parameters with the exponential moving average formula: `ema[name] = 0.999 * ema[name] + 0.001 * param.data` for each parameter. This runs once per step — the overhead is a single multiply-add per parameter, negligible compared to forward/backward.
  3. **Before evaluation (before line 256)**: Swap model parameters with EMA parameters for evaluation, then swap back after. Implementation: save current params → load EMA params → evaluate → restore original params. This ensures training continues with the original SGD weights.

## Configuration Changes
- No configuration changes. All hyperparameters (LR, WD, batch size, schedule, augmentation) remain identical to EXP-009 baseline.

## Execution Environment
- Method: local command `uv run train.py 2>&1 | tee exp-013.log`
- Resources: single H20 GPU (same as baseline)
- Estimated runtime: ~5-6 minutes total (300s training + startup + evaluation overhead)
- Log output: stdout captured to `exp-013.log` in project root via `tee`
- Tool skill: none (local execution)

## Abort Criteria
- Training loss becomes NaN or inf (divergence)
- No output for 60+ seconds (hang)
- OOM error or CUDA error in output
- Per-step time increases by more than 2ms compared to baseline (~16ms) — would indicate EMA overhead is non-negligible and reducing epoch count

## Verification Protocol

### Verification Procedure

After the training script completes:

1. **Condition 1 — Primary metric exceeds threshold**:
   - Command: `grep "^best_test_acc:" exp-013.log | awk '{print $2}' | tr -d '%'`
   - Pass: extracted value > 95.49
   - Fail: value <= 95.49 or missing
   - Timeout: N/A (post-completion check)

2. **Condition 2 — Script completes with full summary block**:
   - Command: `grep -c "^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:" exp-013.log`
   - Pass: count = 10 (all 10 summary lines present)
   - Fail: count < 10

3. **Condition 3 — Validation at most once per epoch**:
   - Command: `grep -c "eval ep" exp-013.log` and compare to `grep "^num_epochs:" exp-013.log | awk '{print $2}'`
   - Pass: eval count <= num_epochs
   - Fail: eval count > num_epochs

### Informational Metrics (Optional)
- training_seconds: `grep "^training_seconds:" exp-013.log | awk '{print $2}'`
- peak_vram_mb: `grep "^peak_vram_mb:" exp-013.log | awk '{print $2}'`
- final_test_acc: `grep "^final_test_acc:" exp-013.log | awk '{print $2}' | tr -d '%'`
- final_test_loss: `grep "^final_test_loss:" exp-013.log | awk '{print $2}'`
- num_epochs: `grep "^num_epochs:" exp-013.log | awk '{print $2}'`
- num_steps: `grep "^num_steps:" exp-013.log | awk '{print $2}'`
- num_params: `grep "^num_params:" exp-013.log | awk '{print $2}'`
