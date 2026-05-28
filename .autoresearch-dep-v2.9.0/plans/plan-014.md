# Plan EXP-014: Full State Dict EMA (β=0.999)
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-014.md

## Milestones

### Milestone 1: Implement Full State Dict EMA in train.py
- [x] After model creation (after line 151), initialize `ema_shadow` as a deep copy of `model.state_dict()` — this includes all parameters AND BatchNorm running_mean/running_var buffers
- [x] After `scaler.update()` (after line 222), add a `torch.no_grad()` block that updates every key in `ema_shadow` via `ema_shadow[k].lerp_(model.state_dict()[k], 1 - EMA_BETA)` — this is the in-place EMA update formula
- [x] Before `evaluator.evaluate()` (before line 256), save the current model state_dict, load `ema_shadow` into the model, run eval, then restore the original state_dict
- [x] Verify: read through the complete modified train.py to confirm no syntax errors, correct indentation, and that EMA update covers all state_dict keys (not just parameters)

### Milestone 2: Run the experiment
- [x] Execute `uv run train.py > run.log 2>&1` with output captured to `run.log`
- [x] Confirm training starts (check early log lines for device, model info, step output)
- [x] Wait for completion (~5 minutes)

### Milestone 3: Verify results
- [x] Extract `best_test_acc` from `run.log` via `grep "^best_test_acc:" run.log`
- [x] Check best_test_acc > 95.49% (baseline 95.39% + 0.1pp threshold)
- [x] Confirm full summary block is printed (training_seconds, num_epochs, etc.)
- [x] Confirm evaluation runs at most once per epoch (count `eval ep` lines vs num_epochs)

## Code Changes
- **train.py**: Three localized additions implementing full state_dict EMA:

  1. **EMA shadow initialization** (after model creation, ~line 151): Add `EMA_BETA = 0.999` constant and `import copy` at top. Initialize `ema_shadow = copy.deepcopy(model.state_dict())`. This deep copy captures all tensors including BatchNorm `running_mean`, `running_var`, and `num_batches_tracked` buffers — the critical fix for EXP-013's failure.

  2. **EMA update in training loop** (after `scaler.update()`, ~line 222): Inside a `torch.no_grad()` block, iterate over the current `model.state_dict()` and update each entry in `ema_shadow` using the exponential moving average formula. Use `torch.lerp` (or `.lerp_()`) for float tensors. For integer tensors like `num_batches_tracked`, copy directly from the model since averaging integers is meaningless. This runs once per step with negligible overhead.

  3. **EMA swap for evaluation** (before/after `evaluator.evaluate()`, ~line 256): Before eval, save the current model state_dict, load `ema_shadow` into the model via `model.load_state_dict(ema_shadow)`. After eval, restore the original state_dict. The model is already in eval mode (`.eval()` not explicitly called but `evaluator.evaluate()` handles it). After restoring, set model back to train mode for the next epoch.

  **Rationale**: EXP-013 proved EMA works (late-training recovery to 94.98%) but failed because `named_parameters()` excludes BN buffers. Using `state_dict()` includes all tensors — parameters, BN running stats, and BN num_batches_tracked — eliminating the mismatch entirely.

## Configuration Changes
- EMA_BETA: N/A → 0.999 (standard value from literature; hlb-CIFAR10 uses EMA with comparable β for short training runs)
- All other hyperparameters unchanged from EXP-009 baseline (BATCH_SIZE=256, LR=0.2, MOMENTUM=0.9, WEIGHT_DECAY=5e-4, WIDTH_MULT=4, NUM_BLOCKS=3, WARMUP_EPOCHS=5)

## Execution Environment
- Method: local command (`uv run train.py > run.log 2>&1`)
- Resources: single H20 GPU (same as all prior experiments)
- Estimated runtime: ~5 minutes total (300s training budget + ~10s startup/eval overhead)
- Log output: stdout+stderr captured to `run.log` in project root via shell redirection
- Tool skill: none (local execution)

## Abort Criteria
- No output in run.log after 60 seconds — indicates script crash at startup
- `NaN` or `inf` appearing in loss values — indicates training instability
- Per-step time exceeding 25ms consistently (baseline is 16ms) — would indicate EMA overhead is non-negligible, reducing epoch count unacceptably
- Python traceback or error in run.log — indicates code bug
- Training not progressing past epoch 1 within 30 seconds — indicates evaluation hang

## Verification Protocol

### Verification Procedure

**Condition 1: best_test_acc > 95.49%**
- Command: `grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- Pass: extracted value > 95.49
- Fail: extracted value ≤ 95.49 or not found
- Timeout: N/A (post-completion check on log file)

**Condition 2: Training script completes and prints full summary block**
- Command: `grep -c "^best_test_acc:\|^final_test_acc:\|^training_seconds:\|^num_epochs:\|^num_steps:" run.log`
- Pass: count = 5 (all five summary fields present)
- Fail: count < 5
- Timeout: N/A

**Condition 3: Validation runs at most once per epoch**
- Command: `num_eval=$(grep -c "eval ep" run.log); num_epochs=$(grep "^num_epochs:" run.log | awk '{print $2}'); echo "evals=$num_eval epochs=$num_epochs"`
- Pass: num_eval ≤ num_epochs
- Fail: num_eval > num_epochs
- Timeout: N/A

### Informational Metrics (Optional)
- training_seconds: `grep "^training_seconds:" run.log | awk '{print $2}'`
- peak_vram_mb: `grep "^peak_vram_mb:" run.log | awk '{print $2}'`
- final_test_acc: `grep "^final_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- final_test_loss: `grep "^final_test_loss:" run.log | awk '{print $2}'`
- num_epochs: `grep "^num_epochs:" run.log | awk '{print $2}'`
- num_steps: `grep "^num_steps:" run.log | awk '{print $2}'`
- num_params: `grep "^num_params:" run.log | awk '{print $2}'`
