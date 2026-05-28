# Plan EXP-025: Gradient Centralization
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-025.md

## Milestones

### Milestone 1: Implement gradient centralization in training loop
- [x] Add `scaler.unscale_(optimizer)` call after `scaler.scale(loss).backward()` to get actual gradients
- [x] Add gradient centralization loop: for each parameter with `grad.dim() > 1`, subtract the mean across all dims except dim 0 (output channel)
- [x] Move `scaler.step(optimizer)` after the GC loop (already in correct position, just verify)
- [x] Verify no syntax errors by doing a dry check of the modified code

### Milestone 2: Run experiment and capture output
- [ ] Run `uv run python train.py > run.log 2>&1` and confirm training starts
- [ ] Confirm training completes within 300s budget with ~99 epochs (zero throughput cost expected)

### Milestone 3: Verify results
- [ ] Extract best_test_acc from run.log
- [ ] Check best_test_acc > 96.56% (baseline 96.46% + 0.1pp threshold)

## Code Changes
- **train.py** (lines 220-222): Insert gradient centralization between the backward pass and the optimizer step. The current code flow is:
  ```
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  ```
  This becomes:
  ```
  scaler.scale(loss).backward()
  scaler.unscale_(optimizer)
  for p in model.parameters():
      if p.grad is not None and p.grad.dim() > 1:
          p.grad.data.sub_(p.grad.data.mean(dim=tuple(range(1, p.grad.dim())), keepdim=True))
  scaler.step(optimizer)
  scaler.update()
  ```

  The `scaler.unscale_(optimizer)` call converts scaled gradients back to actual gradients. Then we apply GC: for each parameter with dim > 1 (conv weights: 4D, linear weights: 2D), subtract the mean across all dims except dim 0 (output channel). This makes each output channel's gradient mean-free. BN parameters (1D) and biases (1D) are automatically skipped by the `dim() > 1` check.

  After GC, `scaler.step(optimizer)` detects that gradients are already unscaled and proceeds directly to `optimizer.step()`. If any gradient contains inf/NaN (from AMP), `scaler.step()` skips the optimizer step entirely — GC does not interfere with this safety mechanism.

## Configuration Changes
- No hyperparameter changes. GC is applied to all conv/linear weight gradients automatically via the `dim() > 1` filter.

## Execution Environment
- Method: local command `uv run python train.py > run.log 2>&1`
- Resources: single H20 GPU (existing setup)
- Estimated runtime: ~310-320s total (300s training budget + ~10-20s startup/eval)
- Log output: stdout+stderr captured to `run.log` in project root
- Tool skill: none (local execution)

## Abort Criteria
- No output in run.log after 60s — indicates crash or hang
- Loss goes to NaN/inf persistently (more than 5 consecutive steps) — indicates GC interaction with AMP causing instability
- Training crashes with CUDA error or OOM — unexpected since GC adds negligible memory (only temporary mean computation)
- Epoch count drops significantly below 99 — would indicate unexpected throughput cost from GC loop

## Verification Protocol

### Verification Procedure

**Condition 1: best_test_acc > 96.56%** (baseline 96.46% + 0.1pp threshold)
- Command: `grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- Pass: extracted value > 96.56
- Fail: value <= 96.56 or missing
- Timeout: 10s

**Condition 2: Clean completion**
- Command: `grep "^best_test_acc:" run.log` (non-empty means summary block printed)
- Pass: line exists with a numeric value
- Fail: missing or malformed
- Timeout: 10s

**Condition 3: Max 1 eval per epoch**
- Command: `grep -c "eval ep" run.log` vs `grep "^num_epochs:" run.log | awk '{print $2}'`
- Pass: eval count <= epoch count
- Fail: eval count > epoch count
- Timeout: 10s

### Informational Metrics (Optional)
- final_test_acc: `grep "^final_test_acc:" run.log | awk '{print $2}'`
- num_epochs: `grep "^num_epochs:" run.log | awk '{print $2}'`
- training_seconds: `grep "^training_seconds:" run.log | awk '{print $2}'`
- peak_vram_mb: `grep "^peak_vram_mb:" run.log | awk '{print $2}'`
