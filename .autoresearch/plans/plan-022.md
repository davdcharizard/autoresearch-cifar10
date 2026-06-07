# Plan EXP-022: Gradient clipping (max_norm=5.0)
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-022.md

## Milestones

### Milestone 1: Code change
- [ ] Add `scaler.unscale_(optimizer)` after backward
- [ ] Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` after unscale
- [ ] Verify no other changes

### Milestone 2: Training completes
- [ ] Run experiment, confirm ~54 epochs, 300s budget

### Milestone 3: Verification
- [ ] best_test_acc >= 96.49%

## Code Changes
- **train.py**: In the training loop, after `scaler.scale(loss).backward()` and before `scaler.step(optimizer)`, add:
  ```python
  scaler.unscale_(optimizer)
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
  ```
  The `unscale_` call is required because AMP's GradScaler scales the loss (and hence gradients) for numerical stability. We need to unscale before clipping so the threshold is in the correct gradient magnitude range. After unscaling, `scaler.step()` will skip the optimizer step if gradients contain inf/nan (existing GradScaler behavior).

## Configuration Changes
- None. All hyperparameters identical to baseline.

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single GPU, AMP
- Estimated runtime: ~5-6 minutes
- Log output: `run.log`

## Abort Criteria
- Loss divergence, crash, OOM
- Epoch count significantly different from ~54

## Verification Protocol

### Verification Procedure
1. Run experiment
2. `grep "^best_test_acc:" run.log` — must be >= 96.49%
3. `grep "^training_seconds:" run.log` — must be <= 300
4. `grep -c "eval ep" run.log` must equal num_epochs

### Informational Metrics (Optional)
- All standard metrics via `grep "^{metric}:" run.log`
