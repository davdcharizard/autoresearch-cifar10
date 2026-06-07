# Plan EXP-020: Extended TTA (spatial shifts) — eval-only
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-020.md

## Milestones

### Milestone 1: Code changes implemented
- [ ] Modify eval-mode forward() to compute 6 views: original, hflip, ±1px left/right/up/down shifts
- [ ] Keep ALL training code identical (no channels_last, no T_max changes)
- [ ] Verify model builds and torch.compile warmup succeeds

### Milestone 2: Training run completes
- [ ] Run: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- [ ] Confirm 300s budget, ~54 epochs (same as baseline)

### Milestone 3: Verification
- [ ] best_test_acc >= 96.49%

## Code Changes
- **train.py**: Only change the eval branch of `ResNet.forward()`. Replace:
  ```python
  logits = self._features(x)
  logits_flip = self._features(x.flip(3))
  return (logits + logits_flip) / 2
  ```
  With 6-view TTA:
  ```python
  logits = self._features(x)
  logits_flip = self._features(x.flip(3))
  logits_sl = self._features(F.pad(x[:,:,:,1:], (0,1,0,0), mode="reflect"))
  logits_sr = self._features(F.pad(x[:,:,:,:-1], (1,0,0,0), mode="reflect"))
  logits_su = self._features(F.pad(x[:,:,1:,:], (0,0,0,1), mode="reflect"))
  logits_sd = self._features(F.pad(x[:,:,:-1,:], (0,0,1,0), mode="reflect"))
  return (logits + logits_flip + logits_sl + logits_sr + logits_su + logits_sd) / 6
  ```
  No other changes. COSINE_T_MAX=49. No channels_last.

## Configuration Changes
- None. All hyperparameters identical to baseline.

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single GPU, AMP
- Estimated runtime: ~7 minutes (300s training + longer eval with 6 views)
- Log output: `run.log`

## Abort Criteria
- Training divergence, crash, or OOM
- num_epochs != ~54 (would indicate unintended training change)

## Verification Protocol

### Verification Procedure
1. Run experiment
2. `grep "^best_test_acc:" run.log` — must be >= 96.49%
3. `grep "^training_seconds:" run.log` — must be <= 300
4. `grep -c "eval ep" run.log` must equal num_epochs

### Informational Metrics (Optional)
- All standard metrics via `grep "^{metric}:" run.log`
