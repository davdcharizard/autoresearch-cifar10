# Plan EXP-019: Channels_last + T_max=49 + extended TTA (spatial shifts)
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-019.md

## Milestones

### Milestone 1: Code changes implemented
- [ ] Add `model.to(memory_format=torch.channels_last)` after model creation, before EMA deepcopy
- [ ] Add `memory_format=torch.channels_last` to training input tensor conversion
- [ ] Extend eval-mode forward() to include ±1px spatial shifts (6 total views: original, hflip, shift-left, shift-right, shift-up, shift-down)
- [ ] Keep COSINE_T_MAX = 49 (unchanged from baseline)
- [ ] Verify model builds and torch.compile warmup succeeds

### Milestone 2: Training run completes
- [ ] Run full experiment: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- [ ] Confirm training within 300s, epochs > 54 (channels_last speedup)

### Milestone 3: Verification
- [ ] Extract metrics, compare best_test_acc against 96.49% threshold

## Code Changes
- **train.py** — four changes:
  1. After `model = ResNet(...).to(device)`, add `model = model.to(memory_format=torch.channels_last)` (before EMA deepcopy so both are NHWC)
  2. In training loop, change `inputs.to(device, non_blocking=True)` to `inputs.to(device, memory_format=torch.channels_last, non_blocking=True)`
  3. In `ResNet.forward()` eval branch, replace the 2-view TTA (original + hflip) with 6-view TTA:
     - original logits
     - horizontal flip logits
     - shift left 1px: `F.pad(x[:,:,:,1:], (0,1,0,0), mode='reflect')`
     - shift right 1px: `F.pad(x[:,:,:,:-1], (1,0,0,0), mode='reflect')`
     - shift up 1px: `F.pad(x[:,:,1:,:], (0,0,0,1), mode='reflect')`
     - shift down 1px: `F.pad(x[:,:,:-1,:], (0,0,1,0), mode='reflect')`
     - Average all 6 logit vectors
  4. COSINE_T_MAX stays at 49 (no change)

## Configuration Changes
- No hyperparameter changes. T_max=49, all other settings identical to baseline.

## Execution Environment
- Method: local command `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single GPU, AMP enabled
- Estimated runtime: ~6-7 minutes total (300s training + startup/compile + eval with 6 views)
- Log output: `run.log` in project root

## Abort Criteria
- Training loss diverges (NaN or increasing after epoch 10)
- No output after 3 minutes
- CUDA OOM
- Fewer epochs than 54 (would mean channels_last is slower, contradicting EXP-018)

## Verification Protocol

### Verification Procedure

1. Run: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
2. Check crash: `grep "^best_test_acc:" run.log` — empty means crash
3. Primary metric: must be >= 96.49% (baseline 96.39% + 0.1%)
4. Training budget: `grep "^training_seconds:" run.log` — must be <= 300
5. Eval frequency: `grep -c "eval ep" run.log` must equal num_epochs

### Informational Metrics (Optional)
- final_test_acc, final_test_loss, training_seconds, total_seconds, startup_seconds, peak_vram_mb, num_epochs, num_steps, num_params — all via `grep "^{metric}:" run.log`
