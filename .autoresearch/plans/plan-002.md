# Plan EXP-002: k=3 Width + Dynamic T_max + CutMix
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md

## Milestones

### Milestone 1: Code changes implemented
- [ ] Change WIDTH_MULT from 2 to 3 ({48, 96, 192})
- [ ] Implement dynamic T_max: after epoch 1, measure time per epoch, compute estimated total epochs, recreate scheduler with correct T_max
- [ ] Replace CutOut with CutMix augmentation (applied in training loop, not transforms)
- [ ] Remove COSINE_T_MAX constant (now computed dynamically)
- [ ] Verify code passes ruff linting

### Milestone 2: Experiment runs successfully
- [ ] Run experiment, confirm completion
- [ ] Check training_seconds <= 300 and epoch count is reasonable (est. 35-45)
- [ ] Verify T_max was set correctly by dynamic calibration

### Milestone 3: Results verified against baseline
- [ ] best_test_acc >= 94.13% (baseline 94.03% + 0.1%)
- [ ] Eval called at most once per epoch

## Code Changes

- **train.py — WIDTH_MULT**: Change from 2 to 3. Channels become {48, 96, 192}. ~2.4M params.

- **train.py — Dynamic T_max**: Remove the static COSINE_T_MAX. After epoch 1 completes (but before its eval), measure seconds_per_epoch = total_training_time. Estimate remaining_epochs = (TIME_BUDGET_S - total_training_time) / seconds_per_epoch. Set T_max = int(remaining_epochs) - WARMUP_EPOCHS + 1 (remaining after warmup). Create the cosine scheduler dynamically. For the first epoch, use just the warmup scheduler.

- **train.py — CutMix**: Remove the CutOut class and transform. Implement CutMix in the training loop: for each batch, with probability 0.5, apply CutMix (generate random lambda from Beta(1.0, 1.0), cut a proportional patch from a shuffled copy of the batch, mix labels). When CutMix is not applied, train normally. This replaces CutOut which was in transforms.

## Configuration Changes

- WIDTH_MULT: 2 → 3 (2.25x more capacity)
- COSINE_T_MAX: 55 → dynamic (computed from epoch 1 timing)
- Augmentation: CutOut(16) → CutMix(alpha=1.0, p=0.5)
- All other hyperparameters unchanged (LR=0.1, batch=128, label_smoothing=0.1, warmup=5)

## Execution Environment

- Method: local command `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single NVIDIA H20 GPU (98GB VRAM)
- Estimated runtime: ~8 minutes (300s training + compilation + startup + eval)
- Log output: redirected to `run.log`

## Abort Criteria

- Run exceeds 10 minutes total wall time
- Python traceback or CUDA error
- No output after 120 seconds
- Training loss NaN/inf
- Fewer than 15 epochs (model too slow for meaningful training)

## Verification Protocol

### Verification Procedure

1. `grep "^best_test_acc:" run.log` — FAIL if empty
2. `grep "^training_seconds:" run.log` — FAIL if > 300
3. best_test_acc >= 94.13% (baseline 94.03% + 0.1%) — FAIL otherwise
4. `grep -c "eval ep" run.log` == num_epochs — FAIL if eval > epochs

### Informational Metrics (Optional)

- All summary metrics: `grep "^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:" run.log`
