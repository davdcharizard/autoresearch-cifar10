# Plan EXP-000: Modern Training Recipe (Cosine LR + CutOut + Label Smoothing)
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-000.md

## Milestones

### Milestone 1: Code changes implemented
- [ ] Replace step LR scheduler with cosine annealing (epoch-based, spanning full training)
- [ ] Add linear warmup for first 5 epochs
- [ ] Add CutOut augmentation (16x16 pixel patches) to training transforms
- [ ] Replace `F.cross_entropy` with label smoothing (0.1) via `nn.CrossEntropyLoss(label_smoothing=0.1)`
- [ ] Remove MAX_STEPS cap (let time budget be the sole termination condition)
- [ ] Verify code passes `ruff` linting

### Milestone 2: Experiment runs successfully
- [ ] Run `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- [ ] Confirm run completes (exit code 0, `best_test_acc` line present in log)
- [ ] Confirm training_seconds <= 300

### Milestone 3: Results verified against baseline
- [ ] Extract best_test_acc from run.log
- [ ] Compare against baseline (91.81%) — must improve by >= 0.1%
- [ ] Verify eval called at most once per epoch (check log for eval lines vs epoch count)

## Code Changes

- **train.py**: Replace `optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32000, 48000], gamma=0.1)` with `optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=estimated_epochs)`. The step-based scheduler never reaches its second milestone (48k) within the 300s budget (only ~36.5k steps complete), so the training never gets the final low-LR fine-tuning phase. Cosine annealing smoothly decays LR over the full training duration. Step the scheduler per-epoch (after each epoch) rather than per-step. Estimate total epochs from the baseline run (~94 epochs in 300s) and use T_max=200 to allow smooth decay even if more epochs complete due to faster training.

- **train.py**: Add linear warmup for the first 5 epochs. Use `torch.optim.lr_scheduler.SequentialLR` to chain a `LinearLR(optimizer, start_factor=0.1, total_iters=5)` with the `CosineAnnealingLR`. This stabilizes early training and is standard practice.

- **train.py**: Add CutOut augmentation to the training transforms. Implement as a custom transform class that randomly masks a 16x16 square region of the image with zeros. Insert after `transforms.Normalize` in the training pipeline. CutOut is proven to add 0.5-1.0% accuracy on CIFAR-10 with small models (DeVries & Taylor 2017).

- **train.py**: Replace `F.cross_entropy(outputs, targets)` with a `nn.CrossEntropyLoss(label_smoothing=0.1)` criterion. Label smoothing prevents overconfident predictions and typically adds 0.2-0.5% accuracy.

- **train.py**: Remove `MAX_STEPS = 64000` cap and the `step >= MAX_STEPS` termination condition. The time budget (`total_training_time < TIME_BUDGET_S`) is the sole termination criterion, which is the correct behavior for time-budgeted training.

## Configuration Changes

- LR scheduler: `MultiStepLR(milestones=[32000, 48000], gamma=0.1)` -> `SequentialLR(LinearLR(start_factor=0.1, total_iters=5) + CosineAnnealingLR(T_max=200))` (fixes misaligned schedule, adds warmup)
- Loss function: `F.cross_entropy(outputs, targets)` -> `nn.CrossEntropyLoss(label_smoothing=0.1)(outputs, targets)` (adds label smoothing)
- Augmentation: add CutOut(16) after Normalize (adds regularization)
- MAX_STEPS: 64000 -> removed (time budget is sole termination)

## Execution Environment

- Method: local command `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single NVIDIA H20 GPU (98GB VRAM), model uses ~330MB
- Estimated runtime: ~6 minutes total (300s training + startup + eval overhead)
- Log output: redirected to `run.log` in project root; metrics extracted via grep
- Tool skill: none (local execution)

## Abort Criteria

- Run exceeds 10 minutes total wall time — kill and treat as failure
- `run.log` shows Python traceback or CUDA error — stop immediately
- No output in `run.log` after 60 seconds — likely hung, kill
- Training loss diverges (NaN or inf) — stop immediately

## Verification Protocol

### Verification Procedure

1. **Check run completion**: Verify exit code was 0 and `best_test_acc` line exists in `run.log`:
   ```bash
   grep "^best_test_acc:" run.log
   ```
   If empty, run crashed — check `tail -n 50 run.log`. FAIL if no metric output.

2. **Check time budget**: Extract training_seconds and verify <= 300:
   ```bash
   grep "^training_seconds:" run.log
   ```
   FAIL if training_seconds > 300.

3. **Check accuracy improvement**: Extract best_test_acc and compare to baseline (91.81%):
   ```bash
   grep "^best_test_acc:" run.log
   ```
   PASS if best_test_acc >= 91.91% (baseline 91.81% + 0.1% threshold). FAIL otherwise.

4. **Check eval frequency**: Count eval lines and verify <= num_epochs:
   ```bash
   grep -c "eval ep" run.log
   ```
   This count must equal the num_epochs value. FAIL if eval count > num_epochs.

### Informational Metrics (Optional)

- final_test_acc: `grep "^final_test_acc:" run.log`
- final_test_loss: `grep "^final_test_loss:" run.log`
- training_seconds: `grep "^training_seconds:" run.log`
- total_seconds: `grep "^total_seconds:" run.log`
- startup_seconds: `grep "^startup_seconds:" run.log`
- peak_vram_mb: `grep "^peak_vram_mb:" run.log`
- num_epochs: `grep "^num_epochs:" run.log`
- num_steps: `grep "^num_steps:" run.log`
- num_params: `grep "^num_params:" run.log`
