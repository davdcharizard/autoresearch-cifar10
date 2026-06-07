# Plan EXP-001: Wider ResNet (k=2) + AMP + torch.compile
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-001.md

## Milestones

### Milestone 1: Code changes implemented
- [ ] Widen channel widths from {16,32,64} to {32,64,128} in ResNet class
- [ ] Replace zero-padding skip connections with projection shortcuts (1x1 conv + BN)
- [ ] Add AMP: torch.amp.autocast('cuda') + GradScaler in training loop
- [ ] Add torch.compile(model) with warmup forward pass before training loop
- [ ] Switch SGD to Nesterov momentum
- [ ] Adjust COSINE_T_MAX to 55 (estimated epoch count for wider model)
- [ ] Verify code passes ruff linting

### Milestone 2: Experiment runs successfully
- [ ] Run `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- [ ] Confirm run completes (exit code 0, best_test_acc line present)
- [ ] Confirm training_seconds <= 300
- [ ] Check actual epoch count vs T_max (should be close)

### Milestone 3: Results verified against baseline
- [ ] Extract best_test_acc from run.log
- [ ] Compare against baseline (92.10%) — must improve by >= 0.1%
- [ ] Verify eval called at most once per epoch

## Code Changes

- **train.py — Channel widths**: Change initial conv from 3→16 to 3→32. Change _make_layer calls to use {32, 64, 128} instead of {16, 32, 64}. Change FC from Linear(64, 10) to Linear(128, 10). This quadruples the model's representational capacity.

- **train.py — Projection shortcuts**: Replace the zero-padding shortcut in BasicBlock with a proper 1x1 conv projection when dimensions change. Add a `self.shortcut` nn.Sequential containing Conv2d(in_channels, out_channels, 1, stride, bias=False) + BatchNorm2d(out_channels). This preserves more information across dimension changes than zero-padding.

- **train.py — AMP**: Wrap forward+loss computation inside `torch.amp.autocast('cuda')` context manager. Use `torch.amp.GradScaler()` to scale the loss, call `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, `scaler.update()` instead of plain `loss.backward()` + `optimizer.step()`. This halves memory bandwidth and roughly doubles throughput.

- **train.py — torch.compile**: After creating the model and moving to device, add `model = torch.compile(model)`. Add a warmup forward pass with a dummy batch BEFORE the training loop (outside the time budget) to trigger compilation and avoid compilation overhead eating into training time.

- **train.py — Nesterov SGD**: Add `nesterov=True` to the SGD constructor. Standard for WideResNets.

- **train.py — T_max**: Change COSINE_T_MAX from 90 to 55 to match the expected epoch count of the wider model. Per Protocol Finding (High Importance): T_max must match actual epoch count.

## Configuration Changes

- Channel widths: {16, 32, 64} → {32, 64, 128} (4x capacity increase)
- Shortcut connections: zero-padding → 1x1 conv projection (better information flow)
- AMP: disabled → enabled (float16 training for ~2x throughput)
- torch.compile: disabled → enabled (~15% additional speedup)
- Nesterov: False → True (standard for WideResNets)
- COSINE_T_MAX: 90 → 55 (match estimated epoch count for wider model)
- All other hyperparameters unchanged (LR=0.1, batch=128, CutOut=16, label_smoothing=0.1, warmup=5)

## Execution Environment

- Method: local command `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single NVIDIA H20 GPU (98GB VRAM); wider model ~1.3GB est.
- Estimated runtime: ~7-8 minutes total (300s training + compilation warmup + startup + eval)
- Log output: redirected to `run.log` in project root
- Tool skill: none (local execution)

## Abort Criteria

- Run exceeds 10 minutes total wall time — kill
- Python traceback or CUDA error in run.log — stop immediately
- No output after 120 seconds (compilation may take ~30-60s, allow extra time)
- Training loss NaN/inf — stop
- If epoch count is < 20 after 300s, model is too slow (note for future, don't abort)

## Verification Protocol

### Verification Procedure

1. **Check run completion**: `grep "^best_test_acc:" run.log` — FAIL if empty (run crashed)

2. **Check time budget**: `grep "^training_seconds:" run.log` — FAIL if > 300

3. **Check accuracy improvement**: `grep "^best_test_acc:" run.log` — PASS if >= 92.20% (baseline 92.10% + 0.1%). FAIL otherwise.

4. **Check eval frequency**: `grep -c "eval ep" run.log` must equal num_epochs value. FAIL if eval count > num_epochs.

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
