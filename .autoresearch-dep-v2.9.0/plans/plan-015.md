# Plan EXP-015: Label Smoothing 0.2 (Standalone)
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-015.md

## Milestones

### Milestone 1: Code change implemented
- [ ] Add `label_smoothing=0.2` parameter to the `F.cross_entropy(outputs, targets)` call in train.py (line ~220)
- [ ] Verify no other changes — no Nesterov, no WD change, no augmentation change

### Milestone 2: Experiment submitted and confirmed running
- [ ] Run `uv run train.py 2>&1 | tee /SPXvePFS/users/david/autoresearch-cifar10/run-015.log`
- [ ] Confirm first epoch completes and loss values are reasonable (not NaN/inf)

### Milestone 3: Training completed and metrics extracted
- [ ] Training completes within 300s budget
- [ ] Extract best_test_acc from log output
- [ ] Verify best_test_acc > 95.49% (baseline 95.39% + 0.1pp threshold)

## Code Changes
- **train.py line ~220**: Change `loss = F.cross_entropy(outputs, targets)` to `loss = F.cross_entropy(outputs, targets, label_smoothing=0.2)`. This is the sole change — adds output-distribution regularization via soft targets. The `label_smoothing` kwarg is natively supported by PyTorch's `F.cross_entropy` with zero computational overhead.

## Configuration Changes
- label_smoothing: 0.0 -> 0.2 (validated by hlb-CIFAR10; higher than EXP-004's 0.1 which was confounded by Nesterov overhead)

## Execution Environment
- Method: local command (`uv run train.py`)
- Resources: single GPU (H20), ~4GB VRAM
- Estimated runtime: ~310-320s total (300s training budget + startup/eval overhead)
- Log output: `uv run train.py 2>&1 | tee /SPXvePFS/users/david/autoresearch-cifar10/run-015.log`
- Tool skill: none (local execution)

## Abort Criteria
- Loss NaN or inf in first 100 steps (label smoothing should not cause divergence, but check)
- No output after 60 seconds
- Training loss increasing monotonically after epoch 5 (underfitting signal)

## Verification Protocol

### Verification Procedure
After training completes, run verification in order — abort on first failure:

1. **Condition: best_test_acc > 95.49%** (baseline 95.39% + 0.1pp)
   - Command: `grep "^best_test_acc:" run-015.log`
   - Parse: extract numeric value after colon, strip `%`
   - Pass: value > 95.49
   - Fail: value <= 95.49

2. **Condition: training script completes with full summary block**
   - Command: `grep -c "^best_test_acc:\|^final_test_acc:\|^training_seconds:\|^peak_vram_mb:" run-015.log`
   - Pass: count >= 4 (all summary fields present)
   - Fail: count < 4

3. **Condition: validation runs at most once per epoch**
   - Command: `grep -c "eval ep" run-015.log`
   - Compare against: `grep "^num_epochs:" run-015.log` (extract number)
   - Pass: eval count <= num_epochs
   - Fail: eval count > num_epochs

### Informational Metrics (Optional)
- training_seconds: `grep "^training_seconds:" run-015.log`
- peak_vram_mb: `grep "^peak_vram_mb:" run-015.log`
- final_test_acc: `grep "^final_test_acc:" run-015.log`
- final_test_loss: `grep "^final_test_loss:" run-015.log`
- num_epochs: `grep "^num_epochs:" run-015.log`
- num_steps: `grep "^num_steps:" run-015.log`
- num_params: `grep "^num_params:" run-015.log`
