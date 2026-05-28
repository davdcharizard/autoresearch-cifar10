# Plan EXP-019: Test-Time Augmentation (Horizontal Flip)
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-019.md

## Milestones

### Milestone 1: Implement TTA evaluation function
- [ ] Add a `tta_evaluate(model, device, loader)` function in train.py that: iterates the test loader, for each batch computes logits on both original and horizontally-flipped images, averages the logits, computes accuracy and loss from the averaged logits
- [ ] Verify function signature returns `(test_loss, test_acc)` matching the existing `evaluator.evaluate()` interface

### Milestone 2: Integrate TTA into training loop
- [ ] Replace the per-epoch eval call `evaluator.evaluate(model, device)` with `tta_evaluate(model, device, evaluator.loader)` so that `best_acc` tracks TTA-based accuracy throughout training
- [ ] Verify the final summary block still prints all required fields correctly

### Milestone 3: Run experiment and verify
- [ ] Run `uv run python train.py` with output captured to log file
- [ ] Confirm training completes within 300s budget, full summary block printed, best_test_acc > 95.67%

## Code Changes
- **train.py**: Add a new `tta_evaluate()` function (~20 lines) that takes `(model, device, loader)`, iterates the test DataLoader, for each batch: (1) runs forward pass on original images, (2) flips images horizontally with `torch.flip(inputs, dims=[3])`, (3) runs forward pass on flipped images, (4) averages the two logit tensors, (5) computes cross-entropy loss and accuracy from averaged logits. Uses `@torch.inference_mode()` and `model.eval()`. Returns `(avg_loss, accuracy_pct)` matching `evaluator.evaluate()` return format.
- **train.py**: Replace `test_loss, test_acc = evaluator.evaluate(model, device)` (line 256) with `test_loss, test_acc = tta_evaluate(model, device, evaluator.loader)`. This ensures `best_acc` is computed with TTA from the first epoch onward.

## Configuration Changes
- No hyperparameter or configuration changes. Training is completely unchanged. Only evaluation is modified.

## Execution Environment
- Method: local command `uv run python train.py`
- Resources: single H20 GPU (same as baseline)
- Estimated runtime: ~310-320s total (300s training + ~10-20s eval overhead from doubled inference passes)
- Log output: `uv run python train.py 2>&1 | tee /SPXvePFS/users/david/autoresearch-cifar10/.autoresearch/logs/exp-019-run.log`
- Tool skill: none (local execution)

## Abort Criteria
- No output after 60 seconds (indicates crash or hang)
- Any Python traceback in output (code error — fix and retry)
- Training time exceeds 300s by a large margin without completing (indicates TTA is somehow counting toward training time — this should not happen but check)
- Loss NaN/inf in first 100 steps

## Verification Protocol

### Verification Procedure
After the experiment completes:

1. **Condition 1 — best_test_acc > 95.67%**: Parse `grep "^best_test_acc:" run.log` and verify the value is strictly greater than 95.67 (baseline 95.57 + 0.1pp threshold). Pass if > 95.67, fail otherwise.

2. **Condition 2 — Full summary block**: Verify all 10 summary fields are present in output: `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, `num_params`. Pass if all 10 present, fail if any missing.

3. **Condition 3 — Eval count ≤ num_epochs**: Count eval lines (`grep -c "eval ep" run.log`) and compare to `num_epochs` from summary. Pass if eval_count ≤ num_epochs, fail otherwise.

### Informational Metrics (Optional)
- training_seconds: `grep "^training_seconds:" run.log`
- peak_vram_mb: `grep "^peak_vram_mb:" run.log`
- final_test_acc: `grep "^final_test_acc:" run.log`
- final_test_loss: `grep "^final_test_loss:" run.log`
- num_epochs: `grep "^num_epochs:" run.log`
- num_steps: `grep "^num_steps:" run.log`
- num_params: `grep "^num_params:" run.log`
