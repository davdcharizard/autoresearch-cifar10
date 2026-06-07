# Plan EXP-017: SE channel attention in BasicBlock
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-017.md

## Milestones

### Milestone 1: Code changes implemented
- [x] Add SE module class to train.py (squeeze-excitation with reduction ratio r=16)
- [x] Integrate SE into BasicBlock: apply SE after second conv+BN, before residual addition
- [x] Verify model builds and prints parameter count (4,360,010 — up from 4,301,898)
- [x] Verify torch.compile warmup succeeds with the modified architecture

### Milestone 2: Training run completes
- [ ] Run full experiment: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- [ ] Confirm training completes within 300s budget
- [ ] Confirm no crashes or NaN losses in run.log

### Milestone 3: Verification
- [ ] Extract metrics from run.log
- [ ] Compare best_test_acc against baseline (96.39%) + 0.1% threshold = 96.49%
- [ ] Collect informational metrics

## Code Changes
- **train.py**: Add an `SEBlock` module (global avg pool → FC reduce by r=16 → ReLU → FC expand → Sigmoid → scale). Integrate into `BasicBlock.forward()`: after `out = self.bn2(self.conv2(out))`, apply `out = self.se(out)` before the residual addition `out += self.shortcut(x)`. The SE module operates on the 1×1 pooled representation so compute overhead is negligible.

Specific changes:
1. Add `SEBlock(nn.Module)` class with `__init__(self, channels, reduction=16)` and `forward(self, x)`
2. In `BasicBlock.__init__`, add `self.se = SEBlock(out_channels, reduction=16)` 
3. In `BasicBlock.forward`, insert `out = self.se(out)` between `out = self.bn2(self.conv2(out))` and `out += self.shortcut(x)`

## Configuration Changes
- No hyperparameter changes. All existing settings remain the same.
- SE reduction ratio r=16 is the standard default from the original paper.

## Execution Environment
- Method: local command `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single GPU, AMP enabled
- Estimated runtime: ~5-6 minutes total (300s training + startup/compile/eval)
- Log output: stdout/stderr redirected to `run.log` in project root
- Tool skill: none (local execution)

## Abort Criteria
- Training loss diverges (NaN or increasing trend after epoch 10)
- No output in run.log after 3 minutes of execution
- CUDA OOM error (unlikely given minimal parameter increase)
- Training time exceeds 300s budget significantly (check for unexpected slowdown from SE overhead)

## Verification Protocol

### Verification Procedure

1. Run the experiment:
   ```bash
   CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
   ```

2. Check for crash:
   ```bash
   grep "^best_test_acc:" run.log
   ```
   If empty, the run crashed — check `tail -n 50 run.log` for stack trace.

3. Extract primary metric:
   ```bash
   grep "^best_test_acc:" run.log
   ```
   Parse the value. Must be >= 96.49% (baseline 96.39% + 0.1% threshold).

4. Verify training completed within budget:
   ```bash
   grep "^training_seconds:" run.log
   ```
   Must be <= 300.

5. Verify eval called at most once per epoch:
   ```bash
   grep -c "eval ep" run.log
   ```
   Count must equal the number of epochs reported in `num_epochs`.

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
