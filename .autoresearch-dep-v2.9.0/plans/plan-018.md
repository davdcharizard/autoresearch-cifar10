# Plan EXP-018: Stochastic Depth (DropPath) on BasicBlock
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-018.md

## Milestones

### Milestone 1: Code changes implemented
- [x] Add `DROP_PATH_RATE` hyperparameter (0.1) at top of train.py
- [x] Add `drop_path_rate` parameter to `BasicBlock.__init__()` and store it
- [x] Modify `BasicBlock.forward()` to apply stochastic depth: during training, randomly drop the residual branch with probability `drop_path_rate`, scaling by `1 / (1 - drop_path_rate)`; during eval, identity (no change)
- [x] Modify `ResNet._make_layer()` to accept and distribute per-block drop rates
- [x] Modify `ResNet.__init__()` to compute linearly spaced drop rates from 0.0 to `DROP_PATH_RATE` across all 9 blocks and pass them to `_make_layer()`
- [x] Verify no syntax errors: `python -c "import train"`

### Milestone 2: Experiment run and results collected
- [ ] Run `python train.py 2>&1 | tee exp-018.log` and confirm training starts
- [ ] Training completes within 300s budget and prints full summary block
- [ ] Record best_test_acc from output

## Code Changes

- **train.py — Hyperparameter section (after line 25)**: Add `DROP_PATH_RATE = 0.1` — maximum drop probability at the deepest block. With p_L=0.9, this means survival probability linearly decays from 1.0 (first block) to 0.9 (last block). Mild enough to avoid over-regularization per Huang et al. 2016 guidance for shallow (9-block) nets.

- **train.py — BasicBlock.__init__()**: Add `drop_path_rate=0.0` parameter. Store as `self.drop_path_rate`.

- **train.py — BasicBlock.forward()**: After computing `out = self.bn2(self.conv2(out))` and before `out += shortcut`, apply stochastic depth during training:
  ```python
  if self.training and self.drop_path_rate > 0.0:
      keep_prob = 1.0 - self.drop_path_rate
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # per-sample mask
      random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
      random_tensor = torch.floor(random_tensor + keep_prob)
      out = out * random_tensor / keep_prob
  ```
  This is the standard DropPath implementation: per-sample Bernoulli mask on the residual branch, scaled by 1/keep_prob to preserve expected value. During eval, `self.training` is False so no drop occurs — deterministic output.

- **train.py — ResNet.__init__()**: Compute linearly spaced drop rates for all 9 blocks. Block indices 0-8, drop rates from `DROP_PATH_RATE / (9 * num_blocks) * (i+1)` simplified to `DROP_PATH_RATE * (i+1) / total_blocks` where total_blocks = 9. Pass the appropriate slice of rates to each `_make_layer()` call.
  ```python
  total_blocks = num_blocks * 3  # 9 blocks total
  drop_rates = [DROP_PATH_RATE * (i + 1) / total_blocks for i in range(total_blocks)]
  # layer1 gets drop_rates[0:3], layer2 gets [3:6], layer3 gets [6:9]
  ```

- **train.py — ResNet._make_layer()**: Accept `drop_rates` list parameter (one rate per block in this layer). Pass each rate to the corresponding `BasicBlock()` constructor.

## Configuration Changes
- `DROP_PATH_RATE`: N/A → 0.1 (maximum drop probability at deepest block; survival prob linearly decays from 1.0 to 0.9 across 9 blocks per Huang et al. 2016)

## Execution Environment
- Method: local command `python train.py 2>&1 | tee exp-018.log`
- Resources: single H20 GPU
- Estimated runtime: ~7 minutes total (300s training + ~120s startup/eval overhead)
- Log output: stdout+stderr captured to `exp-018.log` via tee
- Tool skill: N/A (local execution)

## Abort Criteria
- Loss becomes NaN or inf within first 50 steps
- No output after 60 seconds of training start
- OOM error (unlikely — no parameter count increase, minor memory for random tensors)
- Training throughput degrades >10% (>18ms/step sustained) indicating unexpected overhead from stochastic depth — would reduce epoch count and negate any regularization benefit

## Verification Protocol

### Verification Procedure

**Condition 1: best_test_acc > 95.67%** (baseline 95.57 + 0.1pp threshold)
- Command: `grep "best_test_acc:" exp-018.log | tail -1`
- Pass: extracted value > 95.67
- Fail: value ≤ 95.67

**Condition 2: Full summary block printed**
- Command: `grep -c "best_test_acc:\|final_test_acc:\|final_test_loss:\|training_seconds:\|total_seconds:\|startup_seconds:\|peak_vram_mb:\|num_epochs:\|num_steps:\|num_params:" exp-018.log`
- Pass: count = 10 (all 10 summary fields present)
- Fail: count < 10

**Condition 3: Evaluation runs at most once per epoch**
- Command: `grep -c "eval ep" exp-018.log` and `grep "num_epochs:" exp-018.log | awk '{print $2}'`
- Pass: eval count ≤ num_epochs
- Fail: eval count > num_epochs

### Informational Metrics (Optional)
- final_test_acc: `grep "final_test_acc:" exp-018.log`
- training_seconds: `grep "training_seconds:" exp-018.log`
- num_epochs: `grep "num_epochs:" exp-018.log`
- peak_vram_mb: `grep "peak_vram_mb:" exp-018.log`
- num_params: `grep "num_params:" exp-018.log`
