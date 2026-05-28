# Plan EXP-029: Learned 1x1 Conv Shortcut Projections
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-029.md

## Milestones

### Milestone 1: Replace zero-padding shortcuts with 1x1 conv projections
- [ ] Modify `BasicBlock.__init__` to create a `self.shortcut` module when dimensions change: `nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))`
- [ ] Modify `BasicBlock.forward` to use `self.shortcut(x)` instead of stride+padding when `self.need_pad`
- [ ] Verify the `_weights_init` method handles the new Conv2d in the shortcut (it should — it already applies Kaiming init to all Conv2d modules)
- [ ] Verify model prints correct param count (~4.34M, up from 4.29M)

### Milestone 2: Run experiment and capture output
- [ ] Run `uv run python train.py > run.log 2>&1`
- [ ] Confirm per-step time is still ~16-17ms (minimal throughput impact from 1x1 convs)
- [ ] Confirm training completes with ~96 epochs

### Milestone 3: Verify results
- [ ] Extract best_test_acc from run.log
- [ ] Check best_test_acc > 96.56%

## Code Changes
- **train.py** (BasicBlock class, lines 34-58): Modify `__init__` to create a learned shortcut when dimensions mismatch. Modify `forward` to use it. The specific changes:

  In `__init__`, replace the `need_pad` / `pad_channels` logic with:
  ```python
  self.shortcut = nn.Identity()
  if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
          nn.BatchNorm2d(out_channels),
      )
  ```

  In `forward`, replace the `if self.need_pad` block with:
  ```python
  out += self.shortcut(x)
  ```

  This replaces the stride subsampling + zero-padding with a learned 1x1 conv + BN. The identity case (same channels, stride=1) uses `nn.Identity()` for zero overhead.

## Configuration Changes
- None. Architecture modification only.

## Execution Environment
- Method: local command `uv run python train.py > run.log 2>&1`
- Resources: single H20 GPU
- Estimated runtime: ~310-320s total
- Log output: stdout+stderr captured to `run.log`
- Tool skill: none

## Abort Criteria
- Per-step time > 20ms — would indicate excessive throughput cost from the shortcut convs
- Epoch count drops below 90
- Loss goes to NaN/inf
- Training crashes

## Verification Protocol

### Verification Procedure

**Condition 1: best_test_acc > 96.56%**
- Command: `grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- Pass: > 96.56
- Fail: <= 96.56 or missing
- Timeout: 10s

**Condition 2: Clean completion**
- Command: `grep "^best_test_acc:" run.log`
- Pass: line exists
- Timeout: 10s

**Condition 3: Max 1 eval per epoch**
- Command: `grep -c "eval ep" run.log` vs `grep "^num_epochs:" run.log | awk '{print $2}'`
- Pass: eval count <= epoch count
- Timeout: 10s

### Informational Metrics (Optional)
- num_epochs, training_seconds, peak_vram_mb, num_params
