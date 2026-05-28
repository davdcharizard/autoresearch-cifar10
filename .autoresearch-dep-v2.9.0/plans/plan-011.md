# Plan EXP-011: Squeeze-and-Excitation (SE) Blocks in BasicBlock
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-011.md

## Milestones

### Milestone 1: Implement SE module and modify BasicBlock
- [ ] Add `SEBlock` class to `train.py` (global avg pool → FC(C, C/r) → ReLU → FC(C/r, C) → Sigmoid → channel-wise scale, r=16)
- [ ] Modify `BasicBlock.__init__` to instantiate `SEBlock(out_channels, reduction=16)`
- [ ] Modify `BasicBlock.forward` to apply SE after `self.bn2(self.conv2(out))` and before residual addition
- [ ] Ensure `SEBlock` is included in `ResNet._weights_init` via the existing `self.apply(self._weights_init)` call (it already handles `nn.Linear`)
- [ ] Run syntax check: `uv run python -c "import train; print('OK')"`

### Milestone 2: Run experiment
- [ ] Execute training: `uv run train.py > run.log 2>&1`
- [ ] Confirm output is being produced (tail log after ~30s)
- [ ] Wait for completion (~6 minutes total)

### Milestone 3: Verify results
- [ ] Extract `best_test_acc` from `run.log`
- [ ] Check best_test_acc > 95.49% (baseline 95.39% + 0.1pp threshold)
- [ ] Check summary block is present
- [ ] Check number of eval lines ≤ num_epochs

## Code Changes
- **`train.py`**: Add `SEBlock(nn.Module)` class before `BasicBlock`. The SE module implements: `AdaptiveAvgPool2d(1)` → `Linear(channels, channels // reduction)` → `ReLU` → `Linear(channels // reduction, channels)` → `Sigmoid` → element-wise multiply. Reduction ratio r=16, producing bottleneck sizes of 4/8/16 for the three layer groups (channels 64/128/256). This adds ~32K parameters on top of ~4.29M (0.7% increase).
- **`train.py`**: Modify `BasicBlock.__init__` to add `self.se = SEBlock(out_channels)`.
- **`train.py`**: Modify `BasicBlock.forward` to insert `out = self.se(out)` after `out = self.bn2(self.conv2(out))` and before `out += shortcut`.

## Configuration Changes
- No hyperparameter changes. SE reduction ratio r=16 is the only new parameter, hardcoded in the SE module instantiation.

## Execution Environment
- Method: Local command `uv run train.py > run.log 2>&1`
- Resources: Single H20 GPU, ~865 MB VRAM (baseline), SE adds negligible VRAM
- Estimated runtime: ~6 minutes (300s training + ~30s startup/eval overhead)
- Log output: stdout/stderr captured to `run.log` in project root
- Tool skill: N/A (local execution)

## Abort Criteria
- Loss becomes NaN or Inf (check log for `nan` or `inf` in loss values)
- No output in `run.log` after 60 seconds from start
- OOM error (grep for `CUDA out of memory` or `OutOfMemoryError`)
- Per-step time consistently >25ms (would reduce epoch count from ~98 to <60, negating any per-epoch quality gain from SE). Check early steps in log output.

## Verification Protocol

### Verification Procedure

Baseline: 95.39% (EXP-009, commit cfe19c2). Verification threshold: > 95.49%.

**Condition 1: best_test_acc > 95.49%**
```bash
grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'
```
The extracted value must be strictly greater than 95.49. If the value is ≤ 95.49, the condition FAILS and remaining conditions are not evaluated.

**Condition 2: Training script completes and prints full summary block**
```bash
grep -c "^best_test_acc:" run.log
```
Must return exactly 1. Also verify the summary block contains all expected fields:
```bash
grep -c "^final_test_acc:" run.log && grep -c "^training_seconds:" run.log && grep -c "^num_epochs:" run.log
```
Each must return 1.

**Condition 3: Validation runs at most once per epoch**
```bash
NUM_EVALS=$(grep -c "eval ep" run.log)
NUM_EPOCHS=$(grep "^num_epochs:" run.log | awk '{print $2}')
```
NUM_EVALS must be ≤ NUM_EPOCHS.

### Informational Metrics (Optional)
- final_test_acc: `grep "^final_test_acc:" run.log`
- num_epochs: `grep "^num_epochs:" run.log`
- num_params: `grep "^num_params:" run.log` — verify SE added ~32K params over baseline ~4.29M
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — check VRAM overhead from SE
- training_seconds: `grep "^training_seconds:" run.log` — confirm full 300s budget used
- Per-step timing from early log lines — check SE overhead
