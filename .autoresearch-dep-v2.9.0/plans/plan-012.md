# Plan EXP-012: Conv1x1-based SE Blocks (channels_last-safe)
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-012.md

## Milestones

### Milestone 1: Implement Conv1x1-based SE module and integrate into BasicBlock
- [ ] Add `SEBlock` class to `train.py` using `nn.Conv2d(kernel_size=1)` instead of `nn.Linear`
- [ ] Add `self.se = SEBlock(out_channels)` to `BasicBlock.__init__`
- [ ] Insert `out = self.se(out)` in `BasicBlock.forward` after `self.bn2(self.conv2(out))` and before residual addition
- [ ] Verify no syntax errors: `python -c "import train"`

### Milestone 2: Run experiment and capture output
- [ ] Execute `uv run train.py > run.log 2>&1` (300s budget, local H20 GPU)
- [ ] Confirm training completes and summary block is printed in run.log

### Milestone 3: Verify results
- [ ] Extract `best_test_acc` from run.log and check > 95.49%
- [ ] Confirm summary block present and validation ran at most once per epoch

## Code Changes
- **train.py**: Add `SEBlock(nn.Module)` class between `BasicBlock` and `ResNet` classes. The SE module uses `nn.Conv2d(C, C//16, 1)` → ReLU → `nn.Conv2d(C//16, C, 1)` → Sigmoid for the channel attention path, operating on (B, C, 1, 1) tensors from global average pooling. This preserves the channels_last memory format throughout — the key fix versus EXP-011's `nn.Linear` which broke channels_last and caused ~9ms/step overhead.

  **Difference from EXP-011 (Medium importance failed approach)**: EXP-011 used `nn.Linear` layers in SE, which broke the channels_last format on H20 under AMP, doubling per-step time from ~9ms to ~18ms. This plan replaces `nn.Linear` with `nn.Conv2d(kernel_size=1)`, which operates natively on NCHW/channels_last tensors. The mathematical computation is identical — only the tensor layout changes. The pooled tensor is kept as (B, C, 1, 1) rather than squeezed to (B, C), so Conv2d operates directly without format conversion.

  Specific changes to `train.py`:
  1. **New class `SEBlock`** (insert after `BasicBlock`, before `ResNet`):
     - `__init__(self, channels, reduction=16)`: `self.conv1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)` and `self.conv2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)`
     - `forward(self, x)`: `s = x.mean(dim=(2, 3), keepdim=True)` → `s = F.relu(self.conv1(s))` → `s = torch.sigmoid(self.conv2(s))` → `return x * s`
     - Using `keepdim=True` on the mean preserves the (B, C, 1, 1) shape needed by Conv2d, avoiding any reshape/view that could trigger format conversion
     - `bias=False` following Hu et al. convention and keeping param count minimal
  2. **BasicBlock.__init__**: Add `self.se = SEBlock(out_channels)` after `self.bn2`
  3. **BasicBlock.forward**: Add `out = self.se(out)` after `out = self.bn2(self.conv2(out))` and before the shortcut addition
  4. **No changes to ResNet, hyperparameters, training loop, or anything else**

## Configuration Changes
- None. All hyperparameters remain identical to baseline (EXP-009).

## Execution Environment
- Method: Local command `uv run train.py > run.log 2>&1`
- Resources: Single H20 GPU, ~1-1.5 GB VRAM expected
- Estimated runtime: ~310-320s total (300s training + ~10-15s startup/eval)
- Log output: stdout+stderr captured to `run.log` in project root
- Tool skill: None (local execution)

## Abort Criteria
- No output in run.log after 60 seconds from launch (likely crash or hang)
- Loss is NaN or Inf in the first 500 steps (divergence)
- Per-step time consistently >15ms in early steps (would indicate Conv2d(1x1) still has format overhead — the whole point of the experiment is to keep overhead <1ms)
- OOM error (unlikely given EXP-011 was only 1034MB with nn.Linear SE)

## Verification Protocol

### Verification Procedure

Baseline: 95.39% (EXP-009, commit cfe19c2). Threshold: best_test_acc > 95.49%.

**Condition 1: best_test_acc > 95.49%**
- Command: `grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- Pass: extracted value is strictly greater than 95.49
- Fail: value is ≤ 95.49 or not found
- Timeout: 10 seconds (grep on local file)

**Condition 2: Summary block present**
- Command: `grep -c "^best_test_acc:" run.log`
- Pass: count is exactly 1
- Fail: count is 0 (training crashed before summary)

**Condition 3: Validation ran at most once per epoch**
- Command: `grep -c "eval ep" run.log` for eval count, `grep "^num_epochs:" run.log | awk '{print $2}'` for epoch count
- Pass: eval count ≤ epoch count
- Fail: eval count > epoch count

### Informational Metrics (Optional)
- `final_test_acc`: `grep "^final_test_acc:" run.log | awk '{print $2}'`
- `num_epochs`: `grep "^num_epochs:" run.log | awk '{print $2}'`
- `num_steps`: `grep "^num_steps:" run.log | awk '{print $2}'`
- `peak_vram_mb`: `grep "^peak_vram_mb:" run.log | awk '{print $2}'`
- `training_seconds`: `grep "^training_seconds:" run.log | awk '{print $2}'`
- `num_params`: `grep "^num_params:" run.log | awk '{print $2}'`
- Per-step time: `grep "step 00050" run.log | grep -oP 'dt: \K[0-9]+'` (early step timing to confirm overhead)
