# EXP-011: Squeeze-and-Excitation (SE) Blocks in BasicBlock

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-011.md
- **Plan**: plans/plan-011.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-011
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented SE blocks following the plan's three milestones. Added `SEBlock(nn.Module)` class before `BasicBlock` in `train.py` — a lightweight channel attention module: global average pooling via `x.mean(dim=(2,3))`, two FC layers with reduction ratio r=16, ReLU/Sigmoid activations, and channel-wise rescaling. Modified `BasicBlock.__init__` to instantiate `self.se = SEBlock(out_channels)` and `BasicBlock.forward` to apply `out = self.se(out)` after `self.bn2(self.conv2(out))` and before the residual addition. The existing `ResNet._weights_init` already handles `nn.Linear` with kaiming_normal_, so SE's FC layers get proper initialization automatically. Syntax check confirmed OK.

### Surprises & Discoveries

No surprises — the implementation was straightforward. The SE module's FC layers use `bias=False` following the original Hu et al. implementation (the bias is unnecessary since the sigmoid output is used as a multiplicative gate, and the preceding mean-pooling already removes spatial bias).

### Decisions

Used `x.mean(dim=(2, 3))` instead of `F.adaptive_avg_pool2d(x, 1).squeeze()` for the global average pooling in SE — functionally identical but avoids the overhead of creating a view and squeezing. Both produce a (B, C) tensor from (B, C, H, W).

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27
- **Ended**: 2026-05-27

Description:
- Running `uv run train.py > run.log 2>&1` locally on H20 GPU. This trains the WIDTH_MULT=4 ResNet-20 with SE blocks (reduction r=16) added to each BasicBlock for 300s. We expect ~98 epochs to complete with negligible per-step overhead from SE (<1ms). The hypothesis is that SE's channel attention will improve best_test_acc from 95.39% baseline to ~95.6-95.9%.

Observations:
- Training started successfully, loss decreasing normally: 2.03 → 1.48 over first 350 steps (source: run.log L1-8)
- Per-step time ~18-19ms — significantly higher than baseline ~9-10ms. SE overhead is ~9ms/step, much more than the plan's <1ms estimate. This halves throughput to ~13.9K img/s and will reduce epoch count from ~98 to ~50. (source: run.log step 00050-00350)
- LR warmup working correctly: 0.04 at ep1, 0.08 at ep2, matching 5-epoch linear warmup from LR/5 (source: run.log)
- Epoch 1 eval: test_acc 46.80% — reasonable for first epoch of a wide model with heavy augmentation (source: run.log)
- No NaN/Inf/OOM errors detected

Key Metrics:
- best_test_acc: 95.45%
- final_test_acc: 95.31%
- final_test_loss: 0.1400
- training_seconds: 300.0
- total_seconds: 396.0
- peak_vram_mb: 1034.4
- num_epochs: 83
- num_steps: 16086
- num_params: 4,318,282
- Per-step time: ~18-19ms (baseline ~9-10ms)
- Throughput: ~13.9K img/s (baseline ~26K img/s)

## Verification Results

### Conditions Checked

**Condition 1: best_test_acc > 95.49%** — **FAILED**
- Command: `grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- Actual: 95.45
- Required: > 95.49
- Result: 95.45 ≤ 95.49 → FAIL
- Source: run.log summary block
- Per plan: "remaining conditions are not evaluated" — stopping here.

(Conditions 2 and 3 not evaluated due to Condition 1 failure. For the record: both would have passed — summary block present, evals=83 ≤ epochs=83.)

### Informational Metrics

- final_test_acc: 95.31%
- num_epochs: 83 (baseline ~98 — 15% fewer due to SE overhead)
- num_params: 4,318,282 (baseline ~4,290,000 — +28K from SE, 0.7% increase)
- peak_vram_mb: 1034.4 (baseline ~865 — +169MB from SE FC layers)
- training_seconds: 300.0 (full budget used)
- Per-step time: ~18-19ms (baseline ~9-10ms — SE overhead ~9ms/step)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
