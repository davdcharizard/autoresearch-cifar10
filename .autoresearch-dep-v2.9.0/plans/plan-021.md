# Plan EXP-021: Pre-activation ResNet Blocks (BN-ReLU-Conv)
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-021.md

## Milestones

### Milestone 1: Implement pre-activation BasicBlock
- [ ] Rewrite `BasicBlock.__init__` — change `bn1` to `BatchNorm2d(in_channels)` (BN applied before conv1 on the input) and keep `bn2` as `BatchNorm2d(out_channels)` (BN applied before conv2 on intermediate features)
- [ ] Rewrite `BasicBlock.forward` — new ordering: `bn1(x) → relu → conv1 → bn2 → relu → conv2`, shortcut taken from raw input `x` (before BN), no final relu after residual addition
- [ ] Verify shortcut path: stride-based downsampling and channel padding still operate on raw `x` (unchanged logic)

### Milestone 2: Update ResNet stem and final layers for pre-activation
- [ ] Remove `self.bn1 = nn.BatchNorm2d(16 * WIDTH_MULT)` from `ResNet.__init__` — no longer needed because the first BasicBlock's `bn1` handles normalization of the stem output
- [ ] Add `self.bn_final = nn.BatchNorm2d(64 * WIDTH_MULT)` in `ResNet.__init__` — the last block's output has no trailing BN or ReLU in pre-activation ordering, so a final BN+ReLU is needed before pooling
- [ ] Update `ResNet.forward` — stem becomes `conv1(x)` (no bn1/relu), ending becomes `bn_final → relu → pool → fc`
- [ ] Confirm `_weights_init` still applies correctly (Conv2d, Linear, BatchNorm2d — no change needed)

### Milestone 3: Run experiment and capture output
- [ ] Run training with `uv run python train.py 2>&1 | tee .autoresearch/logs/exp-021-run.log`
- [ ] Confirm training starts, completes ~99 epochs in 300s, and prints full 10-field summary

## Code Changes
- **train.py (BasicBlock.__init__, lines 35-48)**: Change `bn1` from `BatchNorm2d(out_channels)` to `BatchNorm2d(in_channels)`. `bn2` remains `BatchNorm2d(out_channels)`. This is because in pre-activation ordering, BN is applied to the input before the convolution, so bn1 must match the input channel count.
- **train.py (BasicBlock.forward, lines 50-58)**: Replace post-activation forward `relu(bn1(conv1(x))) → bn2(conv2(...)) → add shortcut → relu(out)` with pre-activation forward `relu(bn1(x)) → conv1 → relu(bn2(...)) → conv2 → add shortcut`. The shortcut is taken from raw `x` (before BN/ReLU). No final relu — the identity mapping is clean.
- **train.py (ResNet.__init__, lines 61-76)**: Remove `self.bn1 = nn.BatchNorm2d(16 * WIDTH_MULT)`. Add `self.bn_final = nn.BatchNorm2d(64 * WIDTH_MULT)` after layer3 definition (before fc). The stem outputs raw conv features; the first block's bn1 normalizes them. The final bn_final+relu activates the last block's output before pooling.
- **train.py (ResNet.forward, lines 94-101)**: Change `F.relu(self.bn1(self.conv1(x)))` to just `self.conv1(x)`. Change the ending from `adaptive_avg_pool2d(out, 1)` to `F.relu(self.bn_final(out)) → adaptive_avg_pool2d → fc`.

## Configuration Changes
- None. ESTIMATED_EPOCHS remains 100, all hyperparameters unchanged. Pre-activation blocks have identical parameter count and FLOPs — zero throughput cost.

## Execution Environment
- Method: local command
- Resources: single GPU (H20), AMP (FP16), batch 256
- Estimated runtime: ~408s total (300s training + ~108s TTA evaluation), ~99 epochs
- Log output: `uv run python train.py 2>&1 | tee .autoresearch/logs/exp-021-run.log`
- Tool skill: N/A

## Abort Criteria
- No log output after 60 seconds of starting the command
- Loss becomes NaN/inf in the first 500 steps
- OOM or CUDA error in log output
- Per-step time exceeds 25ms consistently (would indicate unexpected overhead, baseline is ~16ms)

## Verification Protocol

### Verification Procedure

Baseline: 96.46% (EXP-020, queried via `exp-index.sh baseline`)

**Condition 1: best_test_acc > 96.56%** (baseline 96.46% + 0.1pp)
```bash
grep '^best_test_acc:' .autoresearch/logs/exp-021-run.log
```
Extract the numeric value. PASS if > 96.56. FAIL otherwise. Timeout: 5s.

**Condition 2: Full 10-field summary block printed**
```bash
grep -c -E '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' .autoresearch/logs/exp-021-run.log
```
PASS if output is 10. FAIL otherwise. Timeout: 5s.

**Condition 3: Eval count ≤ num_epochs**
```bash
EVAL_COUNT=$(grep -c 'eval ep' .autoresearch/logs/exp-021-run.log)
NUM_EPOCHS=$(grep '^num_epochs:' .autoresearch/logs/exp-021-run.log | awk '{print $2}')
```
PASS if EVAL_COUNT ≤ NUM_EPOCHS. FAIL otherwise. Timeout: 5s.

### Informational Metrics (Optional)
- training_seconds: `grep '^training_seconds:' .autoresearch/logs/exp-021-run.log`
- peak_vram_mb: `grep '^peak_vram_mb:' .autoresearch/logs/exp-021-run.log`
- final_test_acc: `grep '^final_test_acc:' .autoresearch/logs/exp-021-run.log`
- final_test_loss: `grep '^final_test_loss:' .autoresearch/logs/exp-021-run.log`
- num_epochs: `grep '^num_epochs:' .autoresearch/logs/exp-021-run.log`
- num_steps: `grep '^num_steps:' .autoresearch/logs/exp-021-run.log`
- num_params: `grep '^num_params:' .autoresearch/logs/exp-021-run.log`
