# Plan EXP-008: torch.compile with warmup pass
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-008.md

Hypothesis: `torch.compile(model)` reduces per-step time from ~9ms to ~7.5-8ms via graph-level kernel fusion, yielding ~91-96 epochs (vs 83). Additional LR=0.001 epochs raise best_test_acc from 94.82% to 95.0-95.2%. Threshold: >= 94.92%.

## Milestones
### Milestone 1: Code change, ruff pass
- [ ] Add `model = torch.compile(model)` after `.to(device, ...)` call (line ~148)
- [ ] Add warmup forward+backward pass before training loop (after scheduler, before `t_start_training`) to trigger compilation outside the training time budget
- [ ] Run `uv run ruff format train.py && uv run ruff check train.py`

### Milestone 2: Run to completion
- [ ] Run `uv run train.py > run.log 2>&1` and confirm full summary block printed
- [ ] Confirm `training_seconds` <= 300

### Milestone 3: Verification
- [ ] Verify `best_test_acc > 94.92%`
- [ ] Confirm summary block complete
- [ ] Confirm eval count <= num_epochs

## Code Changes
**train.py line ~148**: After `model = ResNet(...).to(device, memory_format=torch.channels_last)`, add `model = torch.compile(model)`. This wraps the model in TorchDynamo for graph-level compilation with Inductor backend.

**train.py lines ~183-189** (before `t_start_training`): Add a warmup block that runs one dummy forward+backward pass to trigger compilation. This is critical — without it, the first training step's `dt` includes ~10-30s of compilation time, eating into the 300s budget. The warmup includes:
1. Create dummy input tensor (BATCH_SIZE, 3, 32, 32) with channels_last format
2. Create dummy target tensor (BATCH_SIZE,) with random class labels
3. Run forward pass under `torch.amp.autocast`
4. Compute loss and call `scaler.scale(loss).backward()`
5. `optimizer.zero_grad()` and `scaler.update()` to reset state
6. Delete dummy tensors and call `torch.cuda.empty_cache()`

The warmup happens before `total_training_time` starts accumulating (since `total_training_time = 0.0` is set after warmup, and only per-step `dt` values inside the loop are added).

## Configuration Changes
- No hyperparameter changes. torch.compile with `mode="default"` (implicit) is a pure throughput optimization.

## Execution Environment
- Method: `uv run train.py > run.log 2>&1` (local)
- Resources: Single H20 GPU, same as EXP-007
- Estimated runtime: ~400-450s total (compilation warmup ~10-30s + 300s training + eval passes)
- Log output: stdout/stderr redirected to `run.log`
- Tool skill: N/A

## Abort Criteria
- No output to `run.log` for >120s → likely compilation hang or OOM during compile
- `RuntimeError` or `BackendCompilerFailed` in log → torch.compile incompatibility with model or AMP
- Training step time consistently >15ms after first 100 steps → compile not providing speedup, compile overhead leaked

## Verification Protocol

### Verification Procedure
Baseline: 94.82%. Threshold: 94.92% (baseline + 0.1pp).

- Condition 1: `grep "^best_test_acc:" run.log | awk '{print $2}' | awk '{val=$1+0; if (val > 94.92) print "PASS: "val; else print "FAIL: "val}'` — best_test_acc > 94.92%
- Condition 2: `grep -c "^best_test_acc:\|^final_test_acc:\|^training_seconds:\|^num_epochs:" run.log` — must be 4 (summary block complete)
- Condition 3: `eval_count=$(grep -c "eval ep" run.log); num_epochs=$(grep "^num_epochs:" run.log | awk '{print $2}'); awk "BEGIN {if ($eval_count <= $num_epochs) print \"PASS\"; else print \"FAIL\"}"` — eval_count <= num_epochs

### Informational Metrics (Optional)
- training_seconds: `grep "^training_seconds:" run.log | awk '{print $2}'`
- peak_vram_mb: `grep "^peak_vram_mb:" run.log | awk '{print $2}'`
- final_test_acc: `grep "^final_test_acc:" run.log | awk '{print $2}'`
- final_test_loss: `grep "^final_test_loss:" run.log | awk '{print $2}'`
- num_epochs: `grep "^num_epochs:" run.log | awk '{print $2}'`
- num_steps: `grep "^num_steps:" run.log | awk '{print $2}'`
- num_params: `grep "^num_params:" run.log | awk '{print $2}'`
