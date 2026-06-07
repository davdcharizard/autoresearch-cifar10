# Plan EXP-018: Channels_last (NHWC) memory format for faster training
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-018.md

## Milestones

### Milestone 1: Code changes implemented
- [ ] Add `model.to(memory_format=torch.channels_last)` after model creation, BEFORE deepcopy for EMA (so both models are NHWC)
- [ ] Add `memory_format=torch.channels_last` to training input tensor conversion
- [ ] Update COSINE_T_MAX from 49 to 55 (to exploit expected additional epochs from speedup)
- [ ] Verify model builds, prints correct param count, and torch.compile warmup succeeds

### Milestone 2: Training run completes
- [ ] Run full experiment: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- [ ] Confirm training completes within 300s budget
- [ ] Verify speedup: check num_epochs > 54 (baseline epoch count)

### Milestone 3: Verification
- [ ] Extract metrics from run.log
- [ ] Compare best_test_acc against baseline (96.39%) + 0.1% threshold = 96.49%
- [ ] Collect informational metrics

## Code Changes
- **train.py**: Three changes:
  1. After `model = ResNet(...).to(device)` and before `ema_model = copy.deepcopy(model)`, add `model = model.to(memory_format=torch.channels_last)`. This ensures both the training model and EMA model (which is deepcopied after) use NHWC format, keeping the EMA parameter update format-consistent.
  2. In the training loop, change `inputs = inputs.to(device, non_blocking=True)` to `inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)`. This converts training inputs to NHWC to avoid per-batch format conversion overhead in cuDNN.
  3. Change `COSINE_T_MAX = 49` to `COSINE_T_MAX = 55`. With ~15% speedup, expected epochs increase from ~54 to ~62. T_max=55 (warmup 5 + cosine 55 = 60) is conservative — ensures the cosine schedule completes before the budget runs out while exploiting most of the extra epochs.

## Configuration Changes
- COSINE_T_MAX: 49 → 55 (rationale: with 15% speedup, ~62 epochs expected; T_max=55 aligns cosine completion with ~epoch 60, leaving a small margin. Even if actual speedup differs, T_max=55 vs 49 is a modest change — at worst the LR reaches minimum slightly later, which is better than too early.)

## Execution Environment
- Method: local command `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single GPU, AMP enabled
- Estimated runtime: ~5-6 minutes total (300s training + startup/compile/eval)
- Log output: stdout/stderr redirected to `run.log` in project root
- Tool skill: none (local execution)

## Abort Criteria
- Training loss diverges (NaN or increasing trend after epoch 10)
- No output in run.log after 3 minutes
- CUDA OOM error
- Fewer epochs than baseline (54) — would indicate channels_last is slower, not faster
- torch.compile error related to channels_last incompatibility

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
   If empty, check `tail -n 50 run.log` for stack trace.

3. Extract primary metric:
   ```bash
   grep "^best_test_acc:" run.log
   ```
   Must be >= 96.49% (baseline 96.39% + 0.1%).

4. Verify training within budget:
   ```bash
   grep "^training_seconds:" run.log
   ```
   Must be <= 300.

5. Verify eval called at most once per epoch:
   ```bash
   grep -c "eval ep" run.log
   ```
   Must equal num_epochs value.

### Informational Metrics (Optional)
- final_test_acc: `grep "^final_test_acc:" run.log`
- final_test_loss: `grep "^final_test_loss:" run.log`
- training_seconds: `grep "^training_seconds:" run.log`
- total_seconds: `grep "^total_seconds:" run.log`
- startup_seconds: `grep "^startup_seconds:" run.log`
- peak_vram_mb: `grep "^peak_vram_mb:" run.log`
- num_epochs: `grep "^num_epochs:" run.log` — key metric for validating speedup
- num_steps: `grep "^num_steps:" run.log`
- num_params: `grep "^num_params:" run.log`
