# Plan EXP-009: Batch Size 256 with Linear LR Scaling
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-009.md

## Milestones

### Milestone 1: Code Changes Implemented
- [x] Change BATCH_SIZE from 128 to 256
- [x] Change LR from 0.1 to 0.2 (linear scaling: 2x batch → 2x LR)
- [x] Add 5-epoch gradual warmup to the wall-clock-fractional LR schedule lambda
- [x] Verify no syntax errors (`python -c "import train"`)

### Milestone 2: Experiment Run
- [x] Run `uv run train.py > run.log 2>&1` and capture output
- [x] Confirm training starts and outputs step logs
- [x] Confirm training completes within 300s budget

### Milestone 3: Verification
- [x] Extract best_test_acc from run.log
- [x] Verify best_test_acc > 94.92% (baseline 94.82 + 0.1pp threshold)
- [x] Verify summary block is complete
- [x] Verify eval count <= num_epochs

## Code Changes
- **train.py line 22**: `BATCH_SIZE = 128` → `BATCH_SIZE = 256` — doubles batch size to improve GPU utilization (484 MB → ~900 MB estimated, well within ~96 GB H20 capacity)
- **train.py line 23**: `LR = 0.1` → `LR = 0.2` — linear scaling rule (Goyal et al. 2017): when batch size doubles, LR doubles
- **train.py lines 172-178** (`_wall_clock_fractional_step_decay`): Add gradual warmup for the first 5 epochs. During warmup, LR ramps linearly from LR/5 to LR. After warmup completes, the existing wall-clock-fractional step decay continues unchanged. Implementation: track epoch count via a mutable cell (similar to `_lr_progress`), compute warmup fraction as `min(1, epoch/5)`, multiply the existing schedule multiplier by warmup fraction when epoch < 5.

## Configuration Changes
- BATCH_SIZE: 128 → 256 (2x increase to improve GPU utilization)
- LR: 0.1 → 0.2 (linear scaling rule: proportional to batch size increase)
- Warmup: none → 5-epoch linear warmup from LR/5=0.04 to LR=0.2 (stabilizes early training at higher LR)

## Execution Environment
- Method: local command `uv run train.py > run.log 2>&1`
- Resources: single GPU (H20, ~96 GB VRAM), ~900 MB estimated peak VRAM
- Estimated runtime: ~300s (wall-clock budget) + ~10s startup
- Log output: stdout/stderr captured to `run.log` in project root
- Tool skill: none (local execution)

## Abort Criteria
- Loss NaN/inf in first 100 steps (gradient overflow at LR=0.2 with FP16)
- No output after 60 seconds (process hang)
- OOM error (unlikely at ~900 MB but check)
- Accuracy below 80% after first epoch (severe divergence indicating warmup insufficient)

## Verification Protocol

### Verification Procedure

**Condition 1: best_test_acc > 94.92%**
```bash
grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'
```
Pass if extracted value > 94.92. Fail otherwise. Timeout: 5s.

**Condition 2: Summary block complete**
```bash
grep -c "^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:" run.log
```
Pass if count = 10 (all summary fields present). Fail otherwise. Timeout: 5s.

**Condition 3: Eval count <= num_epochs**
The training loop calls `evaluator.evaluate()` once per epoch (after the inner loop). Verify:
```bash
grep -c "eval ep" run.log
```
Compare against `grep "^num_epochs:" run.log | awk '{print $2}'`. Pass if eval count <= num_epochs. Timeout: 5s.

### Informational Metrics (Optional)
- training_seconds: `grep "^training_seconds:" run.log | awk '{print $2}'`
- peak_vram_mb: `grep "^peak_vram_mb:" run.log | awk '{print $2}'`
- final_test_acc: `grep "^final_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- final_test_loss: `grep "^final_test_loss:" run.log | awk '{print $2}'`
- num_epochs: `grep "^num_epochs:" run.log | awk '{print $2}'`
- num_steps: `grep "^num_steps:" run.log | awk '{print $2}'`
- num_params: `grep "^num_params:" run.log | awk '{print $2}'`
