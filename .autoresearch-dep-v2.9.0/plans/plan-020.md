# Plan EXP-020: CosineAnnealingLR with Correct T_max
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-020.md

## Milestones

### Milestone 1: Code Changes Implemented
- [ ] Remove `_lr_progress` cell, `_epoch_count` cell, and `_wall_clock_fractional_step_decay` function (lines 189-205)
- [ ] Remove `_lr_progress[0]` update in training loop (lines 248-250)
- [ ] Remove `_epoch_count[0]` update in training loop (line 226)
- [ ] Replace `LambdaLR` scheduler with a `LambdaLR` that composes 5-epoch linear warmup with cosine annealing over estimated total steps
- [ ] Verify no syntax errors by checking `python -c "import train"`

### Milestone 2: Experiment Run
- [ ] Run `uv run python train.py` with output captured to log file
- [ ] Confirm training completes ~98 epochs in 300s (no throughput regression)

### Milestone 3: Results Verified
- [ ] Check best_test_acc > 96.01% (baseline 95.91 + 0.1pp)
- [ ] Full 10-field summary block printed
- [ ] Eval count ≤ num_epochs

## Code Changes
- **train.py (lines 189-209)**: Remove `_lr_progress`, `_epoch_count`, `WARMUP_EPOCHS`, `_wall_clock_fractional_step_decay`, and `LambdaLR` construction. Replace with a single `LambdaLR` that implements 5-epoch linear warmup (epochs 0-4) followed by cosine decay to zero (epochs 5 onwards). The schedule uses step-level granularity: estimate total steps as `ESTIMATED_EPOCHS * steps_per_epoch` where `ESTIMATED_EPOCHS = 100` (based on 98 epochs observed in EXP-019/015/009). During warmup (first `WARMUP_STEPS` steps), LR scales linearly from `1/WARMUP_EPOCHS` to 1.0. After warmup, cosine decays from 1.0 to 0.0 over the remaining steps. This is implemented as a single LambdaLR with a closure that checks the step index against `WARMUP_STEPS`.

- **train.py (line 226)**: Remove `_epoch_count[0] = epoch - 1` — no longer needed since warmup is step-based within the LambdaLR.

- **train.py (lines 248-250)**: Remove `_lr_progress[0] = ...` update — no longer needed since CosineAnnealingLR uses the scheduler's internal step count, not wall-clock progress.

## Configuration Changes
- LR schedule: wall-clock-fractional MultiStepLR (drops at 0.5/0.75 to 0.1x/0.01x) → cosine annealing from LR=0.2 to ~0 over ~100 epochs
- Final LR: ~0.002 (= 0.2 × 0.01) → ~0.0 (cosine decays to zero)
- Warmup: epoch-granularity (checked once per epoch) → step-granularity (smoother ramp over first 5 epochs)
- ESTIMATED_EPOCHS: 100 (new constant, used to compute T_max in steps)

## Execution Environment
- Method: local command `uv run python train.py`
- Resources: single H20 GPU
- Estimated runtime: ~420s total (~300s training + ~120s TTA eval overhead)
- Log output: `uv run python train.py > .autoresearch/logs/exp-020-run.log 2>&1`
- Tool skill: none (local execution)

## Abort Criteria
- No output in log file after 60s from launch
- Loss becomes NaN/inf within first 100 steps
- OOM or CUDA error in log
- Training wall-clock exceeds 350s without reaching epoch 80 (throughput regression from code change)

## Verification Protocol

### Verification Procedure

**Condition 1 — best_test_acc > 96.01%**:
```bash
grep "^best_test_acc:" .autoresearch/logs/exp-020-run.log
```
Extract the numeric value. Pass if > 96.01. Fail otherwise.

**Condition 2 — Full summary block printed**:
```bash
grep -c "^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:" .autoresearch/logs/exp-020-run.log
```
Pass if output is 10 (all fields present). Fail otherwise.

**Condition 3 — Eval count ≤ num_epochs**:
```bash
grep -c "eval ep" .autoresearch/logs/exp-020-run.log
```
Compare with num_epochs from summary block. Pass if eval_count ≤ num_epochs. Fail otherwise.

### Informational Metrics (Optional)
- training_seconds: `grep "^training_seconds:" .autoresearch/logs/exp-020-run.log`
- peak_vram_mb: `grep "^peak_vram_mb:" .autoresearch/logs/exp-020-run.log`
- final_test_acc: `grep "^final_test_acc:" .autoresearch/logs/exp-020-run.log`
- final_test_loss: `grep "^final_test_loss:" .autoresearch/logs/exp-020-run.log`
- num_epochs: `grep "^num_epochs:" .autoresearch/logs/exp-020-run.log`
- num_steps: `grep "^num_steps:" .autoresearch/logs/exp-020-run.log`
- num_params: `grep "^num_params:" .autoresearch/logs/exp-020-run.log`
