# EXP-009: Batch Size 256 with Linear LR Scaling

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-009.md
- **Plan**: plans/plan-009.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-009
- **Commit**: cfe19c2
- **PR**: (failed — token permissions; user may create manually from autoresearch/exp-009 → main)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented all three code changes to `train.py` per plan-009. Changed BATCH_SIZE from 128 to 256 and LR from 0.1 to 0.2 (linear scaling rule). Added a 5-epoch gradual warmup mechanism to the existing wall-clock-fractional LR schedule by introducing a `_epoch_count` mutable cell (mirroring the existing `_lr_progress` pattern) and a `WARMUP_EPOCHS = 5` constant. The warmup multiplier `(epoch + 1) / WARMUP_EPOCHS` is applied multiplicatively to the step-decay schedule multiplier during the first 5 epochs, ramping LR from 0.04 to 0.2. Added `_epoch_count[0] = epoch - 1` after the `epoch += 1` line in the training loop to drive the warmup state.

### Surprises & Discoveries

No surprises — all changes were straightforward and followed the plan exactly. The existing `_lr_progress` mutable cell pattern made the warmup addition clean.

### Decisions

Used `epoch - 1` (0-indexed) for `_epoch_count[0]` so that epoch 1 gets warmup factor `(0 + 1) / 5 = 0.2`, epoch 5 gets `(4 + 1) / 5 = 1.0`. This means the warmup completes at the START of epoch 5, matching the "5-epoch warmup" specification.

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
- Running `uv run train.py > run.log 2>&1` locally on a single H20 GPU. This is the EXP-009 experiment: ResNet-20 with WIDTH_MULT=4, batch size 256 (doubled from 128), LR 0.2 (doubled from 0.1), with 5-epoch linear warmup. We expect ~30-50% throughput improvement over the baseline's 83 epochs, targeting ~108-120 epochs in 300s. The wall-clock-fractional LR schedule drops at 50%/75% of the time budget. We expect best_test_acc >= 95.0%.

Observations:
- Training started cleanly, ~16ms/step, ~16,300-16,500 img/s throughput with batch size 256
- 5-epoch warmup executed correctly: LR ramped from 0.04 to 0.2 over first 5 epochs
- Completed 98 epochs in 300.0s (vs ~83 epochs at batch 128 — 18% more epochs)
- LR cascade: 0.2 → 0.02 at ~50% (150s) → 0.002 at ~75% (225s)
- Accuracy trajectory: warmup → plateau ~84% → first LR drop → jump to ~93% → plateau → second LR drop → jump to ~95.4%
- Best accuracy 95.39% at epoch 96, final 95.29% at epoch 98
- Peak VRAM 864.6 MB — well within H20 capacity

Key Metrics:
- best_test_acc: 95.39%
- final_test_acc: 95.29%
- final_test_loss: 0.1419
- training_seconds: 300.0
- total_seconds: 410.9
- startup_seconds: 1.2
- peak_vram_mb: 864.6
- num_epochs: 98
- num_steps: 19014
- num_params: 4,286,026

## Verification Results

### Conditions Checked

**Condition 1: best_test_acc > 94.92%**
- Command: `grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- Result: 95.39
- Pass/Fail: **PASS** (95.39 > 94.92)
- Source: run.log final summary block

**Condition 2: Summary block complete (10 fields)**
- Command: `grep -c "^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:" run.log`
- Result: 10
- Pass/Fail: **PASS** (10 = 10)
- Source: run.log final summary block

**Condition 3: Eval count <= num_epochs**
- Command: `grep -c "eval ep" run.log` vs `grep "^num_epochs:" run.log | awk '{print $2}'`
- Result: eval_count=98, num_epochs=98
- Pass/Fail: **PASS** (98 <= 98)
- Source: run.log

### Informational Metrics

- training_seconds: 300.0
- peak_vram_mb: 864.6
- final_test_acc: 95.29%
- final_test_loss: 0.1419
- num_epochs: 98
- num_steps: 19014
- num_params: 4,286,026

## Errors & Dead Ends

## Human Notes

> (autopilot session — no human notes)
