# EXP-000: Budget-matched modern training recipe (same ResNet-20)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-000.md
- **Plan**: plans/plan-000.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-000
- **Commit**: be45820 (merged fast-forward into autoresearch/dev)
- **PR**: N/A — no git remote configured (intentional per TASK.md; local-only record)
- **Outcome**: completed

## Implementation Notes

### Summary
All changes confined to train.py per plan. Replaced MultiStepLR with a module-level `lr_at(progress)` one-cycle function keyed to `total_training_time / TIME_BUDGET_S` (15% linear warmup to PEAK_LR=0.4, then cosine to ~0), with the LR written to all param groups each step inside the timed region. Switched to BATCH_SIZE=512, SGD nesterov with two param groups (weight_decay 5e-4 on ndim>1 params, 0 on BN/bias), label_smoothing=0.1, bf16 autocast around forward+loss, TF32 via `set_float32_matmul_precision("high")`, `cudnn.benchmark=True`, channels_last for model and input batches, `persistent_workers=True` on the DataLoader, MAX_STEPS raised to 1,000,000. Eval cadence (once per epoch), seed 42, summary print block, and per-step synchronize timing untouched. Static checks: ast.parse OK, ruff clean, `git status` shows only train.py modified.

### Surprises & Discoveries
- None during implementation — the file structure matched the plan's expectations exactly.

### Decisions
- **Branch name**: plan said `exp/000-budget-matched-recipe`, but the execute skill mandates the `autoresearch/exp-{NNN}` format — used `autoresearch/exp-000`.
- **Optimizer base LR set to 0.0**: the per-step `lr_at()` assignment is the sole LR authority; a nonzero constructor value would only ever apply to step 1 before being overwritten, so 0.0 makes the control flow unambiguous (step 1 trains at ~0 LR, the start of warmup).
- **LR update placed inside the timed region** (after `t0`): it costs microseconds and keeps the budget metering honest.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash task bicnrzm85 (local)
- **Log file(s)**: run.log (project root)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 03:43
- **Ended**: 2026-06-10 03:53

Description:
- Full training run of the modernized recipe on the unchanged ResNet-20: budget-matched one-cycle LR (peak 0.4, 15% warmup), batch 512, bf16 autocast + TF32 + channels_last, nesterov SGD with selective weight decay, label smoothing 0.1. Expectation: throughput rises from ~16k img/s (baseline) to ≥25k img/s, epochs from 97 to ≥150, and best_test_acc from 91.97% to ≥92.5% because the LR anneal now completes within the 300s budget.

Observations:
- Epoch-1 eval healthy: test_acc 43.98%, test_loss 1.5211 — well above the 15% divergence-abort threshold (source: run.log, first `eval ep` line)
- Throughput ~60k img/s at dt 8-9ms/step with batch 512 (~3.8x the baseline's ~16k img/s); LR warmup tracking the schedule (lr 0.072 at 2.7% progress = 0.4 x 0.027/0.15) (source: run.log step 00100-00150 lines)
- Watch item: at this throughput ~300+ epochs fit in the budget; per-epoch eval (~0.7s, outside the training budget but inside wall clock) may push total_seconds toward the 600s cap (abort criterion)

Key Metrics:
- best_test_acc: 93.16% (source: run.log summary block)
- final_test_acc: 93.01% | final_test_loss: 0.2998 (source: run.log summary block)
- training_seconds: 300.0 | total_seconds: 596.7 | startup_seconds: 1.1 (source: run.log summary block)
- peak_vram_mb: 479.7 | num_epochs: 345 | num_steps: 33,463 | num_params: 269,722 (source: run.log summary block)

## Verification Results

### Conditions Checked

1. **Run completes without crashing within the time budget (≤ 10 min total)** — PASS
   - `best_test_acc:` line present in run.log summary; total_seconds = 596.7 ≤ 600; process exit code 0 (source: run.log summary block; background task bicnrzm85)
   - Note: only 3.3s of margin — 345 per-epoch evals (~0.85s each) consumed ~295s of wall clock outside the training budget.
2. **best_test_acc exceeds baseline by ≥ 0.1 pp (≥ 92.07)** — PASS
   - best_test_acc = 93.16% vs baseline 91.97% → +1.19 pp (source: run.log summary; baseline from exp-index.sh)
3. **Validation executed at most once per epoch** — PASS
   - `grep -c "eval ep" run.log` = 345 = num_epochs (one eval per epoch; loop structure unchanged) (source: run.log)

### Informational Metrics

- peak_vram_mb: 479.7 (baseline 330.1 — +45%, trivially within the 98GB soft constraint)
- num_epochs: 345 (baseline 97 — 3.6x throughput from bf16 + channels_last + batch 512 + TF32)
- num_params: 269,722 (identical to baseline — architecture unchanged)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
