# EXP-001: Widen ResNet-20 4x (WRN-style) on the validated recipe

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-001.md
- **Plan**: plans/plan-001.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-001
- **Commit**: bd0976e (merged fast-forward into autoresearch/dev)
- **PR**: N/A — no git remote configured (intentional per TASK.md; local-only record)
- **Outcome**: completed

## Implementation Notes

### Summary
Single-variable capacity change per plan: added `WIDTH_MULT = 4` hyperparameter; `ResNet.__init__` now takes `width_mult` and scales stage widths (16,32,64)→(64,128,256) including conv1/bn1 and the fc in-features; params print line includes the multiplier. The `BasicBlock` zero-pad shortcut needed no change (pads `out_channels - in_channels` generically). Entire EXP-000 recipe untouched (time-keyed one-cycle peak 0.4, bf16/TF32/channels_last, batch 512 nesterov, selective WD, label smoothing, eval once/epoch, seed 42). Ruff clean; only train.py modified.

### Surprises & Discoveries
- None — the change was as mechanical as the plan anticipated.

### Decisions
- Kept `width_mult` as a constructor argument (defaulting to 1) rather than reading the module constant inside the class, so the architecture stays parameterized and the call site documents the experiment variable.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash task bqv4lm81h (local)
- **Log file(s)**: run.log (project root)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 04:00
- **Ended**: 2026-06-10 04:07

Description:
- Full training run of the 4x-wide ResNet-20 (~4.3M params expected) under the unchanged EXP-000 recipe. Tests whether capacity is now the binding constraint. Expected: throughput drops to ~10–20k img/s, ~60–120 epochs, total_seconds well under 600, and best_test_acc ≥ 93.8% (hypothesis), with the pass bar at ≥ 93.26 (baseline 93.16 + 0.1).

Observations:
- Params line confirms the width change: "ResNet-20 (4x wide) | params: 4,286,026" (source: run.log L~2)
- Epoch-1 eval healthy: test_acc 39.20%, test_loss 1.6400 — no divergence at peak-LR scale (source: run.log first `eval ep` line)

Key Metrics:
- best_test_acc: 95.23% | final_test_acc: 95.18% | final_test_loss: 0.2447 (source: run.log summary block)
- training_seconds: 300.0 | total_seconds: 395.8 | startup_seconds: 1.2 (source: run.log summary block)
- peak_vram_mb: 1620.7 | num_epochs: 114 | num_steps: 10,965 | num_params: 4,286,026 (source: run.log summary block)

## Verification Results

### Conditions Checked

1. **Run completes without crashing within the time budget (≤ 10 min total)** — PASS
   - `best_test_acc:` present; total_seconds = 395.8 ≤ 600; exit code 0 (source: run.log summary; task bqv4lm81h)
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 93.26)** — PASS
   - best_test_acc = 95.23% vs baseline 93.16% → +2.07 pp (source: run.log summary; exp-index.sh baseline)
3. **Validation executed at most once per epoch** — PASS
   - eval lines = 114 = num_epochs (source: run.log)

### Informational Metrics

- peak_vram_mb: 1620.7 (vs 479.7 prior — still negligible on 98GB)
- num_epochs: 114 (vs 345 prior — eval overhead dropped to ~95s, total wall clock back to a comfortable 395.8s)
- num_params: 4,286,026 (16x the 270k baseline — width change confirmed)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
