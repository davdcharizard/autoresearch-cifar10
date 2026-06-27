# EXP-010: PEAK_LR 0.4 → 0.6 on the compiled 4x recipe

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-010.md
- **Plan**: plans/plan-010.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-010
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (verification condition 2 — metric below baseline + 0.1pp)

## Implementation Notes

### Summary
Single constant change per plan: `PEAK_LR = 0.4` → `0.6`, comment updated to record the tuning rationale (super-convergence headroom on the augmented 4x recipe; the 0.4 was linear-scaled for EXP-000's unaugmented 1x net). Everything else — architecture, schedule shape, augmentation, WD, compile — byte-identical to baseline 1990397. First experiment on the optimization-hyperparameter surface after capacity (EXP-007/008) and regularization (EXP-009) were closed. Diff: 1 line, train.py only. Ruff clean; GPU 0 idle at launch.

### Surprises & Discoveries
- None at implementation time.

### Decisions
- None beyond the plan.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash task b1lmbkbrp (local)
- **Log file(s)**: run.log (project root)
- **WandB**: N/A
- **Status**: completed (metric below bar)
- **Started**: 2026-06-10 06:30
- **Ended**: 2026-06-10 06:39

Description:
- One-cycle peak LR raised 1.5x (0.4 → 0.6) on the otherwise-frozen compiled 4x TA+RE recipe — directional probe of the never-retuned LR optimum. Expect dt ~22ms, ~137–139 epochs, VRAM ~1.62GB, total ~480–500s. Pass bar ≥ 96.81. Predicted signatures: deeper mid-schedule depression than EXP-006 (hotter peak), full recovery in the anneal, no NaN. Abort only on NaN/collapse (eval < 30% after epoch 20) — depressed-but-finite mid-schedule is the expected signature, not a failure.

Observations:
- Params 4,286,026 — unchanged as required (source: run.log L2)
- Epoch-1 eval: test_acc 34.39% — healthy, indistinguishable from EXP-006's 35.11 (LR at epoch 1 is still in warmup, ~0.03 vs 0.02 baseline) (source: run.log first `eval ep` line)
- dt 22ms at steps 100–150 (~23k img/s) — exactly baseline, as required for a pure LR-constant change (source: run.log step lines)

Key Metrics:
- best_test_acc: 96.14% | final_test_acc: 96.14% | final_test_loss: 0.1881 (source: run.log summary block)
- training_seconds: 300.0 | total_seconds: 480.8 | startup_seconds: 10.8 (source: run.log summary block)
- peak_vram_mb: 1613.0 | num_epochs: 139 | num_steps: 13,470 | num_params: 4,286,026 (source: run.log summary block)
- Throughput identical to baseline as required (139 epochs, dt 22ms) — pure LR effect
- Trajectory: ~3pp below EXP-006 through the hot mid-schedule (ep 80: 87.2 vs ~91; ep 120: 94.86) and the anneal did NOT fully recover — final epochs still creeping (+0.08 over last 4: 96.06→96.14, final=best). The hot peak's lost ground exceeded what 139 epochs of descent could repay (source: run.log eval lines)

## Verification Results

### Conditions Checked

1. **Run completes without crashing within the time budget (≤ 10 min total)** — PASS
   - `best_test_acc:` present; total_seconds = 480.8 ≤ 600; exit 0; no NaN at any point (source: run.log summary; task b1lmbkbrp)
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 96.81)** — FAIL
   - best_test_acc = 96.14% vs baseline 96.71% → −0.57 pp (source: run.log summary; exp-index.sh baseline)
3. **Validation executed at most once per epoch** — skipped — aborted after prior failure (informally: 139 eval lines = 139 epochs, compliant)

### Informational Metrics

- (not collected — necessary condition failed; values noted in Key Metrics for the record)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
