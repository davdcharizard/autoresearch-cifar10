# EXP-007: Aligned width 6x on the compiled recipe

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-007.md
- **Plan**: plans/plan-007.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-007
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (verification condition 2 — metric below baseline + 0.1pp)

## Implementation Notes

### Summary
Single constant change per plan: `WIDTH_MULT = 4` → `6` (stage widths 96/192/384 — all 32-aligned per the project-insights H20 alignment rule), comment updated. The compiled doubly-regularized recipe (torch.compile + pre-loop warmup, TA, RandomErasing, time-keyed one-cycle) stays byte-identical to baseline 1990397. This retries the count-2 failed approach (capacity without throughput) with both root causes removed: 1.22x compile throughput and aligned channels — justification documented in plan-007 § Failed-Approach Retry Justification. Ruff clean; only train.py modified; GPU 0 idle at launch.

### Surprises & Discoveries
- None at implementation time.

### Decisions
- None beyond the plan.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash task b3rwhznuk (local)
- **Log file(s)**: run.log (project root)
- **WandB**: N/A
- **Status**: completed (metric below bar)
- **Started**: 2026-06-10 05:25
- **Ended**: 2026-06-10 05:32

Description:
- 6x-wide (96/192/384, ~9.6M params) compiled TA+RE recipe — the decisive aligned-capacity probe after EXP-006 proved epochs convert. Expect ~75–80 epochs (139 / 2.25^0.76), dt ~47–55ms, VRAM ~3.5GB, total ~430–460s. Pass bar ≥ 96.81; hypothesis ≥ 96.85. Failure signature that closes the width direction: depressed accuracy with final≈best at ~75 epochs.

Observations:
- Params 9,636,202 — matches ~9.6M prediction (source: run.log L2)
- Epoch-1 eval: test_acc 37.87% — healthy (source: run.log first `eval ep` line)
- dt 57ms (~9.0k img/s) vs predicted 47–55ms: 2.59x the 4x step time for 2.25x FLOPs — compiled scaling is slightly SUPERlinear, not the sublinear extrapolation from eager EXP-002. Projects ~54 epochs, BELOW the ~70 starvation floor. Below the 70ms abort bar → run continues; undertraining is now the likelier outcome (source: run.log step lines, ~ep 8)

Key Metrics:
- best_test_acc: 96.00% | final_test_acc: 95.97% | final_test_loss: 0.2067 (source: run.log summary block)
- training_seconds: 300.0 | total_seconds: 395.4 | startup_seconds: 20.9 (source: run.log summary block)
- peak_vram_mb: 2411.8 | num_epochs: 55 | num_steps: 5,287 | num_params: 9,636,202 (source: run.log summary block)
- Throughput: dt 57ms / ~9.0k img/s — compiled width scaling is ~linear-or-worse in FLOPs (2.59x time for 2.25x FLOPs), NOT the sublinear eager extrapolation; 55 epochs landed in the starvation zone
- final ≈ best (95.97/96.00) — the predicted undertraining failure signature, now observed at aligned channels + compile

## Verification Results

### Conditions Checked

1. **Run completes without crashing within the time budget (≤ 10 min total)** — PASS
   - `best_test_acc:` present; total_seconds = 395.4 ≤ 600; exit 0 (source: run.log summary; task b3rwhznuk)
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 96.81)** — FAIL
   - best_test_acc = 96.00% vs baseline 96.71% → −0.71 pp (source: run.log summary; exp-index.sh baseline)
3. **Validation executed at most once per epoch** — skipped — aborted after prior failure (informally: 55 eval lines = 55 epochs, compliant)

### Informational Metrics

- (not collected — necessary condition failed; values noted in Key Metrics for the scaling record)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
