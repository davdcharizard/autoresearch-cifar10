# EXP-073: Nesterov momentum ON → OFF (vanilla heavy-ball SGD)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-073.md
- **Plan**: plans/plan-073.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-073
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Applied Milestone 1 verbatim: single-flag edit at train.py L205, `nesterov=True` → `nesterov=False` inside the `optim.SGD(...)` call. MOMENTUM stays 0.9, everything else byte-identical to EXP-054. Smoke test passed: AST OK; grep confirms exactly one `nesterov=` line reading `False`; `optim.SGD(..., nesterov=False)` constructs with `param_groups[0]['nesterov'] is False`; `git diff --name-only` == train.py only.

### Surprises & Discoveries
- None. The change is the minimal single-flag toggle the plan specified.

### Decisions
- No deviations. Momentum held at 0.9 to isolate the Nesterov look-ahead from the effective-step-magnitude axis (which the momentum-coefficient Ideas 2/3 would have perturbed).

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (background bash, PID at launch)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-10
- **Ended**: 2026-06-10

Description:
- Running the EXP-073 Nesterov-off probe: `CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1` on idle GPU 1 (both GPUs 0%/0 MiB at launch). Tests whether removing Nesterov's look-ahead update (vanilla heavy-ball momentum, same effective step) moves best_test_acc off the 96.45 baseline. Expected: within ±0.25pp of 96.45 (clean null or small regression) — the last untested optimizer cell, closing the optimizer-internal axis. dt should stay 8ms (compute-identical to Nesterov).

Observations:
- **Early gate (≤ep7) PASSED**: dt steady 8ms (occasional 9ms), img/s ~15,400 — no graph break, no contention; heavy-ball is compute-identical to Nesterov as predicted. Eval climbing normally: ep5 71.67%, ep6 77.88% — healthy, tracking EXP-054 within normal run-to-run variation. lr at peak 0.200 annealing. No NaN. (source: run.log eval ep5/ep6 lines, steps 2000-2350)

Key Metrics:
- ep5 test_acc: 71.67%; ep6 test_acc: 77.88% (source: run.log "eval ep" lines)
- **best_test_acc: 96.12%** (−0.33pp vs baseline 96.45) (source: run.log "best_test_acc:")
- final_test_loss: 0.2020 (HIGHER than EXP-054's 0.1968 — both top-1 AND loss worse); training_seconds 300.0; total_seconds 595.5; num_epochs 92; num_steps 35661; num_params 4,299,866; peak_vram_mb 453.8 (source: run.log summary)

## Verification Results

### Conditions Checked

- **Necessary condition 1 — `best_test_acc >= 96.55`**: best_test_acc = **96.12** < 96.55. **FAILED** (−0.43pp below bar, −0.33pp below baseline 96.45). Stop at first failed condition.
- **Necessary condition 2 — clean completion within budget** (recorded for completeness): training_seconds 300.0 ✓, total_seconds 595.5 < 600 ✓, num_params 4,299,866 UNCHANGED ✓, 0 nan/traceback/error ✓, 92 epochs.
- **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only ✓; no new deps; seed 42; evaluate() once/epoch; uncontended (dt steady 8ms).

**Verdict: no-improvement.** Clean valid run (Σdt=300.0, wall 595.5 < 600, dt 8ms / no graph break, train.py only) that missed the bar. Results trustworthy — direct parse, 0 NaN, healthy trajectory. NOT invalid (no breach; optimizer flag, params unchanged) and NOT crash (real interpretable metric).

### Informational Metrics

- num_epochs 92 / num_steps 35661 (throughput unchanged — heavy-ball = Nesterov compute cost; wall 595.5s ≈ EXP-054's 593s).
- **final_test_loss 0.2020 — HIGHER than EXP-054's 0.1968** (both top-1 AND calibration worse — NOT a polish-vs-top1 split; a genuine uniform degradation).
- peak_vram_mb 453.8.
- **Key observation**: removing Nesterov regressed −0.33pp (squarely in the −0.2..−0.6pp scalar-knob band), with loss ALSO worse — confirming the Nesterov look-ahead is genuinely LOAD-BEARING in this tuned recipe (it converges to a meaningfully better minimum, not just lower loss). This closes the last untested optimizer-internal cell: Nesterov is part of the tuned optimum (consistent with EXP-043 showing SGD+Nesterov beats AdamW). The optimizer axis (family + dynamics + the Nesterov flag) is now fully mapped.

## Errors & Dead Ends

<!-- none -->

## Human Notes

> (none — autopilot)
