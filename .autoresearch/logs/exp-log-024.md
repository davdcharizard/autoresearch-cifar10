# Experiment Log EXP-024: Noise-increasing momentum trade — MOMENTUM 0.8 + PEAK_LR 0.8

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-024.md
- **Plan**: plans/plan-024.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-024 (cut from autoresearch/dev @ 1990397)
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed (verification condition 1 not met — converged no-improvement, 96.49 vs 96.81 bar)

## Implementation Notes

### Summary
Two-constant edit to `train.py` per plan-024 Milestone 1: `MOMENTUM 0.9→0.8` and `PEAK_LR 0.4→0.8` — the exact mirror of EXP-023, holding lr/(1−β) = 4 at every schedule point while HALVING the averaging horizon (10→5 steps) to raise effective gradient noise. Comments carry the derivation. Everything else byte-identical to baseline. Syntax check passed; diff is 1 file / 2 lines. This probes the unmeasured INCREASE side of the gradient-noise law (goal-learnings § Patterns) with the trade design EXP-023 just validated (signatures provably baseline-identical).

### Surprises & Discoveries
- None at implementation time.

### Decisions
- None beyond the plan. Reward-hacking guard noted for verification: if the metric clears the bar, the final-7-evals median must be at-or-above the baseline family's (~96.6) — a variance-driven outlier spike over a flat plateau will be flagged toward no-improvement in substance.

## Run Log

### Run 1
- **Description**: Full 300s-budget run of the noise-increasing momentum trade on GPU 0. EXP-011/022/023 measured three gradient-noise REDUCTIONS, all converging below baseline (−0.25/−0.14/−0.30); this run measures the increase side at held first-order step — the only mechanism direction not excluded by the campaign's four laws. Expected: signatures byte-identical to baseline (dt ~22.4ms, ~139 epochs, VRAM ~1613MB); a bouncier hot phase than baseline is acceptable (EXP-012 precedent at lr 0.8); success bar best_test_acc ≥ 96.81 via plateau LEVEL. A converged miss brackets the noise curve and certifies baseline as the noise optimum, closing recipe-space.
- **Job ID**: local background composite (pre-check + launch + inline watchdog)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed, rc=0, clean signatures
- **Started**: 2026-06-10T12:46:15Z
- **Ended**: 2026-06-10T12:54:45Z
- **Observations**: Pristine run — watchdog windows all 21.7–22.7ms, no NaN/inf, no divergence; post-hoc profile 0/268 windows >30ms, mean 22.3ms, expected 139.8 vs 139 actual. Signatures byte-identical to baseline for the second consecutive trade (139 epochs, 13460 steps, VRAM 1613.0MB, startup 12.0s, total 491.9s). The feared lr-0.8 instability did NOT materialize — smoothed train loss declined monotonically across every watchdog sample; mid-run trajectory ran ~1–2pp below the baseline family (ep50 81.77, ep75 87.76, ep100 91.41) and converged into a clean plateau: final 7 evals 96.37–96.49 (median ~96.44), best 96.49 at ep133, final 96.42. Converged, not starved. The noise INCREASE side loses −0.22pp — together with EXP-023's −0.30 on the decrease side, the noise curve is now bracketed and baseline (β 0.9, batch 512) is its measured optimum.
- **Key Metrics**: best_test_acc 96.49 | final 96.42 | final_test_loss 0.1880 | training_seconds 300.0 | total 491.9s | startup 12.0s | VRAM 1613.0MB | 139 epochs | 13460 steps | eval_lines 139 = num_epochs. Source: run.log summary block + task b61fjkzyf output.

## Experimental Adjustments

## Errors & Dead Ends

## Verification Results

Pre-condition (contention profile, plan-024 §Verification): PASS — `windows>30ms: 0 of 268 | mean win 22.3 ms | expected epochs 139.8` vs 139 actual. Clean and analyzable.

### Conditions Checked

1. **best_test_acc ≥ 96.81 (baseline 96.71 + 0.1pp)** — FAILED. Actual: 96.49 (`grep "^best_test_acc:" run.log`). −0.22pp vs baseline, −0.32pp vs bar, from a fully converged plateau (final 7 evals 96.37–96.49, median ~96.44). The plateau-integrity sub-check is moot (no pass to audit); for the record the plateau is tight (spread 0.12pp), so the result is a genuine level deficit, not a variance artifact. First-failure-stop: remaining conditions skipped.
2. **Run completes without crash ≤600s** — skipped (aborted after prior failure). For the record: rc=0, total_seconds 491.9 — would have passed.
3. **Validation ≤ once per epoch** — skipped (aborted after prior failure). For the record: 139 eval lines = 139 epochs — would have passed.

### Informational Metrics

## Human Notes
