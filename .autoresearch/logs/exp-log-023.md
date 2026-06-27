# Experiment Log EXP-023: Heat-constant momentum trade — MOMENTUM 0.95 + PEAK_LR 0.2

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-023.md
- **Plan**: plans/plan-023.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-023 (cut from autoresearch/dev @ 1990397)
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed (verification condition 1 not met — converged no-improvement, 96.41 vs 96.81 bar)

## Implementation Notes

### Summary
Two-constant edit to `train.py` per plan-023 Milestone 1: `MOMENTUM 0.9→0.95` and `PEAK_LR 0.4→0.2`, holding the first-order effective step lr/(1−β) = 4 at every point of the time-keyed schedule (lr_at() is multiplicative in PEAK_LR, so the entire heat profile scales by exactly 0.5 while 1/(1−β) doubles). Comments on both lines updated to carry the trade's derivation. Everything else byte-identical to the EXP-006 baseline — batch 512, default-mode torch.compile, foreach/nesterov SGD, WD 5e-4 selective, LS 0.1, TA+RE. Syntax check passed; diff is 1 file / 2 lines. This isolates a single degree of freedom: the gradient-averaging horizon (10→20 steps).

### Surprises & Discoveries
- None at implementation time — the optimizer consumes MOMENTUM as a scalar with zero execution-path change, exactly as planned.

### Decisions
- None beyond the plan. Signatures are expected to be byte-identical to baseline (dt ~22.4ms, ~139 epochs, VRAM ~1613MB, params 4,286,026); any dt deviation >10% with GPU-0 free is treated as contamination or premise-break, not as an experimental result (plan-023 § Abort Criteria "Signature break").

## Run Log

### Run 1
- **Description**: Full 300s-budget run of the heat-constant momentum trade on GPU 0. The only never-touched recipe constant (MOMENTUM, 0.9 since EXP-000) is probed via the admissible compensated trade — the heat axis is closed for uncompensated changes (EXP-010/014) but the closure entry itself names heat-compensated trades as the remaining class. Expected per hypothesis: signatures identical to baseline, smoother hot-phase trajectory, converged plateau at-or-above baseline's; success bar best_test_acc ≥ 96.81. Win or lose, this completes the constant-bracketing certification of the recipe.
- **Job ID**: local background composite (pre-check + launch + inline watchdog)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed, rc=0, clean signatures
- **Started**: 2026-06-10T12:30:02Z
- **Ended**: 2026-06-10T12:38:17Z
- **Observations**: Pristine run — watchdog windows all 21.7–22.7ms; post-hoc profile 0/268 windows >30ms, mean 22.3ms, expected 139.8 epochs vs 139 actual. Signatures BYTE-IDENTICAL to baseline as the hypothesis required: 139 epochs, VRAM 1613.0MB, startup 12.6s, total 484.6s, params 4,286,026 — the trade was indeed free in heat (first-order), epochs, and numerics. The metric still lost: converged plateau 96.23–96.41 over the final 7 evals (best 96.41 at ep133, final 96.30) — genuinely converged, deficit is dynamics. Early trajectory roughly on-family (ep1 37.04 vs family 38.2–39.0, within noise); mid-run drifted ~1pp below family and never recovered. Reading: doubling the averaging horizon REDUCES effective gradient noise — the same medicine EXP-022 just measured as harmful at 2× batch — and the smoothing cost the converged mean ~0.3pp that no variance effect repaid.
- **Key Metrics**: best_test_acc 96.41 | final 96.30 | final_test_loss 0.1899 | training_seconds 300.0 | total 484.6s | startup 12.6s | VRAM 1613.0MB | 139 epochs | 13460 steps | eval_lines 139 = num_epochs. Source: run.log summary block + task b0mjjuue1 output.

## Experimental Adjustments

## Errors & Dead Ends

## Verification Results

Pre-condition (contention profile, plan-023 §Verification): PASS — `windows>30ms: 0 of 268 | mean win 22.3 ms | expected epochs 139.8` vs 139 actual. Clean and analyzable.

### Conditions Checked

1. **best_test_acc ≥ 96.81 (baseline 96.71 + 0.1pp)** — FAILED. Actual: 96.41 (`grep "^best_test_acc:" run.log`). −0.30pp vs baseline, −0.40pp vs bar, from a fully converged plateau (final 7 evals 96.23–96.41). First-failure-stop: remaining conditions skipped.
2. **Run completes without crash ≤600s** — skipped (aborted after prior failure). For the record: rc=0, total_seconds 484.6 — would have passed.
3. **Validation ≤ once per epoch** — skipped (aborted after prior failure). For the record: 139 eval lines = 139 epochs — would have passed.

### Informational Metrics

## Human Notes
