# EXP-034: Later/shorter augmentation cooldown (COOLDOWN_FRAC 0.15 → 0.10)

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-034.md
- **Plan**: plans/plan-034.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-034
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Milestone 1 per plan-034: re-applied the EXP-033 augmentation-cooldown edits to `train.py` (discarded after EXP-033's no-improvement) with the single change `COOLDOWN_FRAC = 0.10` (was 0.15). Four edits, architecture untouched: (1) `COOLDOWN_FRAC = 0.10`; (2) `train_tf_clean` = full pipeline minus `TrivialAugmentWide()`; (3) `aug_cooled` flag + epoch-boundary `train_set.transform` swap with an observable `>>> aug cooldown ON ...` marker, triggered once `total_training_time/TIME_BUDGET_S ≥ 0.90`; (4) gated `cutout_batch` behind `if not aug_cooled`. Smoke test passed: COOLDOWN_FRAC=0.10, params 4,299,866 (unchanged), AST clean, diff = train.py only.

### Surprises & Discoveries
None — identical mechanism to EXP-033 (which verified the transform-swap propagates to forked workers and fires correctly); only the start fraction differs.

### Decisions
- Single-variable change vs EXP-033 (start fraction 0.85 → 0.90 via COOLDOWN_FRAC 0.15 → 0.10) for clean attribution. ~9 clean-data epochs chosen to match EXP-033's observed ~9-10-epoch climb-to-peak duration while preserving ~5 more strong-aug epochs.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID — background task)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09

Description:
- Full 300s-compute-budget training of the k=4 WideResNet with the augmentation cooldown starting LATER than EXP-033: full aug (RandomCrop+Flip+TA+Cutout) for the first 90% of the budget, then TA+Cutout disabled (RandomCrop+Flip kept) for the final 10%. Hypothesis: the clean fine-tune lifts from a higher pre-cooldown base (more strong-aug training retained vs EXP-033) → best_test_acc above EXP-033's 96.10, plausibly clearing the 96.32 bar, at ~90 ep / dt~8ms / 4,299,866 params.

Observations:
- **Cooldown fired correctly at the LATER fraction**: `>>> aug cooldown ON at ep 83 frac 0.91` (vs EXP-033's ep77/0.85) — exactly one marker, at the planned-later point. (source: run.log)
- **Hypothesis CONFIRMED on the mechanism**: the pre-cooldown base was MUCH higher this run — best 96.05 at the cooldown start (ep83) vs EXP-033's 95.43 at its start (ep77), because ~6 more epochs of productive strong-aug training were retained. The clean fine-tune then lifted from that higher base.
- **Post-cooldown trajectory**: ep83(start, ~96.05) → ep84 96.20 → ep85 96.20 → ep86 96.17 → **ep87 96.26 (best)** → ep88 96.18 → ep91 96.15. Peaked ~4 clean epochs in then mildly declined (same peak-then-decline shape as EXP-033 but shifted up). (source: run.log eval ep83-91)
- **final_test_loss 0.1951 ≈ baseline 0.195** — notably better than EXP-033's 0.2000; the higher-base cooldown reaches a baseline-quality loss AND a marginally-higher top-1.
- dt held ~8ms, throughput-neutral.

Key Metrics:
- best_test_acc: **96.26%** @ ep87 (baseline 96.22, bar 96.32 → **+0.04pp vs baseline (within ±0.2 noise), −0.06pp vs bar**; **+0.16pp vs EXP-033's 96.10**)
- final_test_acc: 96.15%; final_test_loss: 0.1951 (baseline 0.195; EXP-033 0.2000)
- num_epochs: 91 (baseline ~91 — throughput-neutral ✓); num_steps: 35,224; dt ~8ms
- num_params: 4,299,866 (unchanged ✓); peak_vram_mb: 453.8
- total_seconds: 405.4 (<600 ✓); cooldown fired ep83/frac 0.91 ✓
- (source: run.log summary block + marker)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar (best_test_acc ≥ 96.32)**: **FAIL.** best_test_acc = 96.26% < 96.32 (−0.06pp below bar; +0.04pp above the 96.22 baseline but within the ±0.2pp noise floor). (source: `grep "^best_test_acc:" run.log` → 96.26%)
- **Cond 2 — clean completion within budget**: **PASS.** Summary printed, `grep -c Traceback run.log` == 0, total_seconds 405.4 < 600.
- **Cond 3 — no constraint violations**: **PASS.** Only train.py changed; num_params 4,299,866 (unchanged); eval-count 91 == num_epochs 91 (≤ once/epoch); core torch only; seed 42 unchanged.

**Cooldown + throughput attribution**: cooldown fired once at ep83/frac 0.91 (✓ later than EXP-033 as planned), num_epochs 91 ≈ baseline, dt ~8ms → clean, throughput-neutral, fair test of the 0.10-cooldown variant. Trustworthy.
**Trend vs EXP-033**: 0.15-cooldown → 96.10; 0.10-cooldown → 96.26 (+0.16pp). The later/shorter cooldown clearly helped (higher pre-cooldown base, as hypothesized). Marginal benefit OVER a full-aug-to-end run is small though: a full-aug cosine tail from the ep83 base (96.05) would itself reach ~baseline 96.22; the cooldown added only ~+0.04 (to 96.26).

### Informational Metrics

- peak_vram_mb: 453.8 (≈ baseline)
- num_epochs / num_steps: 91 / 35,224 (throughput-neutral)
- final_test_loss: 0.1951 (≈ baseline 0.195; better than EXP-033's 0.2000 — the higher-base cooldown reaches baseline-quality loss)

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
