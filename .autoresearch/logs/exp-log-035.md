# EXP-035: Clean-tail LR reheat (aug cooldown @0.10 + re-annealed LR 0.02→0 on the clean phase)

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-035.md
- **Plan**: plans/plan-035.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-035
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Milestone 1 per plan-035: re-applied the EXP-034 augmentation-cooldown scaffold to baseline `train.py` and added ONE new variable — a clean-phase LR reheat. Six edits, architecture untouched: (1) constants `COOLDOWN_FRAC = 0.10` + `CLEAN_LR0 = 0.02`; (2) `train_tf_clean` = full pipeline minus `TrivialAugmentWide()`; (3) `aug_cooled = False` in loop init; (4) epoch-boundary trigger swapping `train_set.transform` + observable marker, fires once `total_training_time/TIME_BUDGET_S ≥ 0.90`; (5) **NEW** — LR override: when `aug_cooled`, `lr = CLEAN_LR0 * 0.5 * (1 + cos(pi * clean_progress))` where `clean_progress` runs 0→1 over the final 10%, else unchanged `lr_at_fraction(frac)`; (6) gated `cutout_batch` behind `if not aug_cooled`. Smoke test passed: params 4,299,866 (unchanged), AST clean, diff = train.py only, constants present. LR-profile sanity confirmed the reheat gives ~0.0195 at frac 0.91 (vs global cosine 0.0044, ~4.4×) annealing to 0 at frac 1.0.

### Surprises & Discoveries
None during implementation — the cooldown transform-swap mechanism is identical to EXP-033/034 (verified to propagate to forked DataLoader workers and fire exactly once). The only new logic is the scalar LR branch.

### Decisions
- CLEAN_LR0 = 0.02 chosen (not higher) to balance "enough LR for the clean fine-tune to move" against "gentle on the converged solution" (the SGDR destabilization lesson, EXP-029). Grounded in EXP-020's best SWA floor (0.02).
- Re-annealed cosine (0.02→0) rather than a constant floor, to preserve the proven cosine-to-0 endpoint benefit while still front-loading real LR into the clean phase.
- COOLDOWN_FRAC held at 0.10 (EXP-034's best window) so CLEAN_LR0 is the single changed variable vs the 96.26 base — clean attribution.

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
Full 300s-compute-budget training of the k=4 WideResNet with the EXP-034 augmentation cooldown (full aug for first 90% of budget, then TrivialAugment+Cutout off, Crop+Flip kept) PLUS a clean-phase LR reheat: during the final 10% the LR follows a re-annealed cosine 0.02→0 instead of the near-frozen global-cosine tail (~0.001–0.005). Hypothesis: giving the clean fine-tune real LR budget lets the model adapt to the clean/test distribution, lifting best_test_acc above EXP-034's 96.26 toward the 96.32 bar. Expect ~91 ep, dt~8ms, params 4,299,866.

Observations:
- **Cooldown + reheat fired correctly**: `>>> aug cooldown ON at ep 83 frac 0.91` (identical firing point to EXP-034). Clean-phase step LR confirmed at the reheated level — `lr: 0.0145 → 0.0125 → ...` decaying (vs the ~0.0044 the global cosine would give at frac 0.91; the reheat is ~3-4× higher as designed), annealing toward 0. (source: run.log marker + `tr '\r' '\n' | grep "lr:"`)
- **Pre-cooldown base LOWER than EXP-034 (noise)**: best at cooldown start (ep81-82) was 95.80 vs EXP-034's 96.05 — ~0.25pp lower. The augmented phase is the same code path, so this is run-to-run throughput-jitter variance in the time-fraction LR schedule (early dt ran ~9ms before settling to 8ms; documented High-Importance protocol noise). This handicapped the absolute final number.
- **Clean-tail trajectory (reheat)**: ep82(aug) 95.65 → ep83 95.62 → ep84 95.70 → ep85 96.06 → ep86 95.92 → ep87 95.96 → **ep88 96.12 (peak)** → ep89 96.02 → ep90 96.07 → ep91 96.11. The reheat lifted the tail +0.32 over the pre-cooldown best (95.80→96.12) — a LARGER absolute climb than EXP-034's frozen-LR tail (+0.21, 96.05→96.26) — directionally supporting the LR-starvation hypothesis. BUT the tail bounced in a ~0.2pp band and final_test_loss stayed 0.2003 (never settled to baseline's 0.195 / EXP-034's 0.1951): the extra LR bought top-1 movement at the cost of not annealing the loss down.
- dt settled to 8ms; 91 epochs (throughput-neutral). No NaN/errors. (source: run.log)

Key Metrics:
- best_test_acc: **96.12%** @ ep88 (baseline 96.22, bar 96.32 → **−0.10pp vs baseline, −0.20pp vs bar**; −0.14 vs EXP-034's 96.26)
- final_test_acc: 96.11%; final_test_loss: 0.2003 (baseline 0.195; EXP-034 0.1951 — WORSE, reheat did not settle loss)
- num_epochs: 91 (throughput-neutral ✓); num_steps: 35,301; dt ~8ms
- num_params: 4,299,866 (unchanged ✓); peak_vram_mb: 453.8
- total_seconds: 403.5 (<600 ✓); training_seconds 300.0; cooldown fired ep83/frac 0.91 ✓
- (source: run.log summary block + marker + eval lines)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar (best_test_acc ≥ 96.32)**: **FAIL.** best_test_acc = 96.12% < 96.32 (−0.20pp below bar; −0.10pp below the 96.22 baseline). (source: `grep "^best_test_acc:" run.log` → 96.12%)
- **Cond 2 — clean completion within budget**: **PASS.** Summary block printed; `grep -c "Traceback|RuntimeError|NaN|Killed|CUDA error" run.log` == 0; total_seconds 403.5 < 600.
- **Cond 3 — no constraint violations**: **PASS.** `git diff --name-only` == only train.py; num_params 4,299,866 (unchanged); eval-count 91 == num_epochs 91 (≤ once/epoch); core torch only (no new deps/imports); seed 42 unchanged.

**Attribution / trustworthiness**: cooldown+reheat fired once at ep83/frac0.91 (✓), clean-phase LR confirmed reheated (~0.0145→0), 91 ep / dt 8ms (throughput-neutral fair test). Result is trustworthy as a fair test of the reheat mechanism. Caveat for analysis: the absolute number is confounded by a ~0.25pp-lower pre-cooldown base (augmented-phase noise) than EXP-034 — the reheat's relative tail-climb (+0.32) actually exceeded EXP-034's (+0.21), but the lower base + non-settling loss left the final below baseline.

### Informational Metrics
- peak_vram_mb: 453.8 (≈ baseline)
- num_epochs / num_steps: 91 / 35,301 (throughput-neutral)
- final_test_loss: 0.2003 (WORSE than baseline 0.195 and EXP-034's 0.1951 — the reheat kept the model in a higher-loss region; it did not anneal down)
- pre-cooldown base acc: 95.80 @ ep81 (vs EXP-034's 96.05 — augmented-phase run-to-run variance)

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
