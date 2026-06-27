# Experiment Log EXP-027: Baseline variance replicate ×2 (zero diff)

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-027.md
- **Plan**: plans/plan-027.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-027 (cut from autoresearch/dev @ 1990397)
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed (verification condition 1 not met on both replicates — pre-registered no-improvement; measurement objective fully achieved: σ ≈ 0.16, baseline mean ≈ 96.57)

## Implementation Notes

### Summary
No implementation — by design. Milestone 1 verified on branch creation: `git diff --stat` empty, HEAD = 1990397, `grep -c "F.relu" train.py` = 3 (baseline intact). The experiment is two sequential launches of the unmodified baseline with the standard composite launcher (no early-dt gate — baseline dt is certified at 22.4ms). PRE-REGISTERED (plan-027 § PRE-REGISTRATION, binding): verdict is `no-improvement` regardless of measured values; TSV metric = mean(R1, R2); the baseline does not move; a lucky ≥96.81 draw is variance, not improvement.

### Surprises & Discoveries
- None at setup.

### Decisions
- run.log is overwritten between replicates; each replicate's summary block, final-7 evals, and post-hoc profile are persisted into this log BEFORE launching the next run (plan Milestone 2).

## Run Log

### Run 1 (Replicate R1)
- **Description**: Unmodified baseline (seed 42, identical code to 1990397) on GPU 0. Purpose: first draw for the run-level σ estimate — the only varying factor is nondeterminism (cudnn.benchmark autotuning, bf16 atomics). Expected: signatures exactly baseline (dt 22.4ms, ~139 epochs, VRAM 1613MB, params 4,286,026); best_test_acc predicted in 96.5–96.8 (brainstorm-027 hypothesis).
- **Job ID**: local background composite, task bnq7pbaey
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed, rc=0, clean signatures
- **Started**: 2026-06-10T13:46:45Z
- **Ended**: 2026-06-10T13:55:31Z
- **Observations**: Pristine: 0/266 windows >30ms, mean win 22.4ms, expected 139.0 vs 139 actual. Signatures byte-perfect baseline (VRAM 1613.0, params 4,286,026, startup 12.9s, 13390 steps). **R1 best = 96.59 — 0.12 BELOW the standing 96.71.** Plateau very tight: final-7 range 96.48–96.59, median 96.56; best equals the plateau top (no outlier spike). Sub-prediction (c) [at least one replicate < 96.71] already satisfied.
- **Key Metrics**: R1 best_test_acc 96.59 | final 96.56 | final_test_loss 0.1845 | total 503.6s | 139 epochs | final-7 median 96.56. Source: task bnq7pbaey output + run.log (persisted here before overwrite).

### Run 2 (Replicate R2)
- **Description**: Second draw, identical procedure to R1. With {96.71 standing, R1 96.59}, R2 determines whether 96.71 sits at the distribution's top (R2 ≤ ~96.65) or center (R2 ≥ ~96.7).
- **Job ID**: local background composite, task bb1689sy7
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed, rc=0, clean signatures
- **Started**: 2026-06-10T13:56:24Z
- **Ended**: 2026-06-10T14:04:54Z
- **Observations**: Pristine: 0/267 windows >30ms, mean win 22.3ms, expected 139.5 vs 139 actual. Signatures byte-perfect baseline (VRAM 1613.0, params 4,286,026, startup 12.5s, 13446 steps). **R2 best = 96.40 — 0.31 below the standing 96.71.** Plateau again tight (final-7 range 96.23–96.40, median 96.39) and best = plateau top, no spike.
- **Key Metrics**: R2 best_test_acc 96.40 | final 96.40 | final_test_loss 0.1888 | total 493.2s | 139 epochs | final-7 median 96.39. Source: task bb1689sy7 output + run.log (persisted here).

### σ analysis (Milestone 3)
Draws of the identical configuration: **{96.71 (standing, EXP-006), 96.59 (R1), 96.40 (R2)}** → mean **96.567**, sample σ **0.156**, range 0.31. Pre-registered sub-predictions: (a) both replicates < 96.81 ✓; (b) |R1−R2| = 0.19 ≤ 0.2 ✓; (c) min < 96.71 ✓ (both were). **Conclusion: 96.71 is the TOP of the baseline distribution (~+0.9σ above its mean ≈96.57), and the success bar 96.81 sits ≈ +1.5σ above the mean.** A zero-effect intervention has roughly a 5–10% chance per run of crossing the bar by luck; a candidate needs a TRUE effect of ≈ +0.25–0.4pp to clear it reliably. This single number reframes the 21-miss streak: every miss in the −0.05…−0.15 band (EXP-012/013/020/022/026) is within ~1σ — "no detectable effect", not "measured loss" — while the campaign's structural conclusions (deferral law, noise law, etc.) rest on the larger deficits (≥0.2–0.3) and replicated trades, which remain valid.

## Experimental Adjustments

## Errors & Dead Ends

## Verification Results

### Conditions Checked

**Pre-condition — clean post-hoc profiles**: PASSED both replicates. R1: 0/266 windows >30ms, expected 139.0 vs 139. R2: 0/267, expected 139.5 vs 139. Both analyzable.

**Condition 1 — best_test_acc ≥ 96.81**: **FAILED on both replicates numerically** (R1 96.59, R2 96.40 — both < 96.81), AND fails the integrity sub-check by construction (zero diff — no intervention mechanism; pre-registered in plan-027 § PRE-REGISTRATION). Verdict `no-improvement` in every branch as pre-registered. First-failure-stop: conditions 2–3 not evaluated.

**Condition 2 — completes within budget**: skipped per first-failure-stop; would have passed both (rc=0; 503.6s / 493.2s ≤ 600).

**Condition 3 — validation at most once per epoch**: skipped per first-failure-stop; would have passed both (139 = 139 each).

### Informational Metrics

- R1 = 96.59 (final-7 median 96.56), R2 = 96.40 (final-7 median 96.39); spread 0.19
- Sample of identical config: {96.71, 96.59, 96.40} → mean 96.567, sample σ 0.156
- Signatures: both byte-perfect baseline (139 epochs, VRAM 1613.0, params 4,286,026, dt 22.3–22.4ms, startup ~13s)
- Max-over-evals harvests only ~0.0–0.03 above the final-7 median in both replicates (tight plateaus — the max-statistic's reward is plateau LEVEL here, confirming the law's reading)

## Human Notes
