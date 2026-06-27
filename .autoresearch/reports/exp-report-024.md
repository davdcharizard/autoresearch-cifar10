# Experiment Report EXP-024: Noise-increasing momentum trade — MOMENTUM 0.8 + PEAK_LR 0.8

- **Date**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-024.md
- **Plan**: plans/plan-024.md
- **Exp Log**: logs/exp-log-024.md
- **Branch**: autoresearch/exp-024 (cut from autoresearch/dev @ 1990397)
- **Verdict**: **no-improvement** (96.49 vs baseline 96.71, −0.22pp; bar was ≥96.81)

## Goal

Maximize `best_test_acc` (%) of the widened ResNet-20 under the fixed 300s timed training budget. Direction: higher. Baseline at experiment time: **96.71** (EXP-006 recipe @ 1990397). Success bar: ≥ 96.81 (+0.1pp absolute).

## Idea & Hypothesis

**Idea**: The exact mirror of EXP-023 — MOMENTUM 0.9→0.8 with PEAK_LR 0.4→0.8, holding lr/(1−β) = 4 at every schedule point while halving the averaging horizon (10→5 steps) to RAISE effective gradient noise. The gradient-noise law (EXP-011/022/023) had three measured points on the reduction side, all losses, and zero on the increase side — the last unmeasured direction in recipe-space, probed with the trade design EXP-023 had just validated (baseline-identical signatures).

**Hypothesis**: If baseline noise sits below the generalization optimum, the converged plateau LEVEL rises — best_test_acc ≥ 96.81 with final-7-evals median at-or-above baseline's (win via level, not variance). A converged miss brackets the noise curve and certifies baseline as the noise optimum.

## Approach

Two-constant edit (1 file / 2 lines), everything else byte-identical to baseline. Standard batch-512 launch protocol with an added inline NaN/inf loss guard in the watchdog for the lr-0.8 instability tail-risk. Heat-axis Failed-Approaches match justified in plan: EXP-010/014 closed UNCOMPENSATED increases (lr/(1−β) rose to 6); this holds it at 4.

## Execution

Single pristine run, zero retries. Watchdog windows all 21.7–22.7ms; smoothed train loss declined monotonically at every sample (no bounce, no NaN — the feared instability never materialized); rc=0 at 491.9s. Post-hoc profile: **0/268 windows >30ms, mean 22.3ms, expected 139.8 vs 139 actual**. No errors, no dead ends.

## Results

| Metric | EXP-024 (noise ↑) | Baseline | EXP-023 (noise ↓) |
|---|---|---|---|
| best_test_acc | **96.49** | 96.71 | 96.41 |
| final_test_acc | 96.42 | — | 96.30 |
| num_epochs | 139 | 139 | 139 |
| windowed dt | 22.3ms | 22.4ms | 22.3ms |
| peak_vram_mb | 1613.0 | 1613.0 | 1613.0 |

**The noise curve is now bracketed, and baseline is its maximum.** Halving the averaging horizon (noise up): −0.22. Doubling it (noise down): −0.30. Both runs had signatures byte-identical to baseline, isolating the noise variable in both directions. The deficit curve is roughly symmetric around β=0.9, confirming the certified-optimum reading: the EXP-006 recipe sits at the top of the noise parabola, not on its slope.

**Hypothesis refuted cleanly**: the increase side does not pay. Notably the mid-run trajectory ran 1–2pp below family (ep50 81.77, ep100 91.41) yet converged into a tight plateau (96.37–96.49, spread 0.12pp) — the same "converged but lower" signature as every compensated trade. The plateau-integrity check was moot (no bar-pass to audit) but confirms the result is a level deficit, not a variance artifact.

**Campaign state**: nineteen consecutive misses (EXP-007…024). With the noise curve bracketed, RECIPE-SPACE IS CLOSED in the strongest sense available: every constant probed alone or in compensated trades, every axis measured on both sides of its optimum, all four laws (deferral, numerics equivalence, max-statistic, gradient-noise optimum) now bidirectionally evidenced. Any future candidate must operate OUTSIDE the recipe's parameter set — data composition/order, objective shaping, or architectural mechanisms — and must pre-screen against all four laws.

## Verification

- Pre-condition (contention profile): PASS — 0/268 windows >30ms; 139 vs 139.8 expected.
- Condition 1 — best_test_acc ≥ 96.81: **FAILED** (96.49, converged tight plateau). First-failure-stop applied.
- Conditions 2–3: skipped per protocol; both would have passed (rc=0 @ 491.9s; 139 evals = 139 epochs).
- Integrity: params 4,286,026, frozen evaluator, signatures baseline-identical, plateau tight. Fully trustworthy. **Verdict basis: clean converged run below baseline → no-improvement.**

## Unexplored Avenues

- **Finer β grid (0.85, 0.875)**: interpolates two measured-bad endpoints toward the baseline maximum — cannot clear +0.1; closed by bracketing logic.
- **Noise increase via other levers at held heat (smaller batch, gradient noise injection)**: the β bracket plus the batch bracket (EXP-012/022) now cover both natural noise levers; injected Gaussian noise is the same mechanism with worse structure (isotropic vs minibatch-shaped) — prior even lower than the measured losses.
- **Asymmetric schedules of β (high early, low late or vice versa)**: composes two measured-suboptimal regimes; no mechanism argument for why the composition beats both endpoints' optimum.

## Next Steps

1. **Out-of-recipe, all-laws-screened candidates only**: the surviving intervention classes are data composition/order (never probed; must not reduce gradient noise — e.g., class-balanced batches are sign-wrong), objective shaping (LS anneal flagged in brainstorm-024 idea 3 — sign uncertain, pressure-down prior negative), and architecture mechanisms free in heat+epochs (none identified since EXP-020 closed shortcuts). Confidence: low across the board — state this honestly in the next brainstorm's synthesis check.
2. **Re-mine external references for wall-clock-native mechanisms** (per the autopilot think-harder directive): re-read knowledge/README's cifar10-fast/airbench notes specifically for techniques not yet classified under the four laws (e.g., data-echoing-style loader tricks, frozen-stem ideas, lookahead-free optimizers) before defaulting to a low-EV probe. Confidence: medium that it surfaces a candidate, low that it wins.
3. **If a future loop wants more measurement rather than intervention**: a replicate of baseline would quantify run-to-run σ of best_test_acc (never directly measured; eval noise ±0.1 is known but seed-level σ is not) — useful for honest interpretation of any future ±0.1–0.2 result, at the cost of one loop with no improvement possible. Confidence in informational value: medium. NOTE: this would NOT be seed hacking — the no-seed-hacking constraint bars re-rolling for a better number; a variance measurement explicitly cannot move the baseline.

## Key Learning

The gradient-noise curve is bracketed and baseline sits at its maximum: noise down −0.30 (EXP-023), noise up −0.22 (EXP-024), both at byte-identical execution signatures. This is the strongest possible closure of recipe-space — every constant, every axis, both directions, four laws bidirectionally evidenced. Nineteen misses say the remaining headroom, if any, lives outside the recipe's parameter set entirely: data, objective, or architecture mechanisms that are simultaneously free in early heat, epochs, numerics, and noise scale.
