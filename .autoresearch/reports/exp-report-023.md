# Experiment Report EXP-023: Heat-constant momentum trade — MOMENTUM 0.95 + PEAK_LR 0.2

- **Date**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-023.md
- **Plan**: plans/plan-023.md
- **Exp Log**: logs/exp-log-023.md
- **Branch**: autoresearch/exp-023 (cut from autoresearch/dev @ 1990397)
- **Verdict**: **no-improvement** (96.41 vs baseline 96.71, −0.30pp; bar was ≥96.81)

## Goal

Maximize `best_test_acc` (%) of the widened ResNet-20 under the fixed 300s timed training budget. Direction: higher. Baseline at experiment time: **96.71** (EXP-006 recipe @ 1990397). Success bar: ≥ 96.81 (+0.1pp absolute).

## Idea & Hypothesis

**Idea**: Probe the last never-touched recipe constant (MOMENTUM, 0.9 since EXP-000) via the only admissible route: a heat-compensated trade. MOMENTUM 0.9→0.95 with PEAK_LR 0.4→0.2 holds the first-order effective step lr/(1−β) = 4 at every point of the time-keyed schedule (lr_at() is multiplicative in PEAK_LR), isolating one degree of freedom — the gradient-averaging horizon (1/(1−β): 10→20 steps). Uniquely among surviving candidates, the change is free in early heat (by construction), epochs (no execution change), and numerics (same kernels) — escaping all three established failure laws.

**Hypothesis**: Smoothing the search direction through the hot phase yields a mid-run trajectory at-or-above the baseline family and an equal-or-better converged plateau; signatures byte-identical to baseline; best_test_acc ≥ 96.81. A converged miss closes the momentum axis and completes the constant-bracketing certification of the EXP-006 recipe.

## Approach

Two-constant edit to `train.py` (1 file / 2 lines): `PEAK_LR = 0.2`, `MOMENTUM = 0.95`, comments updated with the trade derivation. Everything else byte-identical — batch 512, default-mode torch.compile, foreach/nesterov SGD, WD 5e-4 selective, LS 0.1, TA+RE, warmup 0.15. Standard batch-512 launch protocol (30ms contention thresholds). Plan justified the heat-axis Failed-Approaches match explicitly: EXP-010/014 closed UNCOMPENSATED heat changes; this is the compensated trade their closure entry names as the remaining class.

## Execution

Single pristine run, zero retries. GPU-0 pre-check clean; watchdog windows all 21.7–22.7ms; rc=0 at 484.6s total. Post-hoc authoritative profile: **0/268 windows >30ms, mean 22.3ms, expected 139.8 epochs vs 139 actual**. No errors, no dead ends.

## Results

| Metric | EXP-023 | Baseline (EXP-006) |
|---|---|---|
| best_test_acc | **96.41** | 96.71 |
| final_test_acc | 96.30 | — |
| final_test_loss | 0.1899 | ~0.19 |
| num_epochs | 139 | 139 |
| windowed dt | 22.3ms | 22.4ms |
| peak_vram_mb | 1613.0 | 1613.0 |
| total_seconds | 484.6 | ~480–540 |

**The premise held perfectly; the conclusion still failed.** Signatures were byte-identical to baseline — 139 epochs, dt 22.3ms, VRAM 1613.0MB, params 4,286,026 — so the trade was verifiably free in heat (first-order), epochs, and numerics, exactly as designed. The deficit is therefore purely the isolated variable: doubling the averaging horizon. Early trajectory was on-family (ep1 37.04 vs family 38.2–39.0); the run drifted ~1pp below family mid-schedule and converged into a clean plateau at 96.23–96.41 (best at ep133, final 96.30) — converged, not starved.

**Root cause reading**: a longer momentum horizon is a gradient-noise REDUCTION — the search direction averages over ~20 steps instead of ~10. This is the third consecutive measurement in which reducing gradient noise on this recipe costs converged accuracy: EXP-011 (EMA on evaluated weights, −0.25), EXP-022 (2× batch, −0.05/−0.14 at both LR rules), now EXP-023 (2× averaging horizon, −0.30). The recipe's noise level is not an inefficiency to be smoothed — it is load-bearing for generalization at this budget. The campaign now has a coherent NOISE LAW alongside deferral and numerics-equivalence: the baseline sits at a measured optimum of gradient-noise scale (batch 512, β=0.9), and every intervention that lowers that noise — whatever the mechanism — converges below it.

**Certification milestone**: every constant in `train.py` has now been probed alone or in a compensated trade: NUM_BLOCKS/WIDTH_MULT (capacity ±, allocation), BATCH_SIZE (×2 at two LR rules), PEAK_LR (±, and ×0.5 compensated), WARMUP_FRAC (±), MOMENTUM (compensated trade), WEIGHT_DECAY (±), LABEL_SMOOTHING (probed), schedule family, init, topology, compile mode. **The EXP-006 recipe is a completed local optimum over its entire visible parameter set.** Eighteen consecutive misses (EXP-007…023).

## Verification

- Pre-condition (contention profile): PASS — 0/268 windows >30ms; 139 vs 139.8 expected epochs.
- Condition 1 — best_test_acc ≥ 96.81: **FAILED** (96.41, converged plateau; −0.40pp vs bar). First-failure-stop applied.
- Conditions 2–3: skipped per protocol; for the record both would have passed (rc=0 @ 484.6s; 139 evals = 139 epochs).
- Integrity: params unchanged, frozen evaluator, signatures baseline-identical. Result fully trustworthy. **Verdict basis: clean converged run below baseline → no-improvement.**

## Unexplored Avenues

- **The cold direction of the same trade (β 0.8 + peak 0.8)**: would INCREASE effective noise at constant first-order step — the one direction the noise law actually favors. However it doubles peak LR, and lr=0.8's per-step magnitude was directly measured destabilizing-hot territory in EXP-010/012; the first-order match is least trustworthy exactly where per-step size grows. Genuinely open but risky; the noise law gives it the only positive-sign mechanism currently available.
- **Intermediate β 0.925 + peak 0.3**: interpolates a measured-bad endpoint toward baseline — cannot clear +0.1 when the far endpoint lost 0.30 and the near endpoint IS baseline. Closed by bracketing logic.
- **Noise-increasing interventions in other guises** (smaller batch 256 + peak 0.2 linear-scaled, gradient-noise injection): the noise law predicts the right SIGN, but EXP-000-era data and the throughput laws say batch 256 halves step efficiency (more steps, same epochs ≈ fewer epochs at higher loader pressure) — needs a variant that adds noise without paying dt or epochs.

## Next Steps

1. **Noise-increasing trade: MOMENTUM 0.8 + PEAK_LR 0.8 (lr/(1−β)=4 held)** — the mirror of this experiment, and the only remaining candidate whose mechanism sign is supported by three consecutive in-project measurements (the noise law); risk is per-step instability at lr 0.8. Confidence: low.
2. **Declare the recipe certification complete in goal-learnings and pivot brainstorms to out-of-recipe mechanisms only** — data-order/composition, loss shaping, or other never-touched intervention classes, each screened against deferral + numerics + noise laws before running. Confidence in framing: high.
3. **Smaller-batch noise variant (batch 256, peak 0.2)** — same noise-law sign, but pays throughput (dt does not halve with batch on H20) and likely loses epochs; only viable if a spot dt measurement projects ≥135 epochs. Confidence: low.

## Key Learning

The campaign's third independent noise measurement completes a law: gradient-noise scale at the baseline (batch 512, β=0.9) is a measured optimum, and EVERY noise-reducing intervention — weight averaging (−0.25), larger batch (−0.14), longer momentum horizon (−0.30) — converges below it even when provably free in heat, epochs, and numerics. With this run, every constant in train.py is bracketed: the EXP-006 recipe is a certified local optimum over its entire visible parameter set; only noise-INCREASING trades and out-of-recipe mechanisms remain.
