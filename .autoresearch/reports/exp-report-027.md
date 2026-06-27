# Experiment Report EXP-027: Baseline variance replicate ×2 (zero diff)

- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-027.md
- **Plan**: plans/plan-027.md
- **Exp-log**: logs/exp-log-027.md
- **Verdict**: **no-improvement** (pre-registered; recorded metric 96.50 = mean of replicates 96.59 / 96.40; baseline 96.71 unchanged)

## Goal

Maximize CIFAR-10 `best_test_acc` (%) of the wide ResNet-20 within the fixed 300s timed budget. Baseline: **96.71** @ 1990397 (EXP-006). This loop's objective was MEASUREMENT, not improvement: quantify the run-level noise that twenty-one consecutive verdicts have been rendered against.

## Idea & Hypothesis

**Idea**: Run the unmodified baseline twice (zero diff, seed 42 untouched — nondeterminism comes from cudnn.benchmark autotuning and bf16 atomics). With the standing 96.71, this yields three draws of the identical configuration — the first real estimate of run-level σ and of where 96.71 sits in its own distribution. Deferred by three prior loops until intervention classes thinned; that condition was met after EXP-025 (alignment dead) and EXP-026 (activation dead). **Pre-registered**: verdict `no-improvement` regardless of values; a lucky bar-cross is variance, not improvement; the baseline does not move.

**Hypothesis**: replicates in 96.5–96.8; sub-predictions (a) both < 96.81, (b) spread ≤ 0.2, (c) at least one < 96.71.

## Approach

No code change (verified: empty diff, HEAD = 1990397, 3 `F.relu` sites intact). Two sequential standard composite launches with the full contention protocol; each replicate's summary, final-7 evals, and post-hoc profile persisted to the exp-log before run.log was overwritten.

## Execution

Both replicates pristine first-try (R1 task bnq7pbaey, R2 task bb1689sy7): rc=0, 0 slow windows (266/267 profiled), 139 epochs each, dt 22.3–22.4ms, VRAM 1613.0, params 4,286,026, startup ~13s, totals 503.6s / 493.2s. No retries, no contamination.

## Results

**R1 = 96.59, R2 = 96.40.** Identical-configuration draws: **{96.71, 96.59, 96.40} → mean 96.567, sample σ 0.156, range 0.31.** All three pre-registered sub-predictions held: (a) both < 96.81 ✓; (b) spread 0.19 ≤ 0.2 ✓; (c) both < 96.71 ✓.

**The central finding: 96.71 is the TOP of the baseline's own distribution (~+0.9σ above its mean ≈ 96.57), and the success bar (96.81 = baseline + 0.1) sits ≈ +1.5σ above the true mean.** Consequences, in order of importance:

1. **Required effect size**: to clear 96.81 reliably (>80% of runs), a candidate needs a TRUE effect of ≈ +0.25–0.4pp over the baseline MEAN. The campaign has been implicitly demanding +0.1 over a +0.9σ draw — a much taller order than the bar's nominal phrasing. This explains the 21-miss streak parsimoniously: interventions with true effects in ±0.1 each had ≲10% luck odds, and none hit.
2. **Reinterpretation of the noise band**: the five misses at −0.05…−0.15 (EXP-012 batch+LR, EXP-013 reflect-pad, EXP-020 shortcuts, EXP-022 √LR, EXP-026 hardswish) are within ~1σ of the mean — they measured "no detectable effect", not loss. Their axis-closure status is unchanged in practice (no evidence of GAIN either, and several cost dt/epochs), but their deficits should not be cited as mechanistic evidence. The campaign's structural laws rest on the LARGE deficits (≥0.2–0.3: heat ±, EMA, linear anneal, momentum trades, zero-γ, FixRes tail) and on replicated/bidirectional designs — those remain solid.
3. **Plateau structure**: both replicates show tight final-7 plateaus (ranges 0.11/0.17) with best = plateau top — the max-statistic harvests only ~0.0–0.03 above the plateau median. The metric is effectively "converged plateau level + small noise", confirming the law's reading; run-to-run σ comes from trajectory-level divergence (different cudnn algorithms/reduction orders compounding over 13k steps), not from eval-level jitter.
4. **Integrity note**: recorded metric 96.50 = mean(R1,R2) per plan; the TSV delta (−0.21) is the distance from the standing baseline draw, not an intervention effect.

## Verification

- **Pre-condition (profiles)**: PASS both (0/266, 0/267 slow windows; epochs exact).
- **Condition 1 (≥ 96.81)**: FAILED numerically on both replicates AND failed the integrity sub-check by construction (zero diff — pre-registered). Verdict `no-improvement` in every branch.
- **Conditions 2–3**: skipped per first-failure-stop; both would have passed on both replicates.
- **Verdict basis**: pre-registered measurement design; results fully trustworthy (byte-perfect signatures, pristine profiles).

## Key Learning

A best-over-checkpoints baseline recorded once is a biased reference: selection effects (improvements are adopted when they beat it) make the standing value sit high in its own distribution. Here the recorded baseline is +0.9σ above its measured mean, so the de-facto success bar is ~+1.5σ — candidates must bring ≥ +0.3pp true effects, and band-level results (±0.15) are uninterpretable without replication. Every serious campaign should buy this number early; it cost two runs and reframes all twenty-one prior verdicts.

## Unexplored Avenues

- **More replicates** (tighter σ): diminishing returns — 3 draws suffice for the strategic conclusion; a 4th would shave the σ CI but change no decision.
- **Replicating a near-band intervention** (e.g. rerun EXP-026 hardswish twice): could resolve whether a specific axis is truly ~0 — only worth it if that axis becomes a building block for a composition.

## Next Steps

1. **Only big-swing candidates from here**: target mechanisms with plausible ≥ +0.3pp effects — compositions of multiple small free wins, or genuinely structural changes that survive the four-laws screen plus the new effect-size screen. Confidence: high (as strategy).
2. **Width asymmetry (stage-3-only widen 256→320), gate-screened** — the last named structural unknown; with the new calibration, even its success would need to be large to matter, but its dt/epoch arithmetic is checkable in 90s via the early-dt gate. Confidence: low-medium.
3. **Terminal BN-stat recalibration** — expected magnitude (≤ +0.35 from a depressed level, likely ≪ from the top) is now KNOWN to be sub-bar with high probability; deprioritize unless composed with something else. Confidence: low.
