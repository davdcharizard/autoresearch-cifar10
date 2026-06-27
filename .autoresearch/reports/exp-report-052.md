# Experiment Report EXP-052: Replicate-pair resolution of the anti-aliased shortcut (n=2, MEAN decision)

- **Date**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-052.md
- **Plan**: plans/plan-052.md
- **Exp-log**: logs/exp-log-052.md
- **Verdict**: no-improvement
- **Metric**: 96.70 (pre-registered MEAN of best_A=96.84, best_B=96.56) vs baseline 96.71 (bar 96.81)

## Goal

Maximize CIFAR-10 best_test_acc (%) within the fixed 300s charged training budget; higher is better. Baseline 96.71 @ 1990397 (EXP-006 recipe); improvement requires ≥ +0.1pp, i.e. ≥ 96.81. σ context (EXP-027): recipe mean ≈ 96.57, σ ≈ 0.16, σ_mean(n=2) ≈ 0.113.

## Idea & Hypothesis

EXP-046's anti-aliased shortcut (the pad shortcut's strided slice `[::2,::2]` → `F.avg_pool2d(x, 2)` at both stage transitions; BlurPool anchor, Zhang 2019) held the project's only positive-direction unresolved datum: a single 96.65 read (+0.5σ) at pristine signatures, classified absorbed-null at n=1. Since n=1 cannot distinguish a true +0.1–0.2 effect from zero, EXP-052 re-applied the identical diff and ran a pre-registered replicate pair: two byte-identical runs, decision statistic = MEAN ≥ 96.81 (1.6% false-positive under H0 — stricter than the standard single-run protocol's 6.7%; the max is never a decision input). Hypothesis: if a BlurPool-class effect ≥ +0.2 partially survives augmentation absorption, the mean clears the bar. Pre-registered branches: (i) mean ≥ 96.81 improvement; (ii) mean ∈ [96.61, 96.80] weak-positive-closed; (iii) mean ≤ 96.60 confirmed-null-closed; (iv) gate kills → infra relaunch.

## Approach

One logic line in `train.py` `BasicBlock.forward` (byte-equivalent to the EXP-046 diff): `shortcut = shortcut[:, :, ::self.stride, ::self.stride]` → `if self.stride != 1: shortcut = F.avg_pool2d(shortcut, self.stride)`; channel zero-pad unchanged. Zero parameters (4,286,026 exact), affects only layer2[0] and layer3[0]. CPU sanity all-pass: semantic equality on constant inputs / difference on random, pad-site assert, 2-step smoke decreasing. Both runs launched via `/tmp/exp046_composite.sh` verbatim (dual gates, 26ms D0 threshold, watchdog); working tree asserted unchanged between runs; Run A's run.log preserved to /tmp/exp052_runA.log before Run B.

## Execution

Both runs pristine on first launch, no retries:
- **Run A**: GATES_CLEAR poll 1; D0 = 22.5ms (identical to EXP-046's measured D0); windows 22.0–23.2ms; 138 ep, 13,349 steps, 300.0s training, 493.2s total; **best_A = 96.84**, final_test_loss 0.1790, converged-flat tail (96.84/96.83/96.82/96.79).
- **Run B**: GATES_CLEAR poll 1; D0 = 22.0ms; windows 21.7–23.2ms; 138 ep, 13,363 steps, 300.0s training, 503.8s total; **best_B = 96.56**, final_test_loss 0.1826, converged-flat tail.

**MEAN = 96.70** → branch (ii).

## Results

The pre-registered mean missed the bar by 0.11. The pair tells a textbook variance story: A (96.84) and B (96.56) straddle the recipe mean almost symmetrically; spread |A−B| = 0.28 ≈ 1.2σ of a pair difference (σ√2 ≈ 0.23) — unremarkable, no integrity flag. Pooling all three variant draws (96.65, 96.84, 96.56): mean ≈ 96.68, +0.11 over the recipe mean 96.57, which is +1.2 σ_mean(n=3) — exactly the regime that is statistically unresolvable from zero without n ≈ 20 runs. The data are equally consistent with "no effect" and "a true +0.1 fully absorbed below the bar." Either way the bar decides: the anti-aliased shortcut does not produce a ≥ +0.1pp improvement detectable at affordable n, and per pre-registration the question is **closed permanently** — further sampling would be variance mining.

Honesty note worth its own line: Run A alone (96.84) would have PASSED the standard single-run protocol. The pre-registered mean protocol correctly declined it — under H0 a best-of-45-nulls re-test producing one 96.84 draw out of two is far likelier than a true effect that vanishes to 96.56 in the very next byte-identical run. This is the strongest demonstration yet of why the project's single-run protocol at 6.7% false-positive WILL eventually hand over a false baseline if a near-bar read is ever accepted without replication.

Trajectory context: 46 consecutive non-improvements since EXP-006. Every single-mechanism class is now measured-closed AND the only positive-direction datum is resolved. The remaining unfalsified space is compound interventions of certified components and genuinely novel constructions passing all standing laws.

## Verification

- Integrity pre-condition: PASS both runs (windows ≤23.5 mean/none >27; epochs 138 ∈ [136,142]; steps within ~1% of family band; params 4,286,026; training 300.0s; evals 138 ≤ 138; family-shaped trajectories).
- Condition 1 (MEAN ≥ 96.81): FAIL — 96.70. First-failure-stop. Branch (ii) weak-positive-closed. The single-run fallback did not apply (Run B was obtained pristine); the max was not used.
- Conditions 2–3 (budget ≤600s; eval cadence): PASS both runs (informational).
- Trust review: summary blocks parsed from fresh per-run logs; no cache or dataset ambiguity; results plausible against the EXP-027 σ estimate. Verdict basis: valid result below bar → **no-improvement**.

## Unexplored Avenues

- n≈20 replication could resolve +0.1 vs 0 — explicitly unaffordable (≈3 hours GPU for a sub-bar answer); recorded as the resolution limit, not an avenue.
- Anti-aliasing the conv1 stride-2 path itself (blur before the 3×3 stride-2 conv, true BlurPool) was not tried — it changes the residual path (not just the shortcut), costs throughput (extra depthwise blur), and the deferral law (+1ms ≈ −0.08pp) prices the blur kernel's cost at roughly the size of the entire hypothesized effect. Low promise.
- Compound interventions: pairing the (free, possibly-+0.1) shortcut change with another free near-miss is the only construction in which this diff could still matter — but EXP-009 precedent shows certified-component compounding regressed; any such plan must argue mechanism-independence first.

## Next Steps

1. **Compound intervention of mechanism-independent free components** (medium confidence): the only unfalsified positive space; must pre-argue independence to escape the EXP-009 precedent.
2. **Radical constructions passing all laws** (low-medium): re-read train.py/papers for an angle outside every closed class — e.g., data-ordering/curriculum effects (never probed; zero throughput cost; not an augmentation, loss, LR, or structural change).
3. **Do not** re-sample the shortcut change, retry loss-axis or heat-axis variants, or revisit any class closed by the standing laws (high confidence in the exclusions).

## Key Learning

A pre-registered replicate-pair did its job both ways at once: it declined a bar-clearing single draw (96.84) that the standard protocol would have accepted, and it resolved the project's only positive-direction datum into "≤ +0.1 true effect, unresolvable and sub-bar." The deeper lesson: at σ ≈ 0.16 with a +0.1 bar, the project's detection floor (~+0.3 for one draw) is not a protocol inconvenience — it IS the frontier. Effects smaller than the floor are real possibilities on multiple closed axes, and chasing any of them individually is variance mining by construction; only mechanisms plausibly worth ≥ +0.3, or honest replication budgets, can move the baseline from here.
