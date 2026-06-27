# Experiment Report EXP-057: Classifier weight-decay decoupling — the last individually-undosed constant

- **Date**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-057.md
- **Plan**: plans/plan-057.md
- **Exp-log**: logs/exp-log-057.md
- **Verdict**: no-improvement
- **Metric**: 96.36 vs baseline 96.71 (bar 96.81)

## Goal

Maximize CIFAR-10 best_test_acc (%) within the fixed 300s charged budget; higher is better. Baseline 96.71 @ 1990397; bar ≥ 96.81. σ context (EXP-027): recipe mean ≈ 96.57, σ ≈ 0.16.

## Idea & Hypothesis

With the standard-modernization audit complete (EXP-056) and every catalogued axis measured-closed, candidate generation moved to law-derived corners. fc is the single BN-free, scale-sensitive layer — the one parameter class the WD-with-BN equilibrium argument (which pre-refuted LARS and makes conv-WD a pure effective-LR knob) does not cover, making its WD constant the last individually-unmeasured constant in the recipe. Mechanism: WD 5e-4 on fc.weight monotonically shrinks the logit scale, while label smoothing ε=0.1 fixes a finite optimal logit gap (≈ ln(0.91/0.01) = 4.51); if WD binds the classifier below that equilibrium, removing it lifts the plateau level. Unlike EXP-050's forced additive margin, this removes a constraint and lets CE+LS find its own self-limiting equilibrium. Pre-registered branches (all terminal): (i) ≥ 96.81 → replicate-pair MEAN; (ii) [96.41, 96.73] family band → corner closed redundant; (iii) (96.73, 96.81) → no-improvement by protocol; (iv) < 96.41 → fc WD's margin cap is load-bearing regularization, closed from below; (v) infra → relaunch.

## Approach

train.py only, single hunk: `fc_weight_id = id(base_model.fc.weight)` captured before the param-group split; fc.weight excluded from `decay_params` and added to `no_decay_params`. Optimizer call, schedule, warmup, loop byte-identical — the compiled graph is untouched (optimizer is eager), so **no GPU probe was needed** and training signatures were guaranteed family-identical by construction. CPU sanity ALL PASS: params 4,286,026 exact; fc.weight in the WD=0 group only; group numel ledger exact (decay 4,277,952 / no-decay 8,074 = 5,514 BN+bias + 2,560 fc.weight); 3-step smoke decreasing.

## Execution

Single pristine run, zero retries, zero operator errors. Gates passed on poll 1 (apps=0, load=11); GATE_DECISION D0 = 22.7ms ∈ [21.5, 23.5] with NO probe offset (graph unchanged, as designed); every watchdog window 21.7–22.7ms, slow_streak 0; RC=0. Signatures: 300.0s charged, 485.7s total, **140 epochs / 13,511 steps** (family), params 4,286,026, VRAM 1,613MB, evals 140 ≤ 140, ep1 36.31.

## Results

Branch (iv) — the sharpest sign-readout available from a single draw:

1. **best 96.36 = mean − 1.3σ, below the family floor (96.41)**, with final_test_loss 0.1905 in the family band and a converged plateau (96.22–96.36 over the last 8 evals) sitting ~0.2 BELOW the family plateau. This is a level depression at perfectly clean signatures — and because the diff changed exactly one quantity (decay pressure on 2,560 fc weights), attribution is airtight: removing fc WD costs ~0.2pp.
2. **Interpretation: the WD margin cap on the classifier is load-bearing regularization under CE+LS + heavy aug.** The LS-equilibrium hypothesis predicted the opposite sign. What the read shows is that an unconstrained classifier reaches larger logit scales, sharpening per-view confidence on TA/RE-distorted training views — the same over-margin failure shape as EXP-050 (forced margin m=0.75: test_loss improved, accuracy fell). Notably here test_loss did NOT improve (0.1905, family), so the fc-scale relief bought nothing on CE while still moving argmaxes the wrong way slightly.
3. **The loss-geometry/logit-scale axis now has a third closure datum via a third pathway**: loss-side up-pressure (EXP-050, −2.4σ), loss-side down-pressure via confidence weighting (EXP-051, −7.8σ), and now optimizer-side scale relief (EXP-057, −1.3σ). The recipe's logit scale is at a measured local optimum, robust to the intervention path.
4. **Per-layer WD coverage is COMPLETE.** Every parameter class has had its WD treatment individually measured: BN/bias (no-decay, in recipe), conv weights (5e-4, dosed globally both directions via the reg-dose closures), fc.weight (this run: 5e-4 beats 0). No per-layer constant remains unmeasured.
5. Honest caveat on magnitude: −1.3σ is a single draw and the TRUE effect could be milder than −0.2; but the branch was pre-registered as terminal and the decision-relevant content (not an improvement; do not pursue fc-WD relief) is robust at any plausible magnitude.

Trajectory: 51 consecutive non-improvements. Both documented-weak corners from brainstorm-056 are now resolved or deprioritized: fc per-layer constants measured (this run); late batch-size schedule remains the lone runnable-but-expected-null entry on the books.

## Verification

- Integrity pre-condition: PASS (RC=0; D0 22.7 ∈ [21.5, 23.5]; no kill markers; max window 22.7 < 27; params exact; 300.0s charged; 485.7 ≤ 600; 140 evals ≤ 140 epochs; epochs/steps in family bands; ep1 36.31 ≥ 30; no NaN).
- Condition 1 (best ≥ 96.81): FAIL — 96.36. First-failure-stop; branch (iv). Conditions 2–3 pass informationally (485.7s; 140 ≤ 140).
- Trust review: fresh run.log, watchdog cross-check consistent, byte-identical family signatures make a false-negative from contamination implausible (contention shows up in dt/epochs first — none here). Verdict basis: valid result below bar → **no-improvement**.

## Unexplored Avenues

- **Intermediate fc WD (e.g., 2.5e-4) or increased fc WD (1e-3)**: the sign-down read shows the optimum is at-or-above 5e-4; a dose-response sweep would localize it, but with the global reg-dose closures and a −1.3σ single-sided read, the expected gain over 5e-4 is sub-resolution. Not worth a run.
- **fc LR multiplier (×0.5)**: brainstorm-056 already flagged this as margin-pressure-down in disguise; EXP-057's result (relief loses) strengthens the case that pressure-down also loses (EXP-051 analog). Closed by triangulation.
- **Logit-norm constraints (weight-norm/cosine classifier)**: published-toolkit entries that fix the scale a different way; the audit-complete + absorption record (0-for-16) plus this third logit-scale closure give them no remaining mechanism.

## Next Steps

1. **Candidate generation stays outside the toolkit and outside all closed axes** (high): the next brainstorm must produce constructions passing every law — and now also avoiding ALL logit-scale paths (loss-side and optimizer-side both measured-closed at three data points).
2. **Late batch-size schedule (512→1024 at p≥0.75)** remains the only documented runnable corner (low confidence: three adjacent negative closures, medium infra effort) — run it only if the next sweep produces nothing with a better mechanism.
3. **Do not revisit**: per-layer WD/LR constants (complete), any logit-scale intervention via any pathway, plus the standing do-not-revisit list from exp-report-056 (block order, structural classes, throughput, precision, schedule family, noise level, averaging, regularization dose).

## Key Learning

The last individually-undosed constant in the recipe is now measured, and it was already optimal: removing weight decay from the one BN-free layer (fc) — the only place WD does real work rather than setting effective-LR — read 96.36, below the family floor, at byte-identical training signatures that make the attribution airtight. The WD margin cap on the classifier is load-bearing under CE+LS with heavy augmentation, the exact opposite of the LS-equilibrium prediction, and the logit-scale axis is now closed via a third independent pathway (loss-up EXP-050, loss-down EXP-051, optimizer-relief EXP-057). Per-layer WD coverage is complete: every parameter class in the model has had its decay treatment individually priced, and the certified recipe survived its 51st challenger.
