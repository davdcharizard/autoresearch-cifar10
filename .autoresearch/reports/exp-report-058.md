# Experiment Report EXP-058: Classifier WD ×4 — the fc-decay axis bracketed and closed at the default

- **Date**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-058.md
- **Plan**: plans/plan-058.md
- **Exp-log**: logs/exp-log-058.md
- **Verdict**: no-improvement
- **Metric**: 96.24 vs baseline 96.71 (bar 96.81)

## Goal

Maximize CIFAR-10 best_test_acc (%) within the fixed 300s charged budget; higher is better. Baseline 96.71 @ 1990397; bar ≥ 96.81. σ context (EXP-027): recipe mean ≈ 96.57, σ ≈ 0.16.

## Idea & Hypothesis

EXP-057 produced the first measured directional slope in dozens of loops: fc decay pressure 0 → 5e-4 gains ~0.2pp (the classifier's WD margin cap is load-bearing under CE+LS + heavy aug). This experiment tested the unmeasured side of that axis: fc.weight in its own param group at WD 2e-3 (×4), conv weights at 5e-4, BN/bias at 0. Hypothesis: if the cap optimum lies above the default, the plateau level rises (≥ 96.81 → replicate-pair). Pre-registered branches: (i) ≥ 96.81 → pair escalation; (ii) family band → slope saturates, axis closed flat; (iii) (96.73, 96.81) → no-improvement by protocol; (iv) < 96.41 → over-constrained, optimum bracketed in (0, 2e-3) with 5e-4 the measured best; (v) infra → relaunch.

## Approach

train.py only, two hunks: `FC_WEIGHT_DECAY = 2e-3` constant + three-group optimizer split (conv 4,277,952 params @ 5e-4 / fc.weight 2,560 @ 2e-3 / BN+bias 5,514 @ 0 — ledger asserted exact in CPU sanity, all four checks PASS). Graph/loop byte-identical; no GPU probe per the EXP-057-validated optimizer-only diff class.

## Execution

Two runs. **Run 1 CONTAMINATED**: a foreign GPU-0 job time-sliced the run in episodes below the 4-window kill streak (windows 28.0/27.4/30.0ms; raw dt samples to 95ms), costing ~500 steps (12,916 vs band ≥ 13,300) — integrity pre-condition correctly failed on the step ledger and the run was discarded unanalyzed (its 96.54 read is not evidence; recorded in Errors & Dead Ends). **Run 2 (decision run)**: the gate held launch ~64 min until the foreign job (observed directly at 99–100% util) exited, then ran clean — D0 22.5ms, single transient 25.7ms window, slow_streak 0, RC=0; 300.0s charged, 497.2s total, 138 epochs / 13,322 steps (in band), params exact, evals 138 ≤ 138, ep1 34.94.

## Results

Branch (iv) — the axis is now bracketed on both sides of the default:

1. **best 96.24 = mean − 2.1σ** with family-band test_loss (0.1874) and a depressed converged plateau (96.18–96.24) — the same level-depression-at-family-CE signature as EXP-057's relief direction, now from the opposite side.
2. **The fc-decay axis has a measured interior maximum AT the default**: λ_fc = 0 → 96.36 (−1.3σ); λ_fc = 5e-4 → family mean; λ_fc = 2e-3 → 96.24 (−2.1σ). Both deviations lose with the same signature (accuracy down, CE flat), so the cap optimum sits in the vicinity of 5e-4 and the curvature is real on both sides. The hypothesis that the EXP-057 slope continued upward is refuted; the dose-response is a peak, not a ramp.
3. **Mechanistic reading**: too little cap → over-confident per-view logits hurt argmaxes (EXP-057); too much cap → the classifier is norm-starved and cannot expreß the margins the features support — both ends move argmaxes without moving CE, the recurring decoupling signature (test-CE and test-accuracy decouple, goal-learnings loss-axis entry, now count 4 data points across paths).
4. The interesting-but-inadmissible footnote: contaminated Run 1 read 96.54 at −500 steps. Per protocol it is not evidence (single contaminated draw); the clean Run 2 at full steps is the decision read, and at −2.1σ it is decisive for the branch.
5. **Process note**: the watchdog's kill streak is intentionally conservative; the post-hoc step-ledger gate (steps < family band ⇒ contaminated) is the binding instrument for sub-streak contention — it caught what the live watchdog let pass, exactly as designed after EXP-011/048.

Trajectory: 52 consecutive non-improvements. The EXP-057→058 pair is the program's first completed two-sided dose-response bracket initiated from a measured slope — and it confirms the certified recipe value was already optimal, extending the pattern that EVERY recipe constant sits at a measured local optimum (heat, noise, BN momentum, batch, LS-bearing constants, now fc WD).

## Verification

- Integrity pre-condition: PASS on Run 2 (D0 22.5; max window 25.7 < 27; steps 13,322 ∈ [13,300, 13,600]; epochs 138; 300.0s; 497.2 ≤ 600; 138 evals ≤ 138; ep1 34.94; no NaN). Run 1 failed integrity and was relaunched byte-identically per branch (v), never analyzed.
- Condition 1 (best ≥ 96.81): FAIL — 96.24. First-failure-stop; branch (iv). Conditions 2–3 pass informationally.
- Trust review: fresh run.log, watchdog cross-check consistent, family signatures on the decision run; the contaminated run was excluded by a pre-registered, mechanical rule (step ledger), not by discretion. Verdict basis: valid result below bar → **no-improvement**.

## Unexplored Avenues

- **Interior doses (e.g., 1e-3) or finer bracketing**: with both ±4× endpoints reading ≤ −1.3σ and the default at the mean, any interior gain over 5e-4 is bounded well below one-draw resolution. Closed by bracketing arithmetic (same logic as the BN-momentum closure, EXP-038/039).
- **fc LR multiplier**: both directions now carry direct negative evidence by proxy (cap-relief and cap-tightening both lose through the WD path; the LR path adds head-lag entanglement on top). Closed by triangulation.
- **Other per-layer constants**: none remain — fc was the only layer outside the BN equilibrium argument, and its axis is now measured on both sides.

## Next Steps

1. **The brainstorm frontier is again construction-only** (high): with the fc-WD bracket closed, no candidate anywhere on the books carries a measured positive prior; the next sweep must generate law-passing constructions outside every closed axis, weighting closure-value heavily (exp-report-056/057 framing stands).
2. **Late batch-size schedule 512→1024 at p≥0.75** remains the single documented runnable corner (low confidence: three adjacent negative closures, medium infra effort) — now arguably the default next run absent a better construction, since the books hold nothing else.
3. **Do not revisit**: fc decay/LR in any direction (bracketed), any logit-scale path, plus the standing list (structure/order, throughput, precision, schedule family, noise level, averaging, regularization dose, per-layer constants).

## Key Learning

The program's first slope-initiated dose-response bracket closed in one pair of runs: the classifier WD axis has a measured interior maximum at the shipped default (0 → −1.3σ, 5e-4 → mean, 2e-3 → −2.1σ), with both deviations showing the same accuracy-down/CE-flat decoupling signature from opposite mechanisms (over-confidence vs norm starvation). The deeper pattern is now unbroken across every recipe constant ever dosed — heat, noise, batch, BN momentum, and now fc decay all sit at measured local optima — meaning the certified recipe is not just unbeaten but locally optimal in every measured coordinate. Operationally, the run also validated the step-ledger as the binding contention instrument: a foreign job's sub-kill-streak time-slicing (~500 stolen steps, invisible to the live watchdog) was caught only by the post-hoc step-count integrity gate.
