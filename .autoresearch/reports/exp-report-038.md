# Experiment Report: EXP-038 — BN running-stat momentum 0.1 → 0.02 (last unmeasured implicit constant)

- **Date**: 2026-06-10
- **Verdict**: no-improvement
- **Primary metric**: best_test_acc = **96.27%** (baseline 96.71, bar 96.81, delta −0.44; below the baseline band 96.4–96.7 — genuine negative effect)
- **Branch**: autoresearch/exp-038 (discarded)
- **Artifacts**: brainstorm/brainstorm-038.md · plans/plan-038.md · logs/exp-log-038.md

## Goal
Maximize CIFAR-10 test accuracy (best_test_acc %, higher is better) within the fixed 300s charged training budget, modifying only `train.py`. Baseline 96.71 @ 1990397; bar ≥ 96.81. σ context (EXP-027): baseline mean ≈96.57, σ ≈0.16.

## Idea & Hypothesis
**Idea**: The EXP-036 audit method applied one level deeper found the last unmeasured constant — an implicit one: BN running-stat momentum (PyTorch default 0.1, never set in train.py). In-regime evidence (EXP-029: eval accuracy routes through BN constants at −10.9 sensitivity) says these constants are load-bearing; the default ~10-batch EMA horizon is a high-variance estimator of them. Smoothing to a ~50-batch horizon (momentum 0.02) should cut estimator variance ~5× at the converged plateau, where weights are quasi-static and the max-statistic is taken — at exactly zero cost in every closed currency.

**Hypothesis**: Smoother constants raise the plateau LEVEL (bar-pass if BN-stat noise costs the mean ≥ +0.25); falsified by a plateau within the baseline band. Predicted diagnostic either way: hot-phase eval depression (lag signature).

## Approach
train.py only (4 insertions / 3 deletions): new constant `BN_MOMENTUM = 0.02` passed to all three `nn.BatchNorm2d` construction sites → all 19 BN layers verified at 0.02 (CPU module walk); params 4,286,026 unchanged; momentum set at construction only (no runtime mutation → no compile-guard risk). Training path byte-identical — weights, gradients, schedule, noise untouched; only the stat-buffer EMA coefficient differs.

## Execution
Single pristine run (GATES_CLEAR poll 1; launched 19:41:52; rc=0; total 499.3s; no watchdog trigger). Signatures exactly baseline: 267 profile windows mean 22.3ms with 0 >27ms, 139 epochs / 13,429 steps, VRAM 1613.0MB. No retries, no adjustments, no errors.

## Results
- **best 96.27, final 96.20, final_test_loss 0.1917 — below the baseline band.** The hypothesis is not just falsified but **inverted**: smoothing the estimator made the constants WORSE.
- **Lag dominated variance at every phase.** Hot phase: ep5 eval 35.30 (family ~64) — accuracy DROPPED from ep3's 50.92 while training progressed normally; ep20 65.83 vs ~79. With LR hot, a 50-batch-old activation-statistics mixture describes a materially different network than the current weights. Plateau: last-15 mean 96.022, spread 0.64 (family ~96.5, spread ~0.15) — the prediction said scatter would SHRINK; it grew 4×. Root cause: the "converged plateau" is not static — the cosine tail keeps drifting weights (the same endgame learning EXP-033 showed is load-bearing), so stale constants are systematically misaligned right where the max is harvested. test_loss 0.1917 vs family ~0.185 corroborates.
- **The dial is live with the OPPOSITE sign**, which is itself the most informative null variant: BN constants must track the CURRENT weights; freshness (short horizon) is the operative property, and the default 0.1 sits at-or-near optimum from below. Combined with EXP-029 (constants must come from the augmented TRAINING distribution, not clean data), the picture completes: BN running stats must be (a) estimated on the training distribution and (b) recent — both "improvements" away from the default damage the calibration.
- **Trajectory fit**: 33rd consecutive miss, but unlike the three exact-deficit nulls before it, this is a measured NEGATIVE with a clean mechanism — it closes the implicit-constant audit the same way EXP-015 closed WD (incumbent at optimum, both neighbors worse or flat).

## Verification
First-failure-stop per plan-038. Pre-condition: profile pristine (mean 22.3ms, 0 slow >27 — no quantization ambiguity at this dt per the EXP-037 protocol note), 139 epochs ✓. Integrity: params 4,286,026 ✓, training_seconds 300.0 ✓, eval_lines 139 = num_epochs ✓. **Condition 1 FAILED on merits: 96.27 < 96.81.** Conditions 2–3 skipped per protocol (incidental: rc=0, 499.3s ≤ 600; 139 = 139). No false-failure risk: the deficit has a coherent mechanism signature across hot phase, plateau statistics, and test_loss. Verdict: **no-improvement**.

## Unexplored Avenues
- **Momentum ABOVE 0.1 (0.2–0.3, fresher constants)**: the inverted read makes this direction logically open — but the damage at 0.02 came from lag, and 0.1's ~10-batch horizon is already fresh relative to per-epoch weight drift; the variance cost of a ~3-batch horizon (m=0.3) would likely exceed the marginal freshness gain. Expected ±0.1 at best; only worth a slot if no better instrument exists.
- **Precise BN re-estimation immediately before each eval on the AUGMENTED stream** (combining EXP-029's distribution lesson with this freshness lesson): would cost charged forward passes (~1s/eval × 139) — pure deferral arithmetic kills it (~−6 epochs minimum), and SWA's augmented-loader update_bn (EXP-032) already measured the ceiling of better-estimated constants at ≈ 0 gain.
- The audit method itself (find unmeasured dials) is now exhausted at BOTH explicit and implicit levels — no further unmeasured constants exist in train.py's semantic surface.

## Next Steps
1. **Record the BN-constants law as complete** (distribution: EXP-029; freshness: EXP-038) and treat all eval-constants engineering as closed (high confidence).
2. **The program's measured frontier is now fully closed on every axis** — next loops must either find a mechanism with in-regime evidence (none currently known) or accept that remaining bar-clearance probability rides on interventions yet to be invented; continue the loop with the strongest available screen: in-regime evidence + free-in-all-currencies + ≥ +0.3 plausible effect (high confidence in the framing, low in any specific candidate).
3. **If an idea drought persists**, the directive's escalation path (re-read papers referenced in code/knowledge, recombine near-misses, more radical architecture) remains — with the caveat that radical architecture must satisfy the dt law's whole-block pricing (2.5ms/block) which gate-kills most reshapes a priori (medium confidence).

## Key Learning
BN running statistics must be FRESH, not smooth: a 5×-longer EMA horizon lagged the drifting weights at every phase — including the "converged" plateau, which still drifts enough (cosine-tail endgame learning) that 50-batch-old constants cost −0.3 and 4× plateau scatter. With EXP-029 (must be the augmented training distribution) this completes the BN-constants law: the framework default (recent, train-distribution) is at the joint optimum, and the implicit-constant audit closes with every dial in train.py — explicit and implicit — measured.
