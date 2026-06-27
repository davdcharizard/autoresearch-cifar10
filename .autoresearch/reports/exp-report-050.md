# Report EXP-050: Additive logit margin on the true class (MARGIN = 0.75, training-loss-only)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-050.md
- **Plan**: plans/plan-050.md
- **Log**: logs/exp-log-050.md

## Goal

Maximize CIFAR-10 best_test_acc (%, higher is better) within the fixed 300s charged budget by modifying train.py only. Baseline: **96.71** @ 1990397; bar ≥ 96.81; mean ≈ 96.57, σ ≈ 0.16 (EXP-027).

## Idea & Hypothesis

With the recipe-constant audit complete (EXP-049), the first probe of the last zero-cost unmeasured class: loss geometry. Anchor: the project's own EXP-011/032 insight that the accuracy ceiling is decision-boundary-limited. Chosen idea: subtract MARGIN=0.75 from the true-class logit in the training loss (AM-softmax-style, training-loss-only; eval untouched), shifting the LS-converged logit-gap optimum from 4.51 to 5.26 so converged boundaries sit further from training points. Hypothesis: best ≥ 96.81 if boundary placement is the binding limitation; branch (ii) mean band = absorbed; branch (iii) < 96.42 = margin harmful under heavy aug (destroyed-label amplification).

## Approach

Three edits to train.py: `MARGIN = 0.75` constant; timed-loop loss subtracts `MARGIN * F.one_hot(targets, NUM_CLASSES).to(outputs.dtype)` from outputs; warmup loss mirrored for compiled-graph identity. Diff: 1 file, +8/−2. CPU sanity validated the mechanism numerically before launch: loss-vs-gap argmin shifts 4.512 → 5.262 (+0.750 exactly); at gap = plain-optimum + 0.09 the plain CE+LS gradient pushes the gap DOWN (+0.007) while the margin loss pushes it UP (−0.071); params 4,286,026; m=0 identity; 6-step smoke monotone. One sanity-script lesson (not a code deviation): with LS, the margin LOWERS the loss value at gaps far above the optimum — the meaningful invariant is the argmin shift, not loss-value ordering (recorded in exp-log Surprises).

## Execution

Single pristine run: gates clear poll 1; GATE_DECISION D0=22.7ms; 30 post-gate windows all 22.0–22.7ms, slow_streak 0; RC=0; 139 epochs / 13,431 steps / 300.0s charged / 485.6s total. No retries, no kills, no errors.

## Results

- **Primary metric**: best_test_acc 96.19 (baseline: 96.71, delta: −0.52, −0.54%)
- **Observations**: 96.19 = mean − 2.4σ with a uniformly depressed converged-flat plateau (96.05–96.19 last 8) — an active negative, same shape class as EXP-047, not noise. The decisive mechanistic datum: **final_test_loss 0.1505 vs family ~0.185** — the margin achieved exactly what it was designed to do in logit space (wider gaps → higher p_true → test CE improved by ~19%) while accuracy FELL 0.4pp below the recipe mean. Step ledger 13,431 = family; the op was throughput-free as projected.
- **Analysis**: Pre-registered branch (iii), and the cleanest mechanism separation the project has produced: the experiment moved the exact quantity it targeted (test cross-entropy, i.e., margin/confidence geometry) in the intended direction, and the target metric moved the OPPOSITE way. This is the inverse face of the EXP-011/032 signature (there: smoothing improved loss, accuracy unchanged; here: margin improved loss, accuracy dropped) and together they bound the loss-geometry class from both sides: test-CE-improving interventions are at best accuracy-neutral and at worst accuracy-negative. Root cause reading: under TrivialAugment+RandomErasing, a meaningful fraction of training samples carry effectively-wrong labels; uniform margin pressure forces the network to push THOSE samples' wrong-label gaps past 5.26 too, and the capacity spent memorizing margin on destroyed samples is taken from boundary placement on genuine ones. The accuracy ceiling is decision-boundary-limited (confirmed again), but static loss-side pressure cannot move boundaries in the right direction — it cannot tell genuine boundary samples from destroyed ones. Any future loss-geometry candidate must be selective, not uniform.
- **Key Learning**: Test-CE and test-accuracy decouple HARD at this recipe: a margin that improved test_loss 19% cost −2.4σ accuracy. Uniform loss-side pressure cannot distinguish genuine boundary samples from augmentation-destroyed ones; the loss-geometry class is closed for static/uniform forms.

## Verification

- **Conditions**: Integrity pre-condition PASSED (pristine windows 22.0–22.7ms; 139 epochs; 13,431 steps in family ledger; params exact; 300.0s; 139 evals ≤ 139; trajectory-criterion numerics normal — the test_loss shift is the designed effect and was pre-flagged informational in the plan). Condition 1 (best ≥ 96.81) FAILED: 96.19, below the replicate band. Conditions 2–3 skipped per first-failure-stop (informationally both pass: 485.6s ≤ 600; 139 ≤ 139).
- **Review Notes**: Results trustworthy — clean profile, family signatures to the step, and the loss/accuracy divergence is itself strong evidence the intervention did what it claimed (no silent wiring failure: a no-op would have shown family test_loss). Eval untouched; no integrity concerns.
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure — valid active-negative result (pre-registered branch iii).

## Unexplored Avenues

- **Selective margin (boundary band-pass dosing)**: brainstorm-050 Idea 2 — weight margin/loss pressure by p(1−p) so destroyed-label samples (p→0) are excluded from the push. This is the one form the destroyed-label root cause does NOT discredit; but the dose-128 active negative here means any follow-up must justify why selectivity flips the SIGN, not just attenuates the damage (EXP-047 precedent: smaller doses of a −2.4σ mechanism are implausible bar-clearers). Low prior.
- **Margin dose reduction (m=0.2–0.4)**: discredited by the same sign logic — attenuating a negative mechanism approaches zero from below.
- **Eval-side margin/temperature**: forbidden territory (Eval.evaluate is ground truth; logit rescaling at eval would be measurement manipulation, and argmax is temperature-invariant anyway).

## Next Steps

1. **Treat loss-geometry as closed (static/uniform forms) and update the frontier map** (high confidence): the unfalsified set is now: selective per-sample signals (low prior, sign-flip burden), compound interventions of certified components (EXP-009 precedent against), and qualitatively different training signals that pass the deferral law (none currently constructible — EXP-048 bounded the charged step).
2. **Carry the CE/accuracy decoupling datum into all future evaluation reasoning** (high confidence): test_loss is NOT a proxy for the goal metric at this recipe — candidates motivated by loss improvements (distillation, calibration, smoothing) inherit a measured negative-to-null prior.
3. **Protocol stack unchanged; sanity-script invariant lesson recorded** (medium confidence): when LS is present, loss-VALUE orderings are unintuitive — test mechanism invariants (argmin shifts, gradient signs), not value comparisons.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
