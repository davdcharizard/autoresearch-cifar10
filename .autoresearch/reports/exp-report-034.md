# Report EXP-034: Later/shorter augmentation cooldown (COOLDOWN_FRAC 0.15 → 0.10)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-034.md
- **Plan**: plans/plan-034.md
- **Log**: logs/exp-log-034.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) within the fixed 300s budget, editing only `train.py`. Baseline = **96.22%** (EXP-012, commit 6c417a4); pass bar = **96.32%**.

## Idea & Hypothesis
Refinement of EXP-033's augmentation cooldown. EXP-033 (COOLDOWN_FRAC 0.15, cooldown start frac 0.85) reached 96.10 — its shortfall was starting too early, sacrificing ~5 epochs of productive strong-aug training (clean fine-tune started from a low 95.43 base). Hypothesis: starting the cooldown LATER (COOLDOWN_FRAC 0.10, start frac 0.90) preserves more strong-aug training so the clean fine-tune lifts from a higher base, raising best_test_acc above 96.10 and plausibly clearing 96.32, at unchanged ~91 ep / dt 8ms / 4,299,866 params.

## Approach
Single-variable change vs EXP-033: re-applied the identical four-edit cooldown code (2nd CPU transform `train_tf_clean` = full pipeline minus TrivialAugment; `aug_cooled` flag + epoch-boundary `train_set.transform` swap with an observable marker; Cutout gated behind the flag) with `COOLDOWN_FRAC = 0.10` instead of 0.15. Everything else unchanged. Smoke test confirmed params 4,299,866, COOLDOWN_FRAC 0.10, clean transforms.

## Execution
Single run, exit 0 in 405s wall (300s training). Cooldown marker fired once at `ep 83 frac 0.91` (as planned, later than EXP-033's ep77/0.85). dt steady ~8ms, no NaN, no errors. 91 epochs (throughput-neutral).

## Results

- **Primary metric**: best_test_acc = **96.26%** @ ep87 (baseline 96.22%, delta **+0.04pp** — within the ±0.2pp noise floor; **+0.16pp vs EXP-033's 96.10**); bar 96.32 NOT cleared (−0.06pp).
- **Observations**: The mechanism hypothesis was CONFIRMED — the pre-cooldown base was 96.05 at ep83 (vs EXP-033's 95.43 at ep77), directly showing that retaining more strong-aug training yields a higher base. Post-cooldown: ep84 96.20 → ep87 96.26 (peak, ~4 clean epochs in) → mild decline to 96.15. final_test_loss 0.1951 ≈ baseline 0.195 (better than EXP-033's 0.2000). num_epochs 91, dt 8ms, params 4,299,866.
- **Analysis**: The later/shorter cooldown is clearly better than the early/long one (+0.16pp over EXP-033), validating the "preserve strong-aug training, fine-tune from a higher base" reasoning. BUT the result only matches baseline within noise (96.26 vs 96.22) and does not clear the +0.1 bar. The decisive quantitative read is the **marginal effect over a full-aug-to-the-end run**: a normal cosine tail from the ep83 base (96.05) reaches ~baseline 96.22 on its own; the clean cooldown added only ~+0.04 on top (to 96.26). So the cooldown's true contribution is small and noise-dominated — the net is at its capacity ceiling and the clean-distribution-alignment lever, while real (it improved loss and gave the project's first ≥baseline cooldown result), cannot supply the +0.1 needed. Fits the firmly-established generalization-bound-at-fixed-capacity plateau.
- **Key Learning**: A later/shorter augmentation cooldown (0.10, start frac 0.90) beats the early/long one (0.15) by +0.16pp by fine-tuning from a higher pre-cooldown base (96.05 vs 95.43), reaching 96.26 ≈ baseline + baseline-quality loss — but its marginal lift over a full-aug cosine tail is only ~+0.04pp, so the cooldown matches the plateau rather than breaking it.

## Verification

- **Conditions**: Cond 1 (≥96.32) **FAILED** — 96.26 < 96.32 (+0.04 over baseline, within noise). Cond 2 (clean, <600s, 0 Traceback) passed. Cond 3 (only train.py; params 4,299,866; eval-count 91 == epochs; core torch; seed 42) passed.
- **Review Notes**: Trustworthy. Cooldown fired once at the correct later fraction (ep83/0.91), throughput-neutral (91 ep, dt 8ms) → clean fair test. Intervention is on the augmentation schedule (intended class), not a measurement-gap exploit.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid, trustworthy result; primary verification condition (clear the bar) failed by 0.06pp; +0.04 over baseline is within the noise floor; no constraint violated.

## Unexplored Avenues
- **Even shorter cooldown (COOLDOWN_FRAC ≈ 0.05-0.07)** — the trend (0.15→96.10, 0.10→96.26) is monotone up as the window shrinks, so there may be a shallow optimum around 0.06-0.08. BUT the marginal-over-full-aug analysis caps the expected gain (the cooldown adds only ~+0.04 at 0.10, and shrinking the window toward 0 converges to baseline), so a bar-clear is LOW confidence. Worth ONE probe to bracket the optimum, then close the axis.
- **Cooldown + a complementary clean-tail lever** — e.g. a tiny LR bump or BN-only recalibration during the clean phase. Speculative; most complementary levers are in closed/polish axes.
- The drop-only-TA (keep Cutout) variant from brainstorm-034 remains untried but is now lower-value given the 0.10 result already matches baseline.

## Next Steps
- **EXP-035: shorter cooldown COOLDOWN_FRAC ≈ 0.07** — confidence: low-medium that it improves on 96.26, LOW that it clears 96.32. Brackets the cooldown optimum; the natural continuation of a positive trend and cheap.
- **If EXP-035 also lands ~96.25-96.30, close the augmentation-schedule axis** and treat 96.22 as the confirmed k=4/300s ceiling (now ~26 axes mapped). Confidence: high that the plateau is real.
- Beyond cooldown, no compute-positive or generalization lever remains untried that isn't in a closed/polish axis; the honest call is converging on "plateau is the ceiling."

## Exit Action Results
<!-- No exit actions defined for this goal. -->
