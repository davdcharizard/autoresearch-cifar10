# Experiment Report EXP-063: Augmentation cooldown @0.10 on the AugMix-p0.5 best recipe

- **Date**: 2026-06-09
- **Verdict**: no-improvement
- **Primary metric**: best_test_acc = **96.31%** (baseline 96.45, bar 96.55; delta **−0.14pp**)

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) by editing only `train.py`, within the fixed 300s Σdt GPU-time budget on a single H20 (≤600s wall). Baseline 96.45 (EXP-054); bar = 96.55.

## Idea & Hypothesis
**Chosen idea**: Port the EXP-034 best cooldown (disable train augmentation for the final 10% of the time budget, keeping only crop+flip) onto the current AugMix-p0.5 best recipe. **Rationale**: cooldown is the one documented near-miss (EXP-033/034/035, best 96.26) never combined with the current best — all four prior cooldown runs were on the superseded TrivialAugment recipe. Its mechanism (joint weight+BN re-adaptation to the clean eval distribution in the tail) is distinct from the closed augmentation-strength/policy axis, and distinct from BN-recalib-alone which hurts (EXP-061). **Hypothesis**: the clean-tail re-adaptation would produce a tail-climb raising best_test_acc to ≥96.55; honest prior expectation was a near-noise null (cooldown's net lift on TA was within-noise; the 50%-subset AugMix has a smaller train→eval gap than full-coverage TA).

## Approach
Three edits to train.py (all else byte-identical to EXP-054): (1) `COOLDOWN_FRAC = 0.10`; (2) a clean `cooldown_tf` (crop+flip+ToTensor+Normalize) and second `cooldown_loader`; (3) per-epoch gate `cooldown_active = (total_training_time/TIME_BUDGET_S) >= 0.90` selecting the clean loader and skipping GPU Cutout, plus a one-time `>> AUG COOLDOWN fired` marker. The Cutout gate is a host-level Python branch outside the compiled forward (input shape/dtype unchanged) → cudagraph-safe, no recompile (verified: dt stayed 8ms).

## Execution
One run, GPU 1 (idle; GPU 0 had foreign proc PID 1314331). Exit 0 in 574.1s wall (faster than EXP-054's 593s — cooldown lightens the tail's CPU augmentation, exactly as predicted), 91 epochs, 35323 steps. dt 608×8ms + 96×9ms + 1×10ms + 1×30ms (compile warmup) — uncontended, throughput identical to EXP-054. 0 NaN/error. Cooldown fired ep83/frac0.907 (marker present). No retries or adjustments.

## Results
best_test_acc 96.31% — a **−0.14pp regression** vs baseline 96.45, missing the bar by 0.24pp. The cooldown mechanism **worked exactly as documented**: the per-epoch eval trace shows a clean +0.22 tail-climb after cooldown fired — 96.09 (ep82, pre-cooldown) → 96.11 (ep83) → 96.27 (ep84) → 96.31 (ep86), then a flat plateau ep86-91. This matches EXP-034's +0.21 climb on TA almost exactly. The failure is NOT the mechanism; it is the **base**: the pre-cooldown trajectory at ep82 was only 96.09, well below where EXP-054's uninterrupted full-augmentation trajectory reaches by end-of-budget (96.45). Sacrificing the final ~9 epochs of augmented training (which on the AugMix recipe continue to refine the model) for a clean re-adaptation that climbs only to 96.31 is net-negative. final_test_loss 0.2011 > EXP-054's 0.1968 — the clean tail did not even lower loss, confirming the augmented training those epochs would have done was more valuable than the distribution-match. This is the SAME structural outcome cooldown produced on TA (EXP-034: real climb, net below the no-cooldown baseline): cooldown trades end-of-run augmented training for a clean-tail climb, and on a well-tuned diverse-aug recipe that trade does not pay. This is the 64th experiment / 10th lever-perturbation to leave the 96.45 ceiling intact, and closes cooldown-on-AugMix.

## Verification
- **Necessary condition 1 — `best_test_acc >= 96.55`**: 96.31 < 96.55 → **FAILED**. Stopped at first failed condition.
- Conditions 2 (in-budget) and 3 (no hard-constraint violation) not formally evaluated (aborted), but both would pass: total_seconds 574.1 < 600, num_params 4,299,866, num_epochs 91, 0 NaN, git diff == train.py only, cooldown-fired marker present (mechanism genuinely engaged, not a silent no-op), uncontended 8ms dt.
- **Trustworthiness**: fully trustworthy — clean uncontended run, throughput matched the reference, the cooldown marker + the textbook tail-climb signature confirm the intended mechanism engaged and behaved as documented. No false-failure / false-pass / integrity concerns.
- **Verdict basis**: valid in-budget run that missed the bar → **no-improvement**.

## Unexplored Avenues
- **Shorter cooldown (@0.05) on AugMix**: a shorter clean tail would sacrifice fewer augmented epochs but also give a smaller climb; given the @0.10 base was already below baseline, this is unlikely to clear the bar — low confidence.
- **Cooldown of AugMix ONLY (keep Cutout in the tail)**: a softer cooldown that keeps the cheap GPU Cutout regularizer while removing AugMix. But Cutout is also a train→eval gap source, so this dilutes the mechanism — low confidence.
- The cooldown FAMILY (15%, 10%, 10%+LR-reheat, 10%+GC, and now 10%-on-AugMix) is now exhausted across both recipes — all net-negative. Do not revisit.

## Next Steps
1. **Gradient-norm clipping at a permissive threshold** (low confidence) — the one remaining untested optimizer-adjacent scalar (brainstorm-063 Idea 2); likely a null on a stably-training net but cheap to close.
2. **BN momentum tuning** (low confidence) — untested, but EXP-061 weakly argues against (augmented running stats are the correct operating point).
3. **Accept the scalar/recipe space is exhausted; attempt a genuinely radical architectural change** (per the NEVER-STOP "think harder" directive) — every scalar, schedule, augmentation, capacity, optimizer, and now both near-miss-combinations (EXP-049 cooldown+GC, EXP-063 cooldown-on-AugMix) are closed. The remaining headroom, if any, is a structural idea not yet tried at this budget (e.g. a different residual topology or a compute-neutral attention/gating variant that doesn't add dt) — medium-effort, low-confidence, but the only unexplored class left.

## Key Learning
Aug cooldown produced its documented +0.22 clean-tail climb on the AugMix recipe (matching EXP-034) but climbed from a pre-cooldown base (96.09) below where uninterrupted full-augmentation reaches (96.45) → net −0.14pp. On a well-tuned diverse-aug recipe, trading the final ~9 augmented epochs for a clean-tail re-adaptation does not pay; the augmented training is worth more than the train→eval distribution-match. The cooldown family is now exhausted across both recipes.
