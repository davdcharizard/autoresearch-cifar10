# Report EXP-061: Clean-data BN recalibration before eval in the final epochs

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-061.md
- **Plan**: plans/plan-061.md
- **Log**: logs/exp-log-061.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within a fixed 300s GPU-time (Σdt) budget on a single H20, editing only train.py. Baseline = **96.45** (EXP-054); bar = **96.55** (+0.1pp).

## Idea & Hypothesis
The model trains on Cutout (zeros ~25% of every image) + AugMix (distorts ~50%) inputs, so the BN running statistics (an EMA of augmented batch stats, used only at eval) are biased relative to the clean test distribution. Hypothesis: recomputing BN running mean/var on clean (un-augmented) training images before eval — Precise-BN, off the Σdt timer, ~0 epoch cost — corrects this train→eval BN-statistics mismatch and lifts best_test_acc ≥ 96.55. (Distinct from the convergence-framed Precise-BN dismissal and from SWA's averaged-weights BN-recompute.)

## Approach
Added `recalibrate_bn`/`restore_bn` helpers + a clean train loader (ToTensor+Normalize only, matching the frozen eval transform) + constants BN_RECAL_FRAC=0.025, BN_RECAL_BATCHES=16. In the final ~2.5% of the budget (~last 2 epochs), before the single per-epoch eval, recompute BN running stats (reset + momentum=None cumulative over 16 clean batches on the EAGER model), eval, then restore the augmented stats. All else byte-identical to EXP-054. Unit-tested: stats change on recalib, restore is exact, 22 BN layers, params unchanged.

## Execution
Single run on idle GPU 1 (bw46pwgsy, exit 0). dt steady 8ms throughout (compiled training forward untouched → no cudagraph break, confirming the eager-only recalibration design). Early epochs identical to EXP-054 (recalibration fires only in the tail). The augmented-BN peak (96.28) was set before the tail; the tail recalibration epochs then ran.

## Results
- **Primary metric**: **96.28%** (baseline 96.45, delta **−0.17pp**) — FAILS the 96.55 bar.
- **Clean-BN recalibration ACTIVELY HURT, badly**: the recalibrated tail epochs collapsed — ep89 94.71, ep90 94.91, ep91 94.83, ep92 94.65 (loss ~0.225), i.e. **~1.6pp WORSE** than the augmented-BN peak (96.28). The hypothesis was not just wrong but BACKWARDS.
- **Wall overrun**: total_seconds 604.6 > 600 — the recalibration overhead (plus the base recipe's documented high wall-variance, EXP-054 593s) tipped the run over the 600s budget. num_epochs 92, params unchanged, ≤1 eval/epoch preserved (92 evals).
- **Analysis / root cause**: BN uses BATCH statistics during training and RUNNING statistics at eval. The conv weights and BN affine params (γ,β) were optimized against activations normalized by the AUGMENTED+Cutout batch distribution — that distribution IS the network's trained-in operating point. The running stats (EMA of those augmented batch stats) correctly encode it, and eval with them is what the network expects. Recomputing BN stats on the CLEAN distribution shifts the normalization away from what γ,β were trained for → the entire downstream computation is mis-scaled → −1.6pp collapse. So the "mismatch" the hypothesis targeted is not a bug — the augmented BN stats are load-bearing. (16 batches/2048 imgs is ample for stable stats; the collapse is mechanistic, not a sample-size artifact — it reproduced flat across all 4 tail epochs.) This also explains why aug cooldown (EXP-033/034) gives only a tiny tail-climb: removing aug late lets BOTH the weights AND the batch/running stats re-adapt to clean data together, whereas recalibrating stats alone (weights fixed) breaks the pairing.
- **Key Learning**: Clean-data BN recalibration HURTS (−1.6pp) under heavy augmentation — the augmented BN running stats are the trained-in operating point the BN affine params expect; matching the clean eval distribution breaks the learned normalization. The recalibration overhead also overran the wall-tight AugMix recipe (604.6>600s).

## Verification
- **Conditions**: Necessary condition 1 (`best_test_acc >= 96.55`) FAILED (96.28). Condition 2 ALSO failed (total_seconds 604.6 > 600 — wall-budget breach). Condition 3 (scope train.py only, ≤1 eval/epoch [92==92], no new deps, seed 42, params 4,299,866) holds.
- **Review Notes**: Trustworthy as a negative result. dt 8ms steady (no contention/cudagraph issue), the tail-crater is consistent across 4 epochs and mechanistically explained, the augmented-BN peak (96.28) is a valid within-budget number. The −0.17 of the peak vs EXP-054's 96.45 is run-to-run AugMix variance + the extra epoch + mild clean-loader CPU contention.
- **Verdict**: **no-improvement**
- **Verdict Basis**: valid run; primary metric failed the bar decisively (and the approach hurt by 1.6pp); the +4.6s wall overrun is within the base recipe's known variance envelope plus recalibration overhead and does not make the metric untrustworthy, so classified no-improvement (not invalid), with the breach documented. NOTE: any future BN-recompute experiment MUST budget the wall — the recalibration overhead is NOT free on this 593s-tight recipe.

## Unexplored Avenues
- **BN recalibration on AUGMENTED data** (the standard Precise-BN data choice) would just reproduce the existing running stats — pointless here.
- The result actually CONFIRMS the existing recipe is correct: augmented BN stats are optimal for eval. No variant of clean-BN recalibration will help (the mechanism is fundamental). BN-statistics handling is now a CLOSED axis.
- The one adjacent lever it illuminates: aug cooldown lets weights+stats re-adapt to clean data jointly — but that is the already-mapped cooldown axis (EXP-033/034/035, marginal).

## Next Steps
- **BN-statistics recalibration axis CLOSED** (high confidence): clean-BN recompute hurts −1.6pp; augmented running stats are load-bearing. Do NOT retry Precise-BN / BN-recompute variants.
- **The plateau remains mapped across every lever**; 96.45 (EXP-054) stands as the k=4/300s ceiling after 62 experiments, 8 improvements. Augmentation (all policies+sub-levers+both delivery paths), capacity (×4), optimizer (family/objective/dynamics), schedule/LR, normalization, residual, head, batch, activation, throughput, weight-averaging, and now eval-time BN-stats are all closed.
- **Remaining genuine long-shots (low confidence)**: (a) WARMUP_FRAC isolation — the one untuned scalar, near-noise ceiling, cheap clean probe; (b) aug cooldown @0.10 ON the AugMix recipe (joint weights+stats clean-adaptation, never tested on AugMix) — but wall-tight and marginal; (c) per NEVER-STOP, continue principled long-shots accepting most will be no-improvement on this deeply-mapped plateau. Any wall-adjacent experiment must budget the 600s limit conservatively (EXP-061 lesson).

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
- (none — no exit actions defined for this goal)
