# Adversarial Plan Review - EXP-027

## Concerns

1. A 0.03-point additive cushion can be crossed by max-over-evaluations noise; a credible accept needs endpoint corroboration beyond `best_test_acc` alone.
2. Exact EXP-011 identity is load-bearing because transform/loader construction occurs before model construction; compare the full model and post-construction RNG/shuffle traces against an independent direct `[2,2,3]` oracle.
3. GPU throughput only verifies the deep-model exposure regime; it cannot observe worker augmentation and must not substitute for loader/wall feasibility.
4. EXP-026's wall constants are wrong for the deeper model; anchor projections to EXP-011's 338.5-second, 134-epoch run.
5. The 130-pass floor has little headroom relative to EXP-011's 131.64 projection/132.92 realization and needs lower-variance timing.
6. The intervention stacks early regularization on a deeper model; it does not directly regularize the clean tail. Loss remains a mechanism check, not proof by itself.

## Disposition

- Adopt 1: require both `best_test_acc` and the predetermined `final_test_acc` to reach 94.17%, while retaining final loss below 0.2782 as informative interaction corroboration.
- Adopt 2 with explicit full-state, CPU/CUDA RNG, sampler-label, and mixup-stream oracle checks.
- Keep separate GPU and loader gates and describe their distinct roles; adopt EXP-011 wall anchors per 3/4.
- Keep the brainstorm's fixed 130-pass floor but tighten GPU timing CV to 2% and acknowledge the narrow margin. Do not lower it after review.
- Reframe the mechanism as early invariance shaping a deeper model, not directly regularizing its tail.
