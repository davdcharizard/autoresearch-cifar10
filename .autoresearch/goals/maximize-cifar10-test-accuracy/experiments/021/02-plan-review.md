# EXP-021 Plan Review

1. **Final-window LR is low and SAM may trade too many fine-tuning steps for negligible movement.** This is the central research risk; it remains the convergence-aligned window selected by idea review.
2. **Subtracting perturbations is not bit-exact.** Clone each original parameter before perturbing under no_grad, then restore with copy_ before optimizer.step. Preflight must call the exact production helper.
3. **Exposure threshold needs derivation.** A 10% double-pass window implies ideal whole-run retention about 1/1.1=90.9%; 127 passes is a conservative floor below 141.9/1.1=129.0.
4. **Transition bands are integrity checks, not optimization targets.** Keep one-second bands around fixed counted-time boundaries and classify timing deviations by cause.
5. **Single-run margin cannot establish variance significance.** The fixed seed/no-reroll protocol is user-mandated; report this structural limitation without adding runs.

