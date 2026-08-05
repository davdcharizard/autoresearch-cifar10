# Adversarial Review — EXP-021 Finalists

## Prioritized Feedback

1. **Two-gate SE without diagnostics is a seed-selected, noise-scale replay.** Reusing the 94.16 initialization creates a no-reroll concern, and scalar observation removal likely recovers too little time.
2. **Alpha 0.1 fights the under-regularization prior.** Beta(0.1,0.1) is endpoint-heavy and resembles less effective mixup, while the 50% cutoff already regressed.
3. **SAM has the highest ceiling but the early window is mechanistically misplaced.** Plain SGD for the remaining 90% can erase early flatness. Move the fixed rho-0.05 window to the final 10%, byte-restore BatchNorm buffers after the perturbation pass, and measure its fixed-time cost.
4. **SAM calibration remains risky.** The extra pass reduces late update density and rho 0.05 lacks local calibration, but it is the only genuinely new optimization axis.

## Scored Verdict

- **Ten-Percent SAM**: evidence 6/10, impact 7/10. Highest upside, conditional on a final/convergence window and verified BN restoration.
- **Alpha-0.1 mixup**: evidence 4/10, impact 3/10. Clean but fights multiple under-regularization signals.
- **Two-gate SE without diagnostics**: evidence 2/10, impact 3/10. Constraint-adjacent near replay with noise-level upside.

## Pick

**Ten-Percent SAM, refined to the final 10% counted-time window.** It targets optimization geometry after the accepted trajectory has converged, limiting double-pass cost and avoiding later SGD erasure. Preflight must verify exact perturbation restoration, one persistent BN update per optimizer step, and fixed-time feasibility.

