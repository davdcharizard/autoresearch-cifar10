# Report EXP-021: Final-Ten-Percent SAM
- **Created**: 2026-07-26

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.07% baseline within 300 counted seconds. The proposed SAM treatment also had to satisfy preregistered semantic and fixed-time feasibility gates before scoring.

## Idea & Hypothesis

Apply rho-0.05 non-adaptive SAM only during the final 10% counted-time convergence window, preserving accepted SGD earlier and targeting terminal basin flatness without later SGD erasure.

## Approach

The implementation retained the accepted first forward/backward, cloned parameters, applied normalized perturbations under `no_grad`, ran a second hard-label forward/backward, byte-restored parameters and BatchNorm buffers, then applied one accepted optimizer step using second-pass gradients. Preflight exercised the exact production helper.

## Execution

Semantic preflight passed strict window semantics, 691,674 parameters, exact parameter restoration, one persistent BatchNorm update, finite gradients, and unchanged optimizer groups. Timing then measured normal/SAM steps at 12.447067/26.789581 ms with very low CV. No scored run was launched because whole-run retention 0.896678 failed the preregistered 0.90 floor.

## Results

- **Primary metric**: unavailable (baseline: 94.07%; no scored run)
- **Observations**: SAM steps cost 2.1523x normal. The weighted final-10% design projected 127.2385 passes, but only 89.67% retention versus the required 90%.
- **Analysis**: The semantic design was sound, but cloning/restoration and the second pass cost more than the idealized 2x assumption. Weakening the threshold after measurement would be post-hoc gate fitting. This exact every-step final-window treatment is therefore infeasible under its own fixed-budget premise; periodic SAM could be materially different, but needs a new cadence rationale.
- **Key Learning**: Every-step final-window SAM is semantically controllable but costs 2.15x per step and misses the fixed-time retention floor.

## Verification

- **Conditions**: Verification not run; execution stopped at the preregistered throughput gate.
- **Review Notes**: Preflight evidence is trustworthy; no accuracy claim can be made.
- **Verdict**: crash
- **Verdict Basis**: No scored primary metric was produced because the planned treatment failed feasibility before execution.

## Unexplored Avenues

- **Periodic late SAM**: perturb every second or fourth step to reduce cost, but cadence needs mechanistic justification and separate preflight.
- **Cheaper sharpness proxy**: gradient-noise or perturbation approximations could target geometry without full double passes.

## Next Steps

- **Medium confidence - alpha 0.1 closure**: use the remaining one-line mixup-strength probe despite its under-regularization risk.
- **Low confidence - periodic SAM**: revisit only with a principled cadence and >=90% measured retention.

## Exit Action Results

