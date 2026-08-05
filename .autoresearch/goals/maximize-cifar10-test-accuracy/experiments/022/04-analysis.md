# Report EXP-022: Alternating Final-Ten-Percent SAM
- **Created**: 2026-07-26

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.07% baseline within 300 counted training seconds, with a required acceptance threshold of 94.17%.

## Idea & Hypothesis

Apply rho-0.05 SAM on alternating optimizer steps only during the final 10% of counted time. The hypothesis was that sparse sharpness-aware gradients would improve terminal solution geometry while repairing EXP-021's dense-SAM exposure failure and retaining at least 90% of accepted exposure.

## Approach

`train.py` gained a pure progress/parity predicate and a bounded second-backward helper. The helper normalized first-pass gradients, cloned and perturbed parameters, cleared first-pass gradients, computed hard-label gradients at the perturbed weights, byte-restored parameters and post-first-forward BatchNorm buffers, then allowed the existing Nesterov optimizer to step once. SAM ran only on even steps at or above 90% progress.

## Execution

Semantic preflight passed exact parameter restoration, one persistent BatchNorm update, pure second-pass gradients, unchanged optimizer groups, and strict cadence behavior. Actual alternating-pattern timing projected 94.1746% exposure retention and 133.6337 passes. One scored run then completed on one H20 in 340.4 wall seconds without errors; mixup disabled at 195.0 counted seconds and SAM activated at 270.0 seconds on even step 24,970.

## Results

- **Primary metric**: 93.79% (baseline: 94.07%, delta: -0.28 points, -0.30%)
- **Observations**: The run completed 26,755 steps, 138 epochs, and 136.9856 effective passes. Post-SAM evaluations were 93.35%, 93.79%, and 93.76%; final loss was 0.2329 versus the accepted run's 0.2432, but top-1 remained lower.
- **Analysis**: The intervention achieved its local feasibility and geometry objectives: it retained substantially more exposure than dense late SAM, executed with correct state semantics, and ended with slightly lower test loss than the accepted baseline. It nevertheless lost 0.28 accuracy points. This mirrors EXP-013's lower-loss EMA failure and indicates that terminal probability smoothing or flatness does not move enough examples across the correct top-1 boundary under this schedule. Because dense late SAM is infeasible and alternating late SAM is feasible but harmful, this final-window rho-0.05 family is now closed.
- **Key Learning**: Feasible alternating late SAM lowers terminal loss but disrupts the accepted hard-label top-1 refinement, losing 0.28 points despite 136.99 passes.

## Verification

- **Conditions**: Completion/runtime passed; primary metric failed at 93.79% versus the required 94.17%.
- **Review Notes**: Results are trustworthy: one H20, fixed seed, exact 300-second counted budget, correct parameters/transitions, no duplicate epoch evaluations, and only planned `train.py` changes before scoring.
- **Verdict**: no-improvement
- **Verdict Basis**: A valid finite result was produced, but it was below the baseline and failed the metric condition.

## Unexplored Avenues

- Earlier or lower-rho SAM could change the tradeoff, but it lacks a stronger local rationale and risks either later erasure or a negligible effect; do not prioritize it.
- A cheaper sharpness proxy may avoid double-pass cost, but the top-1 regression here weakens the premise that terminal flatness is the current limiter.

## Next Steps

- **Medium confidence - tail-only mild RandAugment**: introduce one low-magnitude label-preserving transform only after mixup ends, avoiding the repeated early regularization-compounding failure and preflight CPU throughput.
- **Low confidence - alpha-0.1 closure**: run the remaining one-line mixup-strength probe with unchanged duration, accepting the strong under-regularization prior.
- **Low confidence - revisit non-compounding target regularization**: test only if a treatment preserves the accepted clean terminal refinement rather than softening it.
