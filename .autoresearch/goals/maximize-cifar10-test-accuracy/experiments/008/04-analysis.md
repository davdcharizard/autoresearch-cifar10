# Report EXP-008: Decoupled Cosine-to-Zero Floor
- **Created**: 2026-07-24

## Goal

Maximize CIFAR-10 `best_test_acc` above the accepted 94.07% baseline, with at least 94.17% required for improvement, by isolating whether the accepted nonzero cosine endpoint prevents late hard-label settling.

## Idea & Hypothesis

Preserve the accepted 0.002-to-0.2 warmup exactly but anneal only the post-warmup cosine endpoint from 0.002 to zero. The hypothesis was that reducing residual late Nesterov motion would improve final settling, reach at least 94.17%, and keep final test loss no worse than 0.2432. The accepted final-equals-best trajectory was recognized in advance as contrary evidence and the expected ceiling as modest.

## Approach

Added `WARMUP_START_LR = 0.002`, set `MIN_LR = 0.0`, used the new constant in the warmup formula, and initialized SGD from it. An exact four-edit diff allowlist and eight production-function schedule assertions proved the entire 0-5% warmup stayed accepted while the post-warmup schedule reached zero. Model, RNG operations, data, mixup, continuous selective `5e-4` decay, momentum, evaluator, and cadence were unchanged.

## Execution

One fixed-seed run completed without retry or adjustment on one NVIDIA H20. It switched from mixup to hard labels exactly once at epoch 92, step 17,867, 195.0 counted seconds, with the expected LR 0.0598. The run completed 27,833 steps and 143 epochs in 300.0 counted / 341.0 total seconds with no numerical, resource, or cadence failure.

## Results

- **Primary metric**: 93.80% (baseline: 94.07%, delta: -0.27 percentage points, -0.29%)
- **Observations**: Best accuracy occurred at epoch 135 with test loss 0.2620; final accuracy was 93.78% and final loss 0.2629. The run realized 142.50496 passes, 0.6 more than EXP-002's reported 141.9, and used the unchanged 691,674 parameters / 1,094 MiB peak allocation. Accuracy slipped slightly as LR approached zero rather than improving through extra settling.
- **Analysis**: The intervention achieved its exact local effect and retained normal exposure, so insufficient compute or an implementation confound cannot explain the regression. Both accuracy and loss worsened versus the accepted 94.07% / 0.2432 endpoint. The evidence supports the competing interpretation preregistered in brainstorming: the hard-label tail still benefits from a small nonzero update amplitude, and suppressing roughly 52% of LR area in the final 10% under-updates useful margin refinement. The single run rejects this canonical zero endpoint for the accepted recipe; it does not prove every alternative time schedule inferior.
- **Key Learning**: The accepted 0.002 cosine floor supports useful late hard-label refinement; annealing to zero worsens loss and accuracy despite full exposure.

## Verification

- **Conditions**: Completion and process integrity passed; the required 94.17% accuracy threshold failed.
- **Review Notes**: Results confirmed trustworthy. One H20, exit 0, a complete summary, 300.0 counted seconds, 341.0 total seconds, 29 unique allowed evaluations, 142.50 passes, one correct transition, and a `train.py`-only allowlisted diff were verified.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid completed run scored 0.27 points below baseline and 0.37 below the required threshold, with normal exposure and worse final test loss.

## Unexplored Avenues

- An intermediate floor could trade continued refinement against settling, but the accepted 0.002 endpoint already outperformed zero and result-conditioned nearby floor tuning would have weak expected value; no immediate sweep is warranted.
- A different schedule shape that preserves meaningful late LR while reallocating update area earlier is materially broader than the zero-floor hypothesis and would require a new diagnosis.

## Next Steps

- **High confidence**: keep the 0.002 floor, continuous `5e-4` matrix decay, alpha-0.2 mixup, and 65% cutoff fixed; move to an orthogonal optimization or conditioning mechanism.
- **Medium confidence**: test evaluator-consistent channel standardization only with exact train/test placement and matched-throughput preflight, recognizing that stem-output BatchNorm limits its plausible effect.
- **Low confidence**: revisit parameter averaging only with a coherent BatchNorm-state policy; the stable, still-improving accepted tail gives it limited headroom.

## Exit Action Results

No exit actions were defined for this local-only goal.
