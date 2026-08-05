# Report EXP-019: Static First-Block Scale Plus Final SE
- **Created**: 2026-07-26

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.07% WRN-16-2 baseline within 300 counted training seconds. Success required at least 94.17% with all local H20, scope, timing, seed, and evaluation constraints intact.

## Idea & Hypothesis

EXP-017's two conditional stage-3 gates reached 94.16%, while EXP-018's final gate alone fell to 93.67%. Because the removed first gate had a strong mean scale near 0.65 but low across-example output variance, EXP-019 tested whether a cheap learned static channel vector could restore its dominant attenuation while retaining final-block conditional selection.

## Approach

A 128-element no-decay parameter initialized to exact ones scaled the `layer3[0]` residual branch. EXP-018's ratio-16 SE gate, initialized from fixed project seed 42 inside a restored CPU RNG fork, scaled `layer3[1]`. Both transforms occurred after `conv2` and before unchanged shortcut addition. A terminal-only parameter summary measured the learned static vector without adding training-loop instrumentation.

## Execution

Semantic preflight passed placement, 693,986 parameters, exact accepted common state/logits/RNG, unit initial scales, optimizer groups, ungated shortcuts, and gradient opening. Matched H20 timing retained 98.24% throughput with worst CV 0.386%. The sole scored run completed cleanly, disabled mixup once at step 17,174 and 195.0 seconds, evaluated 28 unique epochs, and produced no error signature.

## Results

- **Primary metric**: 93.86% (baseline: 94.07%, delta: -0.21 points, -0.22%)
- **Observations**: The candidate completed 26,758 steps, or 137.00096 passes, in 300.0 training seconds and 340.3 wall seconds. Final accuracy/loss were 93.86%/0.2348. The static scale learned mean 0.674833, population std 0.151918, and range 0.427161-1.095004, closely matching EXP-017 gate 0's 0.6468 aggregate attenuation.
- **Analysis**: The local mechanism succeeded: exact-neutral static parameters moved strongly toward the preregistered attenuation pattern, so failure cannot be attributed to weak gradients or insufficient movement from one. Yet accuracy remained below baseline and 0.30 points below EXP-017 despite 3.36 more passes. Mean/channelwise attenuation is therefore insufficient. Gate 0's input dependence, even though its output variance was modest, or co-adaptation between two conditional gates supplied the near-positive behavior. This closes static approximation as a cheap substitute and demonstrates that component importance cannot be inferred from aggregate variance alone.
- **Key Learning**: Reproducing gate 0's mean attenuation is insufficient; input-dependent two-gate interaction, not static scaling or exposure, drove EXP-017's signal.

## Verification

- **Conditions**: Completion and integrity passed; the primary metric condition failed because 93.86% is below 94.17%.
- **Review Notes**: Results are trustworthy. One H20, fixed seed 42, exact state/RNG control, one scored execution, correct timing/evaluation behavior, and only the planned `train.py` diff were confirmed.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid complete run with no hard-constraint violation, but accuracy fell 0.21 points below baseline and 0.31 below threshold.

## Unexplored Avenues

- **Two conditional gates without diagnostics**: preserves the only positive interaction while removing observation overhead, but likely offers a narrow, noise-scale gain and should require compelling measured recovery.
- **Earlier-stage conditional interaction with cheaper operators**: input dependence appears necessary, but a new mechanism needs global channel mixing without another small latency-bound MLP.
- **Time-restricted input augmentation**: an early-only RandAugment design remains orthogonal if worker-state propagation is verified and the clean tail is preserved.

## Next Steps

- **Medium confidence - extend mixup to 75%**: EXP-004 showed 50% is too short; test the unexplored opposite side while preserving a meaningful 25% clean tail.
- **Medium confidence - early-only mild RandAugment**: use explicit shared worker state and preflight phase propagation before scoring.
- **Low confidence - two-gate SE without diagnostics**: pursue only if exact timing shows diagnostic removal recovers meaningful exposure.

## Exit Action Results

