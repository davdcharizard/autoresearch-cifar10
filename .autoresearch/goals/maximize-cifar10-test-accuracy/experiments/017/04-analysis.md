# Report EXP-017: Neutral Stage-3 Squeeze-and-Excitation
- **Created**: 2026-07-26

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.07% WRN-16-2 baseline within 300 counted training seconds. Success required at least 94.17% with all local H20, scope, timing, seed, and evaluation constraints intact.

## Idea & Hypothesis

The strongest unaccepted results had added dense 8x8 capacity. EXP-017 tested whether input-dependent channel selection could capture that benefit more efficiently than another full residual block. Two ratio-16 squeeze-and-excitation gates were placed on the existing stage-3 residual branches and initialized to exact identity, preserving accepted logits and common state at construction.

## Approach

Each gate pooled the signed residual branch, projected 128 channels through an 8-unit ReLU bottleneck, and applied `2 * sigmoid(logit)` channel scales before shortcut addition. The second projection was zero-initialized so every initial scale was exactly one. Gate creation used preregistered seed 17017 inside a restored CPU RNG fork after accepted initialization. Training-only, non-persistent scalar diagnostics measured gate means, variance, example dependence, saturation, and feature-versus-bias logit energy; their runtime was included in throughput and scoring.

## Execution

Semantic preflight passed exact placement, 696,042 parameters, state/RNG preservation, seed oracle, identity logits, two-step gate opening, optimizer coverage, and diagnostic formulas. Matched timing measured 13.044400 ms for accepted steps and 13.679807 ms for the candidate, 95.36% retention and 135.31 projected passes. The sole scored run completed cleanly on one H20, disabled mixup once at step 16,744 and 195.0 seconds, evaluated 27 unique epochs, and emitted no error signature.

## Results

- **Primary metric**: 94.16% (baseline: 94.07%, delta: +0.09 points, +0.10%)
- **Observations**: The candidate completed 26,101 steps, or 133.63712 dataset-equivalent passes, in 300.0 training seconds and 338.5 wall seconds. Final accuracy/loss were 94.12%/0.2321, versus 94.07%/0.2432 accepted. Gate 0 averaged 0.6468 with 0.00312 across-example variance; gate 1 averaged 0.8695 with 0.02431 across-example variance. Feature logit RMS was about four times bias RMS for both gates, while saturation was negligible for gate 0 and 0.37% for gate 1.
- **Analysis**: The mechanism was real rather than an inert identity parameterization or learned static bias. Both gates became feature-driven, and the final block was especially conditional. The lower final loss and +0.09 top-1 signal support late channel selection, but the exact treatment did not clear the required margin. Its measured 4.6% step overhead reduced realized exposure to 133.64 passes, and the first gate mostly learned strong, weakly conditional attenuation. A distinct follow-up should concentrate selection on the more conditional final block while recovering throughput; changing the ratio, seed, or rerunning this exact placement would be post-hoc rescue and is closed.
- **Key Learning**: Late feature-driven channel selection is directionally useful, but two instrumented ratio-16 gates cost too much exposure and finish 0.01 below acceptance.

## Verification

- **Conditions**: Completion and integrity passed; the primary metric condition failed because 94.16% is below 94.17%.
- **Review Notes**: Results are trustworthy. Only `train.py` changed, accepted common state and RNG were preserved, the fixed gate seed was preregistered, diagnostics were part of measured runtime, and exactly one scored run completed.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid complete run with no hard-constraint violation, but the +0.09-point result missed the minimum improvement margin by 0.01.

## Unexplored Avenues

- **Final-block-only channel selection**: gate 1 showed nearly eight times gate 0's across-example variance, while halving the gate count should recover part of the lost exposure. This is a distinct placement hypothesis rather than a rerun.
- **Kernel-cheaper conditional scaling**: the positive loss/top-1 direction may survive a lower-overhead selector, but any design must preserve exact initialization and demonstrate genuine feature dependence.
- **Early update conditioning**: SAM remains an orthogonal optimization proposal, but it requires a diagnostic confirming that sharpness is a relevant bottleneck before spending a run.

## Next Steps

- **High confidence - final-block-only neutral SE**: retain the gate with the strongest conditional behavior, remove the weakly conditional first gate, and preregister throughput/semantic checks.
- **Medium confidence - mild data invariance**: keep one-operation RandAugment as an orthogonal option if a lighter selector fails.
- **Low confidence - early SAM**: defer until a local sharpness diagnostic supports the mechanism and its extra backward cost fits the budget.
