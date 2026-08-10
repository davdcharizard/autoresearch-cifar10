# Report EXP-023: FP32 Width-3 ResNet-14 Depth-Width Rebalance
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` above the moving 94.15% baseline while preserving the fixed evaluator, one-H20 hardware, 300-second counted training budget, and `train.py`-only scope. This experiment tested whether trading sequential depth for wider channels could improve representation quality while retaining enough fixed-time exposure; formal acceptance required at least 94.25%.

## Idea & Hypothesis

Replace the accepted width-2 ResNet-20 with an FP32 width-3 ResNet-14: two residual blocks per stage at widths 48/96/192. The candidate was selected because width had previously produced the largest architecture gain, full-width-3 FP32 was numerically stable, and fewer blocks could partially offset wider-convolution cost. The hypothesis was that richer channels plus reduced sequential depth would retain at least 20,000 updates and exceed 94.25% without changing the accepted optimizer, augmentation, schedule, transitions, or evaluation.

## Approach

Changed only `NUM_BLOCKS` from 3 to 2, `WIDTH_MULTIPLIER` from 2 to 3, and the descriptive model comment in `train.py`. The resulting postactivation model has six residual blocks, 13 convolutions, stages 48/96/192, unchanged Option-A shortcuts, and exactly 1,540,474 parameters. A 200-batch exact-corpus safety controller and five-pair alternating H20 timing controller were used before the one permitted production run. Cross-architecture loss and gradient ratios were retained as diagnostics rather than vetoes because depth, width, and parameter count all changed.

## Execution

The exact-corpus preflight passed with finite state and no candidate-only prediction concentration. Its candidate/control terminal loss-EMA ratio was 0.820794. Paired timing also passed: weighted step ratio 1.162780, worst pair 1.167083, projected 23,132 updates, 491.59 MiB peak allocation, and 384.53 seconds projected total runtime. One seed-42 production run then completed on the sole idle H20 without retry or error in 329.2 seconds total, including 300.0 counted training seconds. It produced 23,465 optimizer steps over 62 epochs and respected the expected graph, data-phase switch, worker lifecycle, evaluation frequency, and logging protocol.

## Results

- **Primary metric**: 94.00% (baseline: 94.15%, delta: -0.15 point, -0.16%)
- **Observations**: The candidate retained 87.24% of the accepted model's 26,898 updates despite having 43.4% more parameters. Strong-phase accuracy reached 89.28% and was 88.77% at the switch: above the 87.08% underfit marker, though 0.96 point below EXP-010's switch. Its first weak checkpoint was 93.34%, 0.18 point above EXP-010, showing effective initial conversion to hard-label refinement. It peaked at epoch 53 with 94.00% and 0.1930 NLL, then drifted to 93.89% and 0.1979 NLL rather than refining further.
- **Analysis**: The intervention achieved its local systems goal more strongly than its MAC estimate suggested: H20 step time rose only 16.28%, and actual exposure exceeded the timing projection by 1.44%. It also avoided severe strong-phase underfit and entered the weak tail well. The remaining miss is therefore not explained by a timing-gate failure or an immediate optimization collapse. Under this fixed-time recipe, removing one block from every stage appears to reduce iterative feature refinement enough that the extra channels cannot raise the late top-1 ceiling. The 12.76% update loss may contribute, but healthy early-tail conversion followed by regression points to the net depth/width allocation rather than exposure alone. Because width and depth changed jointly, this result rejects this exact width-3/ResNet-14 point; it does not establish that all additional width is harmful.
- **Key Learning**: Width-3 ResNet-14 retained 87.24% of accepted updates but peaked 0.15 points lower; lost depth outweighed added channel capacity.

## Verification

- **Conditions**: Primary metric condition failed; completion/numeric-summary and runtime facts were present and valid but were not used to rescue the first-condition failure.
- **Review Notes**: Results are trustworthy. The run completed once with numeric summaries, exactly one 80% phase switch, eight stopped workers, 18 unique at-most-once-per-epoch evaluations, 300.0 counted seconds, 329.2 total seconds, and the expected parameter count. No hard constraint or evaluation-integrity concern was found.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid `best_test_acc=94.00%` did not exceed the 94.15% baseline and missed the required 94.25% threshold.

## Unexplored Avenues

- Keep ResNet-20 depth and widen only the final stage, where added semantic capacity may cost less total compute and preserve early/middle-stage refinement.
- Test an intermediate or asymmetric channel allocation if it can be expressed cleanly and pass timing; this could separate global-width cost from the depth loss in this net hypothesis.
- A full width-3 FP32 ResNet-20 remains numerically plausible, but its exposure cost must clear a fresh fixed-time timing gate before it deserves a scored run.

## Next Steps

- **High confidence**: Test identity-initialized final-stage ECA on the accepted width-2 ResNet-20 to add conditional channel capacity without removing depth.
- **Medium confidence**: Develop a stage-specific widening candidate that preserves six blocks and spends capacity near the classifier.
- **Low confidence**: Run a cheap channels-last timing probe before coupling memory format to a future capacity experiment.
