# Report EXP-038: Double Only Terminal Classifier Decay
- **Created**: 2026-07-27

## Goal

Maximize fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.48% baseline under the unchanged 300-second counted-training budget. This experiment tested whether stronger continuous shrinkage of only the terminal classifier could improve pooled-representation boundary quality by at least 0.10 points.

## Idea & Hypothesis

EXP037 showed that removing decay from `fc.weight` preserved exposure but slightly reduced accuracy and substantially worsened loss. EXP038 tested the symmetric opposite endpoint, doubling only classifier decay from accepted `5e-4` to `1e-3`, while explicitly treating the prior result as directional rather than monotonic evidence. The hypothesis required at least 127 passes, best accuracy at least 94.58%, final accuracy at least 94.45%, and loss no worse than 0.2456.

## Approach

Production added `CLASSIFIER_WEIGHT_DECAY = 1e-3` and split SGD parameters into three exact groups: 999,856 representation/head matrix elements at `5e-4`, only the 1,280-element `fc.weight` at `1e-3`, and 2,346 rank-below-2 elements at zero. Model graph, initialization, data, RNG, augmentation, schedule, loss, evaluator, seed, and all other optimizer options remained accepted. An evaluator-free preflight independently loaded `a7c42dc`, proved pre-step identity, checked group membership, and verified fresh and preseeded-momentum coupled Nesterov updates against independent oracles.

## Execution

Semantic and timing gates passed before the sole score. Timing retained 0.998113 of accepted throughput, projected 130.058 passes, stayed below 0.51% CV, and peaked at 610.16 MiB. The single local H20 run completed without retry in 300.0 counted and 346.3 wall seconds. Mixup stopped at 195.0 seconds; RandAugment stopped after the active iterator exhausted at 196.1 seconds; 27 unique evaluations followed the required cadence.

## Results

- **Primary metric**: 93.82% (baseline: 94.48%, delta: -0.66 points, -0.70%)
- **Observations**: Final accuracy was 93.74% and loss 0.2598, both worse than the accepted 94.45% and 0.2456. The run delivered 25,399 steps, 130.04288 passes, 131 epochs, 1,096.4 MiB peak VRAM, and the unchanged 1,003,482 parameters.
- **Analysis**: The intervention achieved its intended local optimizer effect with negligible overhead, but stronger terminal shrinkage harmed both top-1 accuracy and loss. Together with EXP037's zero-decay result, the evidence locally favors the accepted `5e-4` classifier decay over the two tested one-sided endpoints. It does not prove monotonicity or formally reject intermediate values or schedules.
- **Key Learning**: Classifier decay is locally bracketed: both zero and `1e-3` lose at normal exposure, so retain `5e-4` and pursue an orthogonal mechanism.

## Verification

- **Conditions**: Completion/resource contract passed; primary metric improvement failed.
- **Review Notes**: Results are trustworthy: the sole score completed normally on one idle H20, scope remained limited to `train.py`, source/RNG/update semantics passed, exposure exceeded 127 passes, and cadence/budget checks passed.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid 93.82% score missed the 94.58% threshold by 0.76 points and the baseline by 0.66 points.

## Unexplored Avenues

- Static classifier decay values strictly between `0`, `5e-4`, and `1e-3` remain formally untested, but the normal-exposure endpoint bracket gives them low priority without a new mechanistic diagnosis.
- Scheduled classifier decay is a distinct intervention and remains untested; it would require an independently motivated temporal hypothesis rather than serving as a rescue of these endpoints.

## Next Steps

- **High confidence**: Preserve accepted `5e-4` classifier decay and seek a low-cost mechanism orthogonal to parameter-norm shrinkage.
- **Medium confidence**: Investigate training-derived classifier geometry such as normalized logits only with an exact scale rationale and a predeclared no-sweep test.
- **Medium confidence**: Revisit the pooled residual head through a structurally distinct interaction, not adjacent width, scale, seed, or optimizer tuning.
