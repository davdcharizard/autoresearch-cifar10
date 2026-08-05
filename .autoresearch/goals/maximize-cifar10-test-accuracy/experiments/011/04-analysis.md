# Report EXP-011: One Extra 8x8 Residual Block
- **Created**: 2026-07-24

## Goal

Maximize CIFAR-10 `best_test_acc` above the accepted 94.07%, requiring at least 94.17%, by testing whether a depth-based low-resolution capacity allocation outperforms the accepted WRN within the fixed 300-second training budget.

## Idea & Hypothesis

Keep accepted widths `[32,64,128]` and change stage block counts from `[2,2,2]` to `[2,2,3]`, adding one unchanged 128-to-128 pre-activation residual block at 8x8. Its parameter/MAC cost closely matches EXP-010's directionally positive width treatment but spends capacity on nonlinear feature refinement. The hypothesis predicted at least 94.17% while retaining 120 or more passes.

## Approach

Replaced the scalar block count with explicit, strictly validated stage counts and used them independently when constructing the three stages. The final topology had 987,098 parameters and logged exact widths/depths. Full constructor/topology checks and an evaluator-free production-path benchmark compared accepted `[2,2,2]` with candidate `[2,2,3]`; FP32 training, initialization, optimizer, schedule, mixup, seed, data, widths, and evaluator behavior were unchanged.

## Execution

The matched preflight retained 92.77% throughput and projected 131.64 passes, passing all semantic, variability, throughput, and exposure gates. One fixed-seed H20 run then completed without retry or adjustment. Mixup disabled once at 195.0 seconds, and the run finished 25,961 steps / 134 epochs in 300.0 counted / 338.5 total seconds. A diagnostic-only preflight import-path error was fixed before model construction and did not affect scoring.

## Results

- **Primary metric**: 94.15% (baseline: 94.07%, delta: +0.08 percentage points, +0.09%)
- **Observations**: Realized exposure was 132.92032 passes, above the 120 interpretation gate and 93.67% of accepted exposure. Best and final accuracy were both 94.15% at terminal epoch 134, with final test loss 0.2782. Peak allocation was 1,096.3 MiB. The candidate improved more than EXP-010's width treatment (+0.08 versus +0.04) at similar exposure, but its test loss was materially worse than the accepted 0.2432 and EXP-010's 0.2457 despite near-zero late training loss.
- **Analysis**: The extra block achieved its local goal of adding low-resolution transformation capacity without an operational bottleneck and produced a directionally positive accuracy delta. However, it missed the strict margin by 0.02 points. Because exposure was adequate, execution stable, and best accuracy occurred at the terminal evaluation, the result is not explained by premature stopping or throughput collapse. The high test loss alongside tiny training loss instead suggests the extra raw depth increased overconfidence or generalization error. This stable negative closes exact `[2,2,3]` as a sufficient standalone treatment, not all low-resolution capacity or more compute-efficient transformation designs.
- **Key Learning**: One extra 8x8 block gains 0.08 points but worsens test loss; raw low-resolution depth is directionally useful yet insufficiently generalizing.

## Verification

- **Conditions**: Completion/process integrity passed; the required 94.17% threshold failed.
- **Review Notes**: Results confirmed trustworthy: one H20, exact topology/count, fail-closed preflight, 300.0 counted seconds, 338.5 total, 132.92 passes, 27 unique accepted-cadence evaluations, one transition, one fixed-seed scored run, and a `train.py`-only diff.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid run gained only 0.08 points, 0.02 short of the preregistered +0.10 margin; no rerun is permitted.

## Unexplored Avenues

- A fully specified 8x8 bottleneck could add nonlinear transformation with fewer MACs and a different regularization profile, but its exact internal width, ordering, initialization, and cost must be preregistered before testing.
- Weight averaging could target the observed generalization/confidence gap without changing model capacity, provided its update overhead and evaluator swap semantics pass an evaluator-free preflight.
- Combining width and depth is not justified from two near misses alone and would increase cost; it should not be used as an immediate result-conditioned rescue.

## Next Steps

- **High confidence**: develop one exact compute-efficient 8x8 bottleneck proposal with auditable parameter/MAC totals and production-path gates.
- **Medium confidence**: investigate a precisely defined local weight-averaging treatment aimed at the higher test loss rather than adding more raw capacity.
- **Low confidence**: retain low-magnitude RandAugment only as a distant fallback because accumulated additive-regularization evidence remains negative.

## Exit Action Results

No exit actions were defined for this local-only goal.
