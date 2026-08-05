# Report EXP-010: Selective 160-Channel Final Stage
- **Created**: 2026-07-24

## Goal

Maximize CIFAR-10 `best_test_acc` above 94.07%, requiring at least 94.17%, by testing whether extra FP32 capacity confined to the low-resolution final WRN stage is a better fixed-time allocation than uniform widening.

## Idea & Hypothesis

Change stage widths from `[32,64,128]` to `[32,64,160]`, adding 39% parameters for only 17% forward MACs while preserving all high-resolution stages and the accepted training recipe. The hypothesis predicted that improved abstract class separation would offset reduced exposure and score at least 94.17%, contingent on at least 120 passes.

## Approach

Replaced the uniform factor with explicit validated stage widths, used them for all stages/final BN/classifier, and logged the exact topology. Full constructor/topology tests verified 961,562 parameters. An evaluator-free production-path preflight separately measured mixup and hard-label regimes, combined them at 65/35 counted-time weights, and gated scoring at 85% retention / 120 projected passes. Training, initialization, FP32 precision, optimizer, schedule, mixup, seed, data, and evaluator logic were unchanged.

## Execution

Preflight retention was 0.923362 with 131.025 projected passes; all regime CVs were below 0.86%. One fixed-seed run completed without retry or adjustment. Mixup disabled at epoch 85, step 16,565, 195.0 seconds with LR 0.0612. The run completed 25,812 steps / 133 epochs in 300.0 counted / 339.2 total seconds.

## Results

- **Primary metric**: 94.11% (baseline: 94.07%, delta: +0.04 percentage points, +0.04%)
- **Observations**: The run realized 132.15744 passes, above the 120 interpretation gate and close to projection. Best accuracy occurred at epoch 130 with loss 0.2435, nearly matching accepted final loss 0.2432. Final accuracy/loss were 94.06% / 0.2457. Peak allocation rose modestly to 1,171.4 MiB.
- **Analysis**: Selective low-resolution capacity produced the first positive delta since EXP-002 and preserved the accepted loss regime despite 6.9% fewer passes, supporting the compute-allocation rationale. However, +0.04 is below the preregistered +0.10 margin, and final accuracy returned to 94.06%. Because exposure exceeded 120 and execution was stable, this run rejects `[32,64,160]` as a sufficient standalone improvement under the accepted optimizer. It does not reject selective capacity generally. The near miss cannot be rerun or rescued by post-hoc width/LR tuning in this loop.
- **Key Learning**: Selective 8x8-stage width is directionally useful at manageable cost, but 160 channels alone gains only 0.04 points and misses the acceptance margin.

## Verification

- **Conditions**: Completion/process integrity passed; the required 94.17% threshold failed.
- **Review Notes**: Results confirmed trustworthy: one H20, exact topology/count, fail-closed preflight, 300.0 counted seconds, 339.2 total, 132.16 passes, 27 unique accepted-cadence evaluations, one transition, and `train.py`-only diff.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid run improved only 0.04 points, 0.06 short of the required margin; no rerun is permitted.

## Unexplored Avenues

- A bottlenecked or depth-based low-resolution capacity change could allocate compute differently, but it requires a new architecture rationale rather than nearby width tuning from this result.
- FP32 fused SGD could potentially recover some exposure without changing model arithmetic; it must first show material matched throughput and should not be assumed to rescue this near miss.
- Another stage width is not justified as an immediate result-conditioned sweep; the exact 160-channel allocation is closed.

## Next Steps

- **High confidence**: measure FP32 fused-SGD opportunity on the accepted model; score only if the production-path gain is material enough for the accuracy bar.
- **Medium confidence**: thoroughly develop a distinct low-resolution bottleneck/depth allocation, informed by the positive delta but not as a width-neighbor retry.
- **Low confidence**: defer RandAugment because accumulated additive-regularization evidence remains negative.

## Exit Action Results

No exit actions were defined for this local-only goal.
