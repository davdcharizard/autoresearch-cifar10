# Report EXP-010: Back-loaded 1-2-3 stage depth
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` under the frozen 300-second charged training budget. EXP-010 grew from EXP-002 at 95.23%, so formal improvement required 95.33%. The global best before and after this experiment is EXP-004 at 95.40%.

## Idea & Hypothesis

The experiment moved one equal-MAC residual block from stage 1 (64 channels, 32x32) to stage 3 (256 channels, 8x8), changing the six-block allocation from 2-2-2 to 1-2-3. The intent was to spend the same major-convolution budget on more high-level semantic capacity while retaining data and optimizer exposure. The preregistered hypothesis was at least 26,500 steps and 95.53% best accuracy; 95.33-95.52 would be a formal but sub-hypothesis improvement.

## Approach

Only `train.py` changed. Its static block list removes the second 64-channel identity block and adds a third 256-channel identity block; model/config labels report the allocation. The candidate retains six blocks, twelve residual 3x3 convolutions, three projections, 392,612,352 Conv/Linear MACs per image, and the full EXP-002 optimizer, CutMix, schedule, precision, seed, timer, and evaluator behavior. Parameters rise from 2,748,890 to 3,855,578. Claude's plan review removed benchmark-only parameterization and bounded the first valid parent-relative latency measurement as decisive.

## Execution

Source, inventory, CPU FP32, and physical-GPU-0 BF16 smokes passed. The first valid paired preflight had only 2.15% parent round drift and needed no repeat. Candidate median/p90 latency was 9.241963/9.410895 ms versus parent 10.000283/10.200921 ms; candidate evaluation was also faster. One fixed-seed full run then completed exit 0 on physical GPU 0 in 455.7 seconds total, with 300.0 charged seconds and no error or nonfinite match. Disposable-harness import and cross-architecture initialization assertions were corrected before the metric run without changing production code or settings.

## Results

- **Primary metric**: 95.04% (parent: 95.23%, delta vs parent: -0.19 points, -0.20%; global best: 95.40%, delta: -0.36)
- **Observations**: The candidate completed 30,558 steps and 157 epochs versus the parent's 27,950 steps and 144 epochs, a 9.3% exposure increase. CutMix exposure was 11,165/22,510 = 0.4960. Final accuracy was 95.04% and final loss 0.2131, versus parent 95.19% and 0.2044. Peak VRAM rose only 14.8 MiB to 1,193.7 MiB despite 40.3% more parameters.
- **Analysis**: The intervention achieved its local systems objective better than expected: moving activation work from 32x32 to 8x8 made the equal-MAC model faster and removed exposure loss as an explanation. It did not achieve the accuracy objective. The fixed package missed the formal threshold by 0.29 points and the stronger hypothesis by 0.49. The -0.19 parent delta overlaps the goal's observed single-run variation, so the evidence supports no detectable improvement, not a directional claim that back-loading is inherently harmful. The package also redistributes stagewise drop-path dose and changes shape-dependent initialization RNG; early-feature removal, regularization allocation, and RNG realization are not isolated.
- **Key Learning**: Moving one equal-MAC block from stage 1 to stage 3 increased steps 9.3% but produced no detectable accuracy improvement.

## Verification

- **Conditions**: Parent-relative accuracy failed: 95.04% was below both parent 95.23% and required 95.33%. The run itself completed cleanly with the correct GPU, scope, budget, summary, and 157 evaluations for 157 epochs.
- **Review Notes**: Claude independently audited the diff, parameter/MAC arithmetic, exposure counts, timing consistency, scope, and verdict. It found the result trustworthy, with the caveat that protocol-mandated transient log deletion leaves the detailed `03-execute.md` transcription as the durable evidence. Causal explanations beyond the fixed package are explicitly treated as unverified.
- **Verdict**: no-improvement
- **Verdict Basis**: A valid metric was produced under all hard constraints, but the primary necessary condition failed. The node is a failed leaf from EXP-002 on `br-000`, commit `160b62f`.

## Unexplored Avenues

- Preserve both 64-channel blocks and reallocate a middle-stage block late (2-1-3), testing whether high-resolution local depth rather than uniform staging is the critical constraint. This is a distinct package and still requires parent-relative latency/accuracy evidence.
- Retain 2-2-2 depth while modestly widening only the final stage. It would add late capacity without deleting local processing, but would no longer be compute-neutral.
- Isolate stagewise stochastic-depth dose from allocation by preserving stage-local survival probabilities. This could clarify mechanism, but a causal ablation has lower payoff than a new accuracy direction after the fixed package missed by 0.29 points.

## Next Steps

- **High confidence**: Prioritize low-overhead late-iterate averaging or another mechanism that targets observed checkpoint variance while preserving the validated 2-2-2 representation and full data exposure.
- **Medium confidence**: Explore an existing-convolution change that retains both early blocks, using the demonstrated activation-traffic latency benefit as a systems constraint rather than assuming parameter count predicts speed.
- **Low confidence**: Return to back-loaded depth only for a deliberately isolated stage-allocation/regularization study, not as an immediate accuracy bet.

## Exit Action Results

- No exit actions were defined for this goal.
