# Proposal: FP32 Width-3 ResNet-14 Depth–Width Rebalance

## Intervention and hypothesis

Change only `NUM_BLOCKS = 3 -> 2` and `WIDTH_MULTIPLIER = 2 -> 3`. This replaces the accepted width-2 ResNet-20 with a postactivation width-3 ResNet-14: two residual blocks per stage at channels `48/96/192`. Preserve FP32/default-TF32 execution, batch 128, ordinary SGD momentum 0.9, all-parameter coupled decay `1e-4`, the complete LR schedule, N1/M7 plus p=0.5 alpha-1 CutMix through 80%, hard weak tail, seed 42, workers, timer, and evaluator.

The hypothesis is that wider features provide more useful strong-view capacity while removing one block per stage avoids the full cost of width-3 ResNet-20. Point prediction: a candidate/control synchronized step ratio near `1.30`, about 20,690 updates in 300 seconds, and `best_test_acc >= 94.25%` (point estimate 94.35%). This is a net fixed-time architecture hypothesis, not equal-compute proof that width is intrinsically better than depth.

## Evidence and mechanism

EXP-007 is the strongest local architecture evidence: width-2 ResNet-20 raised the strong checkpoint by 5.48 points and best accuracy by 1.25 despite retaining only 70.76% of width-1 updates. That establishes channel capacity as valuable under the hard-view phase. EXP-010 then showed that width 2 can absorb conservative CutMix and reach 94.15%. Wide Residual Networks independently supports trading extreme sequential depth for width on CIFAR, but its preactivation/dropout, longer schedules, and stronger decay do not transfer directly.

The candidate is deliberately smaller than EXP-016's full width-3 ResNet-20. EXP-016 counted 2,412,730 parameters and found the matched width-3 FP32 trajectory did not cross the BF16 candidate's concentration veto; the invalid result localized failure to BF16, not width-3 FP32. Removing three residual blocks reduces sequential Conv/BN backward work—the measured 75.46% bottleneck—while preserving the accepted postactivation blocks and Option-A transitions.

## Exact architecture and cost

The graph must contain:

- stem `3 -> 48`, then stages `48/96/192`, two `BasicBlock`s each;
- six residual blocks, 13 `Conv2d` layers total, two stride-2 Option-A slice/pad transitions, unchanged global average pooling, and `Linear(192,10)`;
- exactly **1,540,474 trainable parameters**: 1,535,760 convolution weights, 2,784 BN affine parameters, and 1,930 classifier parameters.

This is 1.4344x the accepted 1,073,962 parameters but only 63.85% of full width-3 ResNet-20. Approximate convolution MACs are 234.9M/image versus 161.3M for width-2 ResNet-20 (`1.456x`), but six fewer Conv/BN layers and wider H20 kernels may make wall-step scaling sublinear; local timing is decisive. Memory is not limiting: accepted peak allocation is 598.7 MiB on a 97,871-MiB H20.

## Structural, safety, and timing gates

Before production:

1. Assert the exact channels, block/layer counts, parameter count, stage shapes `[128,48,32,32]`, `[128,96,16,16]`, `[128,192,8,8]`, two Option-A pads, postactivation ordering, FP32 parameters/buffers, and unchanged optimizer groups and hyperparameters. Require only the two architecture constants to differ in production.
2. Verify finite hard- and probability-target forward/loss/backward/update paths; every parameter receives an expected finite gradient; BN counters advance once; momentum buffers are FP32; and evaluation returns finite `[N,10]` logits without changing model state.
3. Replay at least 200 persisted production-distribution batches through fresh accepted and candidate processes. Require finite loss/state, candidate terminal loss EMA no more than 1.5x control, no candidate-only prediction concentration above 95%, and no extreme gradient/update spike above 2x control. Persist the exact augmented corpus because seed-only forkserver replay is not reliable. This rechecks the shallower architecture even though EXP-016's full-depth width-3 FP32 control was stable.
4. On one idle H20, run five alternating fresh-process pairs with identical persisted strong/weak inputs, 100 warmups, and at least 1,000 measured complete synchronized steps per arm. Include H2D, zero-grad, forward, loss, backward, SGD, and synchronization; combine strong/weak means 80/20. Require candidate/control weighted mean at most `1.345`, every pair below `1.38`, trial-mean CV below 2%, candidate p95 below `1.45x` control mean, peak allocation below 1.25 GiB, and projected exposure at least **20,000 updates** (`74.35%` of EXP-010, 2.56M images or 51.2 dataset passes).
5. Project startup plus 300 counted seconds, loader transition, and unchanged per-epoch evaluation below 540 seconds. The slower candidate will naturally have fewer epochs and test looks; do not add evaluations or change the evaluator. Require at most one evaluation per epoch and preserve the terminal evaluation.

The 20,000-update floor projects about 16,000 strong and 4,000 weak updates—roughly ten weak epochs. It is slightly stronger than EXP-007's successful 70.76% relative retention and prevents a severely underoptimized wider model from consuming the scored run.

## Risks

- Removing one block from every stage reduces nonlinear depth and late-stage feature refinement; width cannot guarantee replacement of hierarchical composition or receptive-field use.
- Width-1 to width-2 improvement need not extrapolate. The accepted width-2 model may already sit near the capacity/time frontier, while this candidate adds 43.4% parameters and about 45.6% MACs.
- Fixed `1e-4` decay and LR may regularize/optimize the new parameterization differently. EXP-008 shows that increasing decay to `5e-4` severely suppresses strong fit, so no decay rescue belongs in this test.
- Fewer steps and dense-tail epochs may leave the larger classifier or BN statistics less refined. Conversely, fewer evaluation looks are a conservative scoring disadvantage.
- Architecture construction consumes a different RNG count before loader iteration, so the fixed-seed augmentation stream may differ. Do not add RNG alignment as a third mechanism; report the result as the normal seed-42 net architecture effect.

## Production success and falsification

Run once only after every gate passes, on exactly one 98-GB H20, with `uv run train.py > run.log 2>&1`; no retry, alternate width/depth, precision change, LR/decay adjustment, or seed reroll. Require 300 counted seconds, total below 600 seconds, one 80% loader switch with eight workers stopped, expected target formats and CutMix share, at least 20,000 actual steps, exactly 1,540,474 parameters, and all numeric summary fields.

Accept only if `best_test_acc >= 94.25%` and integrity/exposure conditions pass. A timing or safety-gate failure is an invalid/no-go for this exact width-3 ResNet-14 point. A valid run below 94.25% falsifies the depth–width trade under the accepted recipe; use switch fit, first-weak accuracy, NLL, and exposure to distinguish capacity saturation from depth loss or underoptimization, but do not rescue it within EXP-023.
