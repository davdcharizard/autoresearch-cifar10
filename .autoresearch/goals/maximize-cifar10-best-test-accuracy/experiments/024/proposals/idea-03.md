# Proposal: Depth-Preserving Final-Stage Widening to 160 Channels

## Intervention and hypothesis

Preserve the accepted nine-block, 19-convolution postactivation ResNet-20 and widen only `layer3` from 128 to 160 channels. The resulting stage allocation is **32/64/160**: the stem, all three `layer1` blocks, all three `layer2` blocks, and the number and ordering of residual blocks remain unchanged. The `layer2 -> layer3` transition remains the existing stride-2 Option-A slice/pad shortcut; it pads 96 channels rather than 64. The classifier changes from `Linear(128, 10)` to `Linear(160, 10)`.

Preserve every other accepted EXP-010 choice: FP32/default-TF32 execution, batch 128, standard SGD momentum 0.9, all-parameter coupled decay `1e-4`, the 80% high-LR hold and cosine tail, N1/M7 plus probability-0.5 alpha-1 CutMix during the strong phase, crop/flip hard-label weak tail, seed 42, worker lifecycle, timer, and evaluator.

**Primary hypothesis:** extra capacity allocated only to 8x8 semantic features will raise `best_test_acc` from 94.15% to at least the required **94.25%** while retaining the iterative depth and substantially more fixed-time exposure than EXP-023. Point prediction: paired step ratio `1.06-1.10`, 24.4k-25.4k optimizer steps, and `best_test_acc` around **94.25-94.30%**. This is a net fixed-time architecture test, not an equal-compute proof that late width is independently causal.

## Why this is the most credible stage allocation

The local evidence supports additional channels but rejects removing depth. Width-2 ResNet-20 gained 1.25 points in EXP-007 despite 29.2% fewer updates than width 1. EXP-023's width-3 ResNet-14 remained numerically healthy, entered the weak tail well, and used H20 kernels efficiently, but peaked at 94.00% after deleting one block from every stage. Its report therefore recommends preserving ResNet-20 depth and testing stage-specific capacity rather than concluding that all width above 2 is harmful.

`layer3` is the most defensible place to spend the next channels. Its features are closest to the classifier and operate at 8x8 resolution, so added weights cost much less per image than widening the 32x32 or 16x16 stages. Keeping 32/64 through `layer2` also leaves the accepted low- and mid-level augmentation processing bit-for-bit structurally unchanged. This is a narrower claim than full width 3 and a cleaner test than compensating for width by deleting blocks.

Three nearby allocations define the choice:

| Stage widths | Parameters | Approx. conv MACs/image | Relative MACs | Assessment |
|---|---:|---:|---:|---|
| 32/64/128 (accepted) | 1,073,962 | 161.32M | 1.000x | Control |
| 32/64/144 | 1,279,370 | 174.44M | 1.081x | Low cost, but only 12.5% more final channels and less favorable 16-channel alignment |
| **32/64/160** | **1,507,818** | **189.04M** | **1.172x** | Selected knee: meaningful 25% late-width increase with 32-channel alignment |
| 32/64/192 | 2,033,834 | 222.66M | 1.380x | Too close to a costly width-3 retry and pads twice as many transition channels as it preserves |

The selected point has almost the same parameter count as EXP-023's 1,540,474-parameter model, but retains all three blocks per stage and has only about 80.5% of its estimated 234.9M MACs. EXP-023 measured 1.456x MACs at only 1.163x H20 step time, showing that MAC ratios are conservative here; fresh paired timing remains decisive.

## Exact architecture and arithmetic

The candidate must contain:

- stem `Conv2d(3,32,3)` and stages `32/64/160`, three `BasicBlock`s per stage;
- nine residual blocks and 19 `Conv2d` layers total, exactly as in accepted ResNet-20;
- stage outputs `[N,32,32,32]`, `[N,64,16,16]`, and `[N,160,8,8]`;
- two parameter-free Option-A transitions, with channel pads of 32 and 96;
- unchanged global average pooling and `Linear(160,10)`;
- exactly **1,507,818 trainable parameters**: 1,503,072 convolution weights, 3,136 BN affine parameters, and 1,610 classifier parameters.

The change adds 433,856 parameters (+40.40%) but only about 27.72M convolution MACs/image (+17.18%) because every extra spatial convolution is at 8x8. It does not add a residual block, an attention path, a normalization layer, or any new sequential operator. Peak memory should remain far below 1 GiB on the 97,871-MiB H20; the accepted model used 598.7 MiB, and early-stage activations are unchanged.

## Minimal implementation

Keep `WIDTH_MULTIPLIER = 2` as the accepted base-width contract and add one explicit `FINAL_STAGE_CHANNELS = 160` override. Extend `ResNet.__init__` with an optional final-stage width, derive `c1=32` and `c2=64` exactly as today, and use the override only for `c3`. Instantiate production as `ResNet(NUM_BLOCKS, NUM_CLASSES, WIDTH_MULTIPLIER, FINAL_STAGE_CHANNELS)`. A preflight control should remain `ResNet(3, 10, 2)` and the candidate should be `ResNet(3, 10, 2, 160)`.

Do not generalize this experiment into arbitrary per-block widths or add a learned transition. In particular, do not widen `layer2`, add a second in-stage transition, alter Option-A, retune decay/LR, change precision or memory format, or add ECA. Those would obscure whether static depth-preserving late capacity is valuable.

## Structural and numerical safety gates

Before production, require all of the following:

1. Assert exact stage shapes, nine-block/19-convolution counts, parameter count 1,507,818, classifier input 160, exactly two Option-A paths, pads 32/96, and unchanged postactivation ordering. Confirm FP32 parameters/buffers, unchanged optimizer group membership and hyperparameters, and no tracked-file change beyond the reviewed `train.py` architecture lines.
2. Exercise fresh hard- and probability-target forward/loss/backward/update paths. Require finite logits, loss, gradients, parameters, buffers, and momentum; every trainable tensor receives an expected finite gradient; BN batch counters advance once; evaluation emits finite `[N,10]` logits without mutating model state.
3. Reuse the EXP-023-style immutable production-distribution protocol: persist at least 200 exact post-transform batches (100 hard and 100 soft), hash the corpus, and train fresh accepted and candidate processes on the identical sequence. Require no candidate-only class concentration above 95%, no nonfinite state, and candidate terminal loss EMA no more than 1.5x control. Record normalized gradient/update ratios and per-stage norms as diagnostics, but do not veto merely because unequal parameterizations yield different ratios unless a predeclared catastrophic threshold (for example >2x control) is crossed.
4. Explicitly inspect the 64 retained shortcut channels and 96 zero-padded channels at the second transition. The larger `c3/c2=2.5` ratio is the candidate's main architectural risk; verify shapes and gradients, not just successful execution.

Seed-only forkserver replay is insufficient after EXP-019/021, so persisted tensors and their SHA-256 must be retained in the experiment evidence. EXP-023 found width-3 FP32 safe, but this asymmetric Option-A geometry is new and still earns a fresh candidate/control check.

## H20 timing and exposure gates

On exactly one otherwise-idle H20, benchmark control and candidate with five alternating fresh-process pairs after conditioning. Each arm must use the same persisted strong/weak tensors, at least 100 warmup steps, then 800 strong and 200 weak complete synchronized steps including H2D, zero-grad, forward, loss, backward, SGD, and synchronization. Weight the means 80/20.

Proceed only if:

- overall candidate/control weighted mean is at most **1.12** and every pair is below **1.15**;
- both trial-mean CVs are below 2%, and candidate p95 is below 1.20x the paired control mean;
- projection from EXP-010's 26,898 updates is at least **24,000 updates** in 300 counted seconds;
- peak allocation is below 1.0 GiB;
- projected startup, training, phase transition, unchanged at-most-once-per-epoch evaluations, and summary finish below 540 seconds total.

The 24,000-step planning floor retains 89.23% of accepted exposure, about 3.07M examples or 61.4 dataset passes. At the unchanged 80/20 time split it should still provide roughly 4,800 weak-tail updates (about 12.3 epochs). This directly protects against explaining another miss by an unnecessarily short refinement tail. Timing failure invalidates this exact 160-channel point; do not silently fall back to 144 channels inside EXP-024.

## Production verification and falsification

After all gates pass, run once with `uv run train.py > run.log 2>&1` on the sole idle approximately 98-GB H20. Do not retry, reroll, or adjust channels after seeing results. Require:

- exactly 300 counted training seconds, total runtime below 600 seconds, and a complete numeric summary;
- exactly 1,507,818 parameters, one 80% strong-to-weak switch, eight workers stopped, expected hard/soft target formats, and realized CutMix near 50%;
- at most one evaluation per epoch with the existing checkpoints and terminal evaluation;
- at least **23,500 actual optimizer steps**, leaving margin around the 24,000 timing projection while still retaining 87.37% of accepted exposure;
- `best_test_acc >= 94.25%` for acceptance.

Record the 80% switch accuracy, first weak checkpoint, peak/final accuracy and NLL, steps, epochs, evaluation count, and peak VRAM. A switch below the existing 87.08% underfit marker would mean asymmetric width impaired strong optimization. Healthy switch fit but weak first-tail conversion below EXP-010's 93.16% would implicate the widened transition or classifier. Matching early-tail conversion followed by a lower peak/final result would reject this static late-capacity allocation as a generalization improvement, even if training loss is lower.

## Risks and interpretation limits

- The second Option-A shortcut now preserves only 40% of output channels and zero-pads 60%, versus a balanced 50/50 split in the accepted model. This may weaken identity propagation even though depth and operator order are preserved.
- Extra static channels may reduce training loss without improving generalization; EXP-023's late regression cautions that parameter count is not the limiting quantity by itself.
- Width consumes a different initialization RNG count and therefore can alter subsequent shuffle/augmentation draws under the fixed global RNG flow. Do not reseed or realign the stream after observing the result; report the ordinary fixed-seed net architecture effect.
- The 32/64/160 point changes both final-stage capacity and fixed-time exposure. Timing and trajectory diagnostics separate plausible mechanisms but do not make the comparison equal-compute.
- Wide Residual Networks supports width as a broad CIFAR direction, not this asymmetric postactivation, Option-A, 300-second operating point. The local EXP-007/023 evidence is more relevant than its headline accuracy.

## Decision rule

Accept only if every integrity/runtime condition passes and `best_test_acc >= 94.25%`. A safety or timing failure is an invalid/no-go for 32/64/160. A valid run below 94.25% falsifies depth-preserving final-stage widening at this allocation; do not rescue it with ECA, learned shortcuts, width 144/192, or hyperparameter changes inside the experiment.

If it fails with healthy switch fit and full exposure, prefer a conditional-capacity idea such as identity-scale final-stage ECA rather than another static-width increment. If it fails specifically through Option-A transition dilution, a learned projection is not a direct rescue because EXP-017 already found learned pool-first transitions harmful; any alternative transition would require a new independent hypothesis and review.

## Sources

- `experiments/007/04-analysis.md` — accepted width-2 capacity result and measured exposure trade.
- `experiments/010/04-analysis.md` — current 94.15% recipe, switch/tail diagnostics, and accepted model facts.
- `experiments/023/04-analysis.md` and `experiments/023/timing-report.json` — depth-loss diagnosis and sublinear H20 scaling of wider CIFAR kernels.
- `knowledge/papers/wide-residual-networks.md` — width/depth motivation and transfer caveats.
