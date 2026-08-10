# Proposal 03: BF16-Funded Width-3 Postactivation ResNet-20

## Proposal

Spend H20 BF16 throughput on a single fixed capacity increase: change the accepted width multiplier from 2 to 3 and run its training forward plus cross-entropy under CUDA BF16 autocast. Keep FP32 master parameters, gradients, BatchNorm persistent state, ordinary SGD, and FP32 evaluation.

```python
WIDTH_MULTIPLIER = 3

optimizer.zero_grad()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
loss.backward()
optimizer.step()
```

This is deliberately a two-part resource exchange, not an attribution experiment. Width 3 supplies the accuracy mechanism; BF16 must fund enough updates to make that capacity usable inside 300 counted seconds. Test exactly this operating point once. Do not fall back to width 2, width 2.5, width 4, FP32 width 3, FP16, a scaler, channels-last, compilation, or altered optimizer/data settings if any gate fails.

## Evidence and Rationale

EXP-007 established the strongest local architecture result: width-2 postactivation ResNet-20 improved best accuracy by 1.25 points over width 1 despite retaining only 70.76% of its updates. Its strong-view switch checkpoint rose 5.48 points, localizing the benefit to representation capacity under N1/M7 rather than extra optimization. EXP-010 then added p=0.5 CutMix and raised width 2 to the current 94.15% frontier with 26,898 steps, 89.73% at the switch, 93.16% at the first weak checkpoint, and final equal to best.

Later failures sharpen the design. Decay changes, stronger CutMix, full preactivation, and selective zero-gamma either reduced strong fit or failed the metric gate. EXP-015's selective zero-gamma retained equal compute but finished at 93.80% after lowering switch fit by 3.25 points. Width 3 instead preserves the exact accepted postactivation ordering and initially active residual branches while increasing feature capacity.

The H20 profile attributes 2.408 ms to forward and 8.220 ms to backward: together 97.57% of CUDA-stage time, with only 0.034 ms of visible wall/launch gap. BF16 is therefore relevant only if lower-precision convolution/backward kernels save measured GPU time. BF16 alone was correctly criticized in EXP-015 because more accepted width-2 updates had no demonstrated accuracy mechanism; pairing it with the locally successful width direction addresses that critique, at the cost of intentionally weaker attribution.

## Exact Model and Parameter Count

Set only `WIDTH_MULTIPLIER = 3`. The existing generator yields channels `48/96/192`, preserving:

- nine postactivation `BasicBlock`s and 19 convolutions total;
- accepted `Conv-BN-ReLU-Conv-BN-add-ReLU` block ordering;
- seven identity shortcuts and two raw Option-A slice/zero-pad transitions;
- global average pooling and a `192 -> 10` linear classifier;
- existing Kaiming initialization, BatchNorm defaults, and all state semantics.

The exact parameter count is **2,412,730**, versus 1,073,962 for width 2 (`2.2466x`). No depth, shortcut, kernel, pooling, branch initialization, or classifier mechanism changes. Peak width-2 allocation was only 598.7 MiB on a 97,871 MiB H20, so capacity is not a memory-limit hypothesis.

Keep the complete EXP-010 statistical recipe unchanged: batch 128; LR `0.1` through 80% elapsed time; step to `0.01` then cosine to `1e-4`; momentum 0.9; coupled all-parameter decay `1e-4`; N1/M7 and p=0.5 alpha-1 CutMix through 80%; hard weak crop/flip tail; seed 42; persistent workers; `MAX_STEPS=64000`; and the accepted evaluator/checkpoint policy.

Width-3 construction consumes a different number of initialization RNG draws and therefore may alter the later seeded loader stream when workers start. Do not add RNG-fork alignment as a third mechanism. The fixed seed remains 42, no reroll is allowed, and the result is the net width-3/BF16 method under the repository's normal construction order.

## BF16 and FP32 State Semantics

PyTorch 2.9 recommends wrapping forward and loss in autocast and running backward outside it. CUDA `conv2d` and `linear` are lower-precision-eligible; `cross_entropy` is FP32-policy; backward ops use the dtype chosen for their corresponding forward ops. Do not manually cast model or inputs. Source: [PyTorch 2.9 AMP documentation and CUDA op policy](https://docs.pytorch.org/docs/2.9/amp.html).

- Construct width 3 normally in FP32 and move it with `.to(device)`, never `.bfloat16()`.
- FP32 CUDA inputs enter the autocast region. Eligible width-3 convolutions and linear execute with BF16 operands and are expected to expose BF16 outputs.
- BatchNorm is not explicitly force-listed by CUDA autocast. Its visible dtype follows its input path; record observed inputs/outputs, but do not claim or force an undocumented internal reduction dtype. BN affine parameters, running mean/variance, and all master parameters remain FP32; `num_batches_tracked` remains integer.
- Hard labels remain int64; CutMix probability labels remain FP32. `F.cross_entropy` must return a finite FP32 scalar for both.
- Exit autocast before backward. Gradients accumulated into FP32 leaf parameters must be FP32.
- Keep ordinary unfused SGD outside autocast. Momentum buffers and coupled-decay arithmetic remain FP32, and no update may be skipped.
- Do not use `GradScaler`. BF16 has FP32-like exponent range; a scaler adds state, checks, possible skipped steps, and a third mechanism. Any non-finite value is a hard candidate failure.
- Keep every evaluation outside autocast. `Eval.evaluate(model, device)` receives the FP32 width-3 model and runs the unmodified evaluator's FP32 path.

Add only a CUDA/BF16 capability assertion and an optional one-line provenance message. Do not add production dtype hooks or warmup outside the timer.

## Production-Distribution Numerical Gate

Before timing, use materialized N1/M7 production batches, not Gaussian pixels; EXP-015 showed out-of-distribution safety inputs can produce false collapse signals. Cover both hard int64 and valid CutMix probability targets.

Create two width-3 models and optimizers from the same cloned FP32 state. Compare FP32 and BF16-autocast arms after resetting model/optimizer state for each of at least 20 paired batch-128 inputs. Require:

- exactly 2,412,730 FP32 parameters in both arms; all populated parameter gradients, BN floating buffers, and SGD momentum buffers stay FP32;
- all observed convolution/linear outputs and final logits are BF16 under autocast; representative BN dtypes and all persistent state dtypes are recorded; loss is FP32 in hard and soft paths;
- finite logits, loss, gradients, parameters, BN buffers, and optimizer state;
- per-batch BF16/FP32 relative loss error at most 2%, FP32-cast logit cosine at least 0.995, and gradient-vector cosine at least 0.99;
- BF16/FP32 gradient-norm and one-step update-norm ratios each in `[0.90, 1.10]`, with update cosine at least 0.99;
- the BF16 arm increases exactly-zero gradient elements by no more than 1 percentage point;
- BN counters match exactly and normalized L2 differences in running mean/variance remain at most 2% after the paired step;
- after 200 paired production-distribution training steps, BF16 has no loss above 2x same-step FP32 loss, no class concentration above 95% when the FP32 control is below it, and no non-finite or skipped update.

These are catastrophic numerical guards, not accuracy-tuning gates. A failure retires the exact combination; do not enable scaling or narrow the autocast region as a rescue.

## Three-Arm Fresh-Process Timing Gate

Benchmark three arms to prove both parts of the resource exchange:

- **A:** accepted width-2 FP32;
- **B:** width-3 FP32;
- **C:** proposed width-3 BF16.

Use five fresh-process triplets on one idle H20. Rotate order with a balanced schedule so each arm appears early/middle/late across trials. Each arm constructs a fresh seeded model/optimizer, runs 100 unmeasured warm steps for cuDNN/autocast caches, then measures at least 500 synchronized batch-128 steps. Alternate real-shaped hard and probability targets. Use identical input tensors within each triplet and cloned width-3 state for B/C.

Time the exact production interval: pinned H2D copy, unchanged zero-grad, forward, cross-entropy, backward, SGD, and final synchronize. Separately record CUDA-event transfer, forward, loss, backward, and optimizer stages. Report every trial's mean, median, p95, images/s, peak allocation, and paired ratios.

All gates are conjunctive:

1. **BF16 funding:** C synchronized step time is at most `0.86957x` B in the median (`>=1.15x` width-3 speedup), and C is at least 1.12x faster than B in every triplet.
2. **GPU-stage mechanism:** C `(forward + backward)` CUDA-event time is at most `0.85x` B, backward alone is at most `0.90x` B, and at least 90% of absolute CUDA-stage savings comes from forward/backward.
3. **Usable capacity exposure:** project from accepted actual exposure as `floor(26,898 * median_step_A / median_step_C)`. Require at least **22,863 updates**, exactly 85% of EXP-010, or 2,926,464 presented images / 58.5 dataset passes. Equivalently, C must be no slower than about `1.17647x` A.
4. **Stable tails:** C p95 step time is no more than 1.25x A median; trial-mean CV is below 3% per arm and ratio CV below 2%; no order trend reverses either funding or exposure result.
5. **Memory:** C peak allocation is below 2 GiB, no allocation grows monotonically across steps, and the model fits with broad H20 margin.

The 85% floor is deliberately stronger than width 2's successful 70.76% retention. It projects about 18,290 strong-phase and 4,573 weak-tail updates, enough for roughly 11-12 full weak epochs while testing 2.25x as many parameters. A lower floor would make a miss uninterpretable as capacity starvation. Conversely, this experiment does not require matching width-2 exposure: EXP-007 proves useful capacity can dominate some update loss.

If C passes versus A but fails to accelerate versus B, the “BF16-funded” mechanism is false and the experiment does not run. If C accelerates B but misses 22,863 projected updates, BF16 does not fund enough width and the experiment does not run. No threshold relaxation or width/precision tuning is allowed.

## Loader and Evaluation Fairness

Batch size remains 128 and C cannot pass while consuming faster than A by construction, so the accepted loader should have at least its existing headroom. Still measure at least 1,000 real strong batches with C. Require median iterator wait below 10% of C step time, p95 below 20%, successful CutMix hard/soft provenance, and exact shutdown of all eight strong workers before one hard weak batch.

Preserve the accepted evaluation rule exactly. The slower width-3 candidate will complete fewer epochs and therefore likely receive fewer dense-tail test looks; do not add evaluations, convert to fixed-time checkpoints, or evaluate FP32/BF16 variants separately. Require at most one evaluation per epoch and unique evaluated epochs. A win with fewer opportunities remains valid; fewer looks are a conservative disadvantage, not permission to change the harness.

Because scoring width 3 in FP32 is slower, benchmark the unmodified batch-256 evaluator shape including its final partial batch. Project cold startup + 300 counted training seconds + loader switch + all expected width-3 FP32 evaluations below 540 seconds, leaving 60 seconds before the hard 600-second kill. Preflight warmups remain disposable and never move outside the production timer.

## Accuracy Hypothesis and Failure Modes

**Primary hypothesis:** BF16 width 3 retains at least 22,863 updates under the complete EXP-010 recipe, preserves strong-phase fit, and raises FP32-evaluated `best_test_acc` from 94.15% to at least 94.25%.

The capacity rationale is local but not monotonic proof. Width 2 gained 1.25 points while losing 29.2% of updates, and width 3 at the timing floor loses only 15% relative to width 2. However, width 2 may already sit near the optimal capacity/time frontier; width 3 has 2.25x parameters, less optimization per parameter, and the same scalar decay/LR.

Pre-register 87.08% as the recurring strong-underfit marker and 89.0% as a healthier expectation. Falling below either is diagnostic only and cannot trigger a retry or alter the formal metric verdict. Compare first weak accuracy, tail slope, final/best gap, and final NLL to EXP-010.

Main risks:

- **Capacity starvation:** 22,863 updates may still be insufficient for 2.41M parameters, especially under N1/M7 plus CutMix.
- **Width saturation/overfit:** the width-1 to width-2 result need not extrapolate; `1e-4` decay may regularize width 3 differently, but changing it is forbidden after two failed width-2 decay variants.
- **BF16 gradient drift:** reduced mantissa precision changes convolution/residual gradients and BN-observed activations even with FP32 master state and loss.
- **FP32 already accelerated:** TF32/cuDNN may leave less than 15% BF16 benefit; the exact combination then stops at timing.
- **Two-mechanism attribution:** a success validates the net BF16-width3 method, not BF16 or width 3 alone. The B timing arm establishes funding but does not provide an accuracy ablation.
- **Fewer evaluations:** fewer epochs mean fewer max-metric observations and possibly a shorter weak tail. No extra evaluations are allowed.
- **Initialization/data divergence:** larger initialization consumes more seeded RNG before loader iteration. Seed 42 remains fixed, but streams are not batch-aligned with EXP-010.
- **Wall time:** FP32 width-3 evaluation is excluded from training time but included in the 600-second supervisor.

## One-Run Verification

If and only if numerical, timing, loader, lifecycle, and wall projections pass:

1. Confirm the moving baseline is 94.15% at `7c1e7d8`, so improvement requires 94.25%.
2. Confirm exactly one idle NVIDIA H20 with approximately 98 GB VRAM, compute capability 9.0, and `torch.cuda.is_bf16_supported()`.
3. Verify only `train.py` differs and the diff contains exactly width 3, the BF16 capability/provenance, and the training forward/loss autocast context. Run syntax/lint/pre-commit.
4. Assert channels `48/96/192`, 19 convolutions, nine accepted postactivation blocks, two Option-A transitions, exactly 2,412,730 FP32 parameters, unchanged optimizer groups, batch 128, transforms, CutMix, LR schedule, seed, timer, workers, evaluator, and summary fields.
5. Remove stale `run.log`; launch once as `uv run train.py > run.log 2>&1` on the pinned sole H20 under the 600-second supervisor. No valid-run retry or alternate operating point is allowed.
6. Require exit 0, approximately 300 counted seconds, total below 600, finite standard summary, unchanged parameter count, and no non-finite values.
7. Require one strong-to-weak switch near 80%, all eight old workers stopped, realized CutMix near 50%, integer weak targets, at most one evaluation per epoch, and no duplicated evaluation epoch.
8. Require at least 22,863 actual updates / 2,926,464 images to support the funded-capacity mechanism. Record actual exposure ratio, strong/tail step counts, epochs, evaluation count, peak memory, startup, and total wall time.
9. Require `best_test_acc >=94.25%` for improvement. Report switch accuracy, first weak checkpoint, best/final, final NLL, and trajectory relative to EXP-010.
10. Remove `run.log` after analysis. Restore accepted `7c1e7d8` on every no-go or no-improvement outcome.

## Decision Rules

- **Preflight no-go:** any numerical, dtype/state, funding, stage-attribution, 85%-exposure, loader, lifecycle, or wall gate failure blocks the full run.
- **Accept:** all integrity conditions pass, actual exposure is at least 22,863 updates, and accuracy is at least 94.25%.
- **Accuracy miss after valid run:** reject the net combination. Do not separately retry width-3 FP32, another BF16 width, a different scaler, or changed regularization.
- **Accuracy pass below exposure floor:** formal metric evidence must be reported honestly, but the declared funded-capacity mechanism is unsupported; mandatory adversarial analysis decides categorization without a rerun.
- **Non-finite or scope/evaluator/seed/timing violation:** invalid and revert; no fallback.

## Recommendation

This is a high-upside, high-risk finalist. It spends measured mixed-precision potential on the only architecture lever with a large local gain, while retaining accepted postactivation branch activity and batch-128 noise. The three-arm gate prevents BF16 from receiving credit for width-3 speed that it did not fund and prevents a severely underexposed width 3 from consuming the one fixed-seed run. If it clears 15% BF16 funding and 85% accepted exposure, the local width result makes one accuracy run defensible; otherwise stop without substitution.
