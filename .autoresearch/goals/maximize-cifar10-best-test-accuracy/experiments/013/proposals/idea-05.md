# Proposal: Fixed Average-Max Final Spatial Aggregation

## Decision

Change only the accepted model's final spatial aggregation. Preserve the 128-wide classifier and replace pure global average pooling with an equal, parameter-free blend of global average and global maximum features:

```python
POOL_MAX_WEIGHT = 0.5

avg_features = F.adaptive_avg_pool2d(out, 1).flatten(1)
max_features = F.adaptive_max_pool2d(out, 1).flatten(1)
features = torch.lerp(avg_features, max_features, POOL_MAX_WEIGHT)
return self.fc(features)
```

`torch.lerp(avg, max, 0.5)` pins the feature to the arithmetic midpoint, modulo ordinary FP32 operation ordering. Use `adaptive_max_pool2d`, whose backward routes each channel's max gradient through its selected index; do not substitute `torch.amax`, concatenate statistics, learn a blend coefficient, normalize either statistic, or tune 0.5 after measurement.

The one-run hypothesis is that retaining strong localized evidence alongside spatially distributed evidence will improve EXP-010's 94.15% frontier to at least the 94.25% acceptance threshold, with a point prediction of **94.30%**, while retaining at least 97% of its 26,898 updates.

## Why a Fixed Blend Instead of Concatenation

Concatenating 128 average and 128 maximum features would resize the classifier from `128 -> 10` to `256 -> 10`, adding 1,280 weights and changing classifier capacity, fan-in scaling, initialization draws, post-construction RNG state, and potentially the subsequent DataLoader shuffle/worker seeds. A gain could then come from pooling, capacity, initialization, or a changed stochastic trajectory.

The fixed convex blend keeps the exact classifier module and every parameter tensor. It asks the narrower question: does a different spatial statistic improve how the already-validated representation is read out? The average half preserves dense, area-sensitive evidence; the maximum half adds peak salience. The 50/50 operating point is symmetric and predeclared, not presented as optimal. A learned scalar would add capacity and could collapse to either endpoint; it is deliberately excluded.

For a spatially constant channel, average equals maximum and the candidate is exactly identical to accepted pooling. For a localized response, the blend moves between its area-weighted mean and peak without discarding either statistic. Because the last accepted activation is post-add ReLU, both statistics are nonnegative and the blended feature remains within each channel's observed `[mean, max]` range.

## Local CutMix Mechanism

EXP-010 gained 0.60 points by applying alpha-1 CutMix to half of N1/M7 plateau batches, with near-identical compute exposure. That is direct local evidence that class-bearing spatial regions improve this model. Pure global average pooling scales a detector's contribution approximately with its activated area, so a compact donor object or discriminative part can be diluted across the final 8x8 map. The max component preserves the strongest channel response even when it occupies one or a few cells, while the average component continues to encode extent and distributed support.

This is plausible, not automatically aligned with CutMix. CutMix target weights are proportional to patch area, whereas maximum pooling is largely area-insensitive. A tiny donor patch that produces one strong response could be overrepresented relative to its soft target. The fixed average path reduces but does not eliminate that mismatch. N1/M7 can also create isolated augmentation artifacts that max pooling amplifies. The hard weak tail may recalibrate the unchanged classifier to ordinary full-image responses, but no special transition or loss change is allowed to force that outcome.

## Exact Scope and Identity

Keep the complete EXP-010 model and recipe unchanged except the three pooling lines:

- width-2 postactivation ResNet-20, nine accepted blocks, raw/Option-A shortcuts, stem, BNs, ReLUs, and 128-to-10 classifier;
- exactly 1,073,962 trainable parameters and the existing state-dict keys/shapes;
- seed 42 and all current default/Kaiming initialization;
- batch 128; N1/M7 and alpha-1 CutMix probability 0.5 through 80%; hard weak crop/flip tail;
- SGD momentum 0.9, all-parameter coupled decay `1e-4`, and the accepted elapsed-time LR schedule;
- loader/collator RNG handling, persistent-worker lifecycle, timer, synchronization, evaluator cadence, evaluator, maximum-step guard, and summary.

No module or parameter is added. From identical seed-42 construction, every parameter and buffer, parameter object order, optimizer group, post-construction CPU/CUDA RNG state, and subsequent loader seed must be bitwise identical to control. Adaptive average pool, adaptive max pool, and `torch.lerp` consume no RNG. Shared forward activations must remain bitwise identical through `layer3`; only final aggregation and its downstream gradients differ.

Do not combine this with a wider classifier, dropout, attention, zero-gamma, preactivation, projection shortcuts, compilation, larger batches, pooling normalization, or a learned/fixed coefficient other than 0.5.

## Cost Model

The accepted final activation has shape `[128, 128, 8, 8]`, or 1,048,576 FP32 values. The candidate retains the existing average reduction and adds:

- one adaptive maximum reduction over those approximately 1.05M values;
- one 16,384-element `torch.lerp` pointwise operation;
- one sparse max-pool backward that routes gradient through 16,384 selected indices;
- saved max indices of approximately `128 * 128 * 8` bytes, or 128 KiB if represented as int64, plus small output/autograd tensors.

Classifier work and parameter-update work are unchanged. Expected peak allocation should remain under approximately 605-610 MB versus EXP-010's 598.7 MB. Arithmetic is negligible beside 19 convolutions, but launch count and reduction latency cannot be inferred from FLOPs. The rejected all-block SE design added only modest arithmetic yet measured a 1.23324x training ratio and projected just 21,810 steps because nine reductions and many tiny sequential kernels were launch-bound. This candidate places one reduction and one pointwise operation only at the network endpoint, so its expected overhead is much smaller, but the same paired H20 discipline is mandatory.

## Gradient Semantics

For a channel with a unique maximum on an `H x W` map and upstream feature gradient `g`, the 50/50 blend gives:

```text
non-maximum location: 0.5 * g / (H*W)
selected maximum:     0.5 * g / (H*W) + 0.5 * g
```

At the accepted 8x8 endpoint, every location retains a dense `g/128` contribution from the average path, while the selected maximum receives an additional `g/2`; its gradient is 65 times a non-maximum location's gradient. Thus the candidate is not pure sparse max pooling, but it strongly concentrates half the readout signal on one position per channel.

This concentration is both mechanism and risk. It can sharpen localized class detectors, but it can reduce pressure for spatial coverage, make gradients sensitive to crop/flip/RandAugment-induced extrema, and switch discontinuously when two positions exchange rank. Ties use the exact index semantics of `adaptive_max_pool2d`; do not change the operation after observing tie behavior. The full gradient must remain finite for hard and probability targets, and the dense average component must be verified rather than assumed.

## Structural and Functional Gates

Before H20 timing, run disposable-process tests without changing tracked files:

1. Construct paired accepted/candidate models from reset seed 42. Require exactly 1,073,962 parameters, identical state-dict keys, tensor values, object ordering, buffers, optimizer membership, and post-construction CPU/CUDA RNG states.
2. Hook stem and all three residual stages. Require bitwise-identical activations through `layer3` and candidate logits exactly equal to control for spatially constant final feature maps.
3. On seeded nonconstant `[128,128,8,8]` features, compare candidate pooled values to an explicit FP32 `avg + 0.5 * (max - avg)` reference and require every value to lie between the corresponding mean and maximum.
4. Use a unique-maximum synthetic tensor and fixed upstream gradient to prove all 64 spatial positions receive the declared dense average gradient and only the chosen position receives the extra max gradient. Require the selected/nonselected gradient ratio to match 65 within FP32 tolerance.
5. Test a tied-maximum tensor and record the installed PyTorch adaptive-max index behavior. Require repeatability within a process and no RNG-state change; do not impose `torch.amax`'s distributed-tie semantics.
6. Run full forward/backward/SGD smoke steps with hard `[128]` and CutMix probability `[128,10]` targets. Require `[128,10]` finite logits/losses and finite nonzero gradients for every Conv/BN/classifier parameter.
7. Require no extra evaluator call, no timer-boundary change, and only the final aggregation plus one named constant in the tracked diff. Compile, Ruff, and pre-commit must pass.

Any shape, RNG, shared-state, gradient, tie-repeatability, target, scope, or evaluator failure is a no-go. Fix only implementation faults; do not change the blend or architecture.

## Paired H20 Timing Gates

On one idle 97,871 MiB H20, run five alternating fresh-process control/candidate pairs with cloned seed-42 model state and fresh identical SGD state. Use the exact accepted batch-128 hard/probability-target path, 100 warmups, and at least 500 synchronized complete training steps per trial. Record trial means, medians, p95, CV, images/s, and peak allocation.

Require:

- candidate/control median-of-trial-means training ratio `<=1.03`;
- projected exposure `floor(26,898 * control_time / candidate_time) >=26,091`, retaining 97% of EXP-010;
- CV of trial means `<=2%` for each model and candidate p95 no more than 1.08x control;
- candidate peak allocated memory `<620 MB` and no more than 16 MB over paired control;
- finite losses and gradients throughout.

Benchmark evaluation separately using `model.eval()`, `torch.inference_mode()`, 100 warmups, and at least 500 synchronized forwards in the same five-pair design. Require candidate/control inference ratio `<=1.05`, CV `<=2%`, and a conservative total-runtime projection below 540 seconds after charging the incremental cost across EXP-010's 19 evaluator passes.

These gates deliberately measure end-to-end kernels rather than accepting a FLOP argument. Any miss retires this exact blend; do not concatenate, lower max weight, use a learned gate, remove average pooling, or add compilation/mixed precision as a fallback.

## One-Run Hypothesis and Verification

If all functional and timing gates pass, run the exact candidate once at seed 42 with required output redirection.

**Hypothesis:** the fixed average-max blend will preserve both area-weighted and localized class evidence from the accepted CutMix representation, retain at least 26,091 updates, and reach **94.30%** point-estimate best test accuracy, clearing the formal **94.25%** threshold without pushing the 80% strong checkpoint below the 87.08 underfit marker.

Require exit zero, all ten finite summary fields, 300.0 counted training seconds, total below 600 seconds, 1,073,962 parameters, peak allocation below the measured gate, one 80% augmentation/CutMix switch, eight stopped workers, about 50% strong mixed batches, hard weak targets, and no duplicate evaluation epoch. Compare actual steps, strong checkpoint, first weak checkpoint, final NLL, late trajectory, best/final gap, runtime, and VRAM with EXP-010.

The primary decision is unchanged:

- `best_test_acc >=94.25%` with every integrity/timing gate is improvement;
- a valid lower result is no-improvement, with no rerun or blend-weight rescue;
- an accuracy pass with fewer than 26,091 steps is formally above the metric gate but mechanism attribution is timing-confounded and must not be described as a low-cost pooling win;
- a strong checkpoint below 87.08 is augmentation/aggregation underfit evidence only, never an adaptive trigger;
- crash or protocol failure may be fixed only while preserving the exact 50/50 design and seed.

## Failure Mechanisms

- **CutMix area mismatch:** a tiny donor region can dominate a max channel despite receiving a small soft-target weight.
- **Augmentation sensitivity:** RandAugment or crop boundaries can create isolated extrema that the max path amplifies.
- **Sparse-gradient domination:** half the pooling gradient concentrates at one selected location per channel, encouraging brittle parts rather than spatially supported features.
- **Feature-scale shift:** because post-ReLU maximum is normally above mean, the blend raises classifier input magnitude even though the classifier initialization is identical; early logits and effective optimization geometry change.
- **Rank discontinuity and ties:** small activation changes can switch selected spatial indices; repeated zeros or equal responses make gradients backend-index dependent.
- **Lost extent information:** the average half remains, but a 50% maximum weight may still underweight object area and context useful for full CIFAR images.
- **Endpoint launch cost:** one extra reduction, pointwise kernel, saved-index path, and sparse backward may reduce fixed-time exposure more than their arithmetic suggests.
- **Representation mismatch:** EXP-010 validates spatial mixing as a training target, not specifically max-readable localized features; convolutional layers may already spread evidence enough that average pooling is optimal.
- **Single-seed resolution:** the required 0.10 point is ten CIFAR-10 examples. A bare pass is protocol-valid but weak causal evidence and cannot be confirmed by reroll.

## Evidence

- `goals/maximize-cifar10-best-test-accuracy/01-definition.md`: only-`train.py`, one-H20, fixed-seed/time, evaluator, and 94.25% moving acceptance rule.
- `goals/maximize-cifar10-best-test-accuracy/02-system-understanding.md`: 97.6% forward/backward cost, accepted exposure, memory headroom, and final aggregation as an open question.
- `goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md` and `04-results.tsv`: width/CutMix wins and failures from stronger regularization.
- `goals/maximize-cifar10-best-test-accuracy/experiments/010/04-analysis.md`: accepted 94.15% CutMix trajectory, 26,898 steps, 598.7 MB, and class-bearing regional evidence.
- `goals/maximize-cifar10-best-test-accuracy/experiments/012/04-analysis.md`: representation near miss and the need to preserve healthy strong-phase fit.
- `goals/maximize-cifar10-best-test-accuracy/experiments/012/proposals/idea-02.md` plus EXP-012 preflight records: the 1.23324x SE timing failure despite low nominal arithmetic.
- `goals/maximize-cifar10-best-test-accuracy/experiments/013/01-brainstorm.md`: pooling seed, CutMix localization rationale, and requirement for a measured kernel-cost gate.
- `train.py`: accepted postactivation endpoint, 8x8 final map, global average pool, 128-wide classifier, initialization order, RNG, and protocol.
