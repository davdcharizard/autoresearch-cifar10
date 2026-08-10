# Proposal: Scale-Matched Bounded Average–GeM-3 Pooling

## Decision

Test one final-aggregation change on the accepted EXP-010 recipe: replace pure global average pooling with a **single learned, globally shared convex mixture of average pooling and scale-matched GeM-3 pooling**. Keep the accepted classifier and every convolutional feature unchanged.

For the final nonnegative post-ReLU tensor `x` of shape `[N,128,8,8]`, define

```text
a       = GAP(x)
q       = (mean(clamp(x, 1e-6)^3))^(1/3)
s       = stop_gradient(||a||_2 / max(||q||_2, 1e-6))   # per example
q_norm  = s * q
alpha   = (1/8) * sigmoid(mix_logit)
pooled  = (1 - alpha) * a + alpha * q_norm
```

Initialize `alpha` to exactly `1/64` up to ordinary FP32 rounding by setting `mix_logit = log(1/7)`. Thus the salient arm starts at only 1.5625% and can never exceed 12.5%. Implement the scalar in a tiny pooling submodule registered **after** `fc`, so the existing parameter traversal and Kaiming initialization draws remain in their accepted order and the one deterministic scalar is appended last. The exact constants are:

```python
POOL_GEM_P = 3.0
POOL_EPS = 1e-6
POOL_ALPHA_MAX = 0.125
POOL_ALPHA_INIT_LOGIT = math.log(1.0 / 7.0)
```

Use the accepted `adaptive_avg_pool2d(..., 1).flatten(1)` for `a`; compute GeM in FP32. The ratio `s` is detached deliberately: normalization controls the forward scale but must not create a dense, input-coupled norm-gradient path. Do not learn `p`, remove the detach, use raw max, add a second classifier, use per-channel gates, give the scalar a special optimizer group, or tune any constant after measurement.

This adds exactly one parameter, for **1,073,963 total parameters**. Preserve width-2 postactivation ResNet-20, Option-A shortcuts, seed 42, batch 128, N1/M7 plus alpha-1 CutMix on half the strong batches through 80%, the hard weak tail, SGD `lr=0.1`/momentum `0.9`/all-parameter decay `1e-4`, elapsed-time LR schedule, timer, workers, evaluator, and evaluation cadence.

## Why this is the strongest safe pooling variant

The AISTATS mixed-pooling paper supports learning one mixing coefficient per layer rather than committing to a universal fixed statistic, and reports that learned max/average combinations can improve over conventional pooling. Radenovic, Tolias, and Chum's GeM work establishes a differentiable average-to-max continuum, although its direct result is image retrieval rather than CIFAR classification. This proposal uses the evidence conservatively: fixed moderate `p=3` supplies smooth magnitude-weighted spatial gradients, while only a tightly bounded scalar decides how much of that statistic enters the classifier.

It addresses both objections to fixed GeM-3 from EXP-023. First, the candidate can remain near average pooling when the salience prior conflicts with area-proportional CutMix targets. Second, vector-norm matching removes GeM's immediate feature-scale increase instead of merely measuring it after implementation. The scalar is global, not conditioned on target format, so the strong/weak phase transition does not introduce a hidden policy branch.

The local hypothesis is that the accepted representation contains compact class evidence that GAP dilutes, but that only a small learned salience correction is useful. The candidate should retain at least 97% of EXP-010's 26,898 updates, keep the strong-phase checkpoint healthy, and raise `best_test_acc` from 94.15% to at least 94.25%.

## Explicit distinction from EXP-014

EXP-014 added an **independent raw-global-max classifier**. Its empty branch received a classifier gradient 4.10 times the average branch, reached a max/average weight-norm ratio of 1.221 after one update, raised same-batch loss about tenfold, and collapsed every prediction to one class. Zero initial output protected only step zero.

This candidate has no independent classifier and no unbounded max contribution:

- GeM-3 distributes gradient across all activations above epsilon rather than routing it through one argmax.
- Per-example `q_norm` has the same L2 norm as `a` (within FP32 tolerance), and both vectors are nonnegative. Therefore `||pooled||_2 <= ||a||_2` by convexity; the candidate cannot inflate pooled-feature norm.
- At initialization, `||pooled-a||_2 <= 2*alpha*||a||_2 = 3.125%*||a||_2`. Even at the hard cap, the bound is 25%.
- With the scale ratio detached, `s <= 1`: for nonnegative features, each GeM-3 channel is at least its arithmetic mean. For an 8x8 map, a GeM-3 derivative is at most `1/4` at any location. The candidate's per-location pooling derivative is therefore at most `(1-alpha)/64 + alpha/4`, or **2.875 times** the accepted GAP derivative at the cap and about **1.234 times** at initialization. Every spatial location still receives the dense average contribution.

These are structural bounds, not claims inferred from a short probe. The probe below additionally tests optimizer continuity on the real production distribution.

## Semantic and scale preflight

Run disposable diagnostics only; no hook, alternate forward, or safety branch may remain in tracked production code.

1. Construct accepted and candidate models from reset seed 42. Require all accepted parameter/buffer tensors to be bitwise identical, unchanged accepted parameter order, identical post-construction CPU/CUDA RNG states, and only the appended scalar in candidate state. Confirm the scalar gives `alpha=1/64` within `1e-7` and total parameters equal 1,073,963.
2. On synthetic nonnegative `[128,128,8,8]` tensors, compare the module against a direct FP32 reference. Require finite outputs; `0 < alpha <= 1/8`; `s <= 1+1e-6`; `||q_norm||/||a||` within `[0.99999,1.00001]` whenever `||a||>1e-5`; and `||pooled|| <= ||a||*(1+1e-6)`.
3. Use constant, all-zero, one-hot-spike, tied-peak, and random feature maps. Require constant maps to agree with accepted GAP within FP32 tolerance, zero maps to remain zero, no RNG-state change, and autograd gradients to match a direct reference. Require the maximum absolute local candidate/GAP pooling-gradient ratio to be at most 1.24 at initialization and 2.88 when the scalar is forced near its cap.
4. Materialize one immutable production-distribution corpus of 200 post-transform batches, with 100 hard-target and 100 alpha-1 CutMix-target batches in the intended strong/weak proportions; record its SHA-256. Both arms must consume byte-identical tensors in the same order from independently restored accepted weights and SGD state.
5. Before updating, record pooled-feature norms, feature displacement, logit RMS, loss, prediction histogram, classifier-gradient norm, and backbone-gradient norm. Require every value finite, no candidate-only class concentration above 95%, candidate loss no more than `1.25x` control, classifier/backbone gradient ratios no more than `1.5x`, and
   `||pooled-a|| / max(||a||,eps) <= 0.03126` for every example.
6. From independently restored state on every corpus batch, perform one normal `lr=0.1` SGD update and reevaluate the same batch. Require finite parameters, buffers, momentum, logits, and loss; candidate/control RMS logit-displacement ratio at most `1.5`; candidate post/pre loss ratio at most `2.0`; no candidate-only >95% class concentration; and post-step `alpha <= 1/32`.
7. Continue aligned control/candidate trajectories for all 200 corpus steps. Require no nonfinite state or candidate-only concentration, candidate terminal loss EMA at most `1.25x` control, and `alpha <= 1/16`. Record alpha trajectory rather than interpreting lower loss as a safety override.

Any semantic, bound, RNG, target, gradient, displacement, concentration, or trajectory failure retires this exact candidate. Repair implementation defects only; do not lower the initialization, cap, exponent, scalar LR, or weight decay as a rescue.

## Timing and resource gates

On exactly one idle 97,871-MiB H20, run five alternating fresh-process control/candidate pairs after conditioning. Restore identical accepted tensors/optimizer state and use persisted byte-identical hard and CutMix batches. Each arm gets 100 warmups and at least 1,000 synchronized full training steps including transfer, forward, loss, backward, SGD, and synchronization. Measure strong and weak target paths separately and combine their means 80/20.

Proceed only if all hold:

- weighted candidate/control mean step time `<=1.03`, every pair `<=1.045`, and per-arm trial-mean CV `<=2%`;
- candidate p95 step time `<=1.06x` the control mean;
- projected exposure `floor(26,898 * control_mean / candidate_mean) >=26,242` (97% retention);
- candidate peak allocation `<650 MiB` and no more than 32 MiB over its paired control;
- finite losses/gradients and a valid evolving scalar in every trial.

Benchmark evaluation separately with five paired runs, 100 warmups, and at least 500 inference forwards. Require candidate/control mean `<=1.10`, CV `<=2%`, finite evaluator logits, and a conservative total projection below 540 seconds after charging all 19 accepted evaluation opportunities. Count the cube, mean, root, two norms, scale, interpolation, scalar sigmoid, and their backward costs; no fusion, channels-last, autocast, or algebraic rewrite may be introduced as a fallback.

## Production verification

If and only if every preflight and timing gate passes, run the exact candidate once at seed 42 with output redirected only to `run.log`. At each existing evaluation and in the final summary, log `pool_alpha`; this is provenance, not an adaptive trigger, and adds no evaluation.

Require exit zero, all standard finite summary fields, 300.0 counted training seconds, total below 600 seconds, at least 26,242 updates, 1,073,963 parameters, one 80% augmentation/CutMix switch, eight stopped workers, hard weak-tail targets, approximately 50% strong-phase CutMix, and no duplicate evaluation epoch. Compare the actual number of looks with EXP-010's 19 because a max-over-checkpoints metric with fewer looks has a small downward bias.

Record switch accuracy against EXP-010's 89.73%, first-weak accuracy against 93.16%, best/final accuracy, final NLL against 0.1934, update count, runtime, memory, and alpha trajectory. A switch below 87.08% diagnoses strong-phase underfit but cannot stop, tune, or rerun the candidate.

The decision rule is fixed:

- `best_test_acc >=94.25%` with all integrity/resource gates is improvement;
- a valid lower result is no-improvement, with no alpha/exponent/cap rescue;
- an accuracy pass with fewer than 26,242 updates is timing-confounded and cannot be called an efficient pooling win;
- a crash or protocol failure may be repaired only when an independent implementation/environment defect is shown and the exact operator remains unchanged.

## Risks and falsification

- **CutMix mismatch:** even bounded GeM can overweight a compact donor or augmentation artifact relative to its area-proportional label.
- **Norm-direction change:** equal feature norms do not imply equal logits; changing channel direction can alter classifier margins, so logit and update gates remain necessary.
- **Detached-scale semantics:** the forward normalization is input-adaptive while its derivative is intentionally frozen; this is a designed straight-through-like asymmetry and may harm representation learning.
- **Scalar saturation:** the global learning rate or momentum can drive the logit quickly toward a bound. The output remains structurally safe, but early saturation falsifies the intended gradual-adaptation mechanism under the registered preflight limits.
- **Low effect ceiling:** an alpha capped at 1/8 may be too small to move ten CIFAR-10 examples; that is preferable to repeating EXP-014's uncontrolled branch.
- **Endpoint kernel cost:** several small reductions and pointwise kernels can be launch-bound despite negligible FLOPs.
- **Indirect evidence:** GeM's direct evidence is retrieval, and the AISTATS learned-mixture result is not this normalized global classifier endpoint. The local accuracy prior remains weak.
- **Single-seed resolution:** 0.10 point is ten test examples. A bare pass is protocol-valid but weak causal evidence and cannot be confirmed by rerolling.

## Evidence

- Goal and protocol: `goals/maximize-cifar10-best-test-accuracy/01-definition.md`.
- Bottleneck and open pooling question: `goals/maximize-cifar10-best-test-accuracy/02-system-understanding.md`.
- Accepted width-2/N1-M7/CutMix result: `experiments/010/04-analysis.md`.
- Raw-max collapse and measured 4.10x first-gradient mechanism: `experiments/014/04-analysis.md`.
- Fixed GeM-3 design and its adversarial rejection: `experiments/023/proposals/idea-02.md` and `experiments/023/01-idea-review.md`.
- Lee, Gallagher, and Tu, *Generalizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated, and Tree*, AISTATS 2016: https://proceedings.mlr.press/v51/lee16a.html.
- Radenovic, Tolias, and Chum, *Fine-tuning CNN Image Retrieval with No Human Annotation*, TPAMI 2019 / arXiv 1711.02512: https://arxiv.org/abs/1711.02512.
- Accepted implementation endpoint and initialization order: `train.py`.
