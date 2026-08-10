# Proposal: Conv2d-Only Kaiming `fan_out` Initialization

## Decision and falsifiable hypothesis

Initialize every `nn.Conv2d.weight` with Kaiming normal `mode="fan_out", nonlinearity="relu"`, matching the official torchvision ResNet convention. Keep the `nn.Linear` initialization call exactly as accepted—default Kaiming normal/fan-in—and leave BN defaults, biases, model graph, optimizer, data curriculum, schedule, evaluator, timer, precision, and seed unchanged.

**Hypothesis:** backward-variance-preserving initialization in the stem and two widening convolutions will improve early gradient transport and strong-phase representation formation without recurring runtime cost, raising seed-42 `best_test_acc` from 94.15% to at least **94.25%** while retaining approximately 26.9k steps. Point prediction: **94.28%**, switch fit near the accepted 89.73%, and final NLL no worse than 0.1934. A valid lower result falsifies this exact Conv-only fan-out point; it does not authorize post-result layer selection or blended scales.

## Evidence and local scope

Official PyTorch initialization documentation states that `kaiming_normal_` defaults to `mode="fan_in"`: fan-in preserves forward activation variance, while fan-out preserves backward gradient variance. The official torchvision ResNet implementation explicitly uses `kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")` for Conv2d and BN scale/bias 1/0. This is strong implementation precedent, not direct accuracy evidence: torchvision targets ImageNet-style ResNets, while the local network is a short CIFAR ResNet-20 trained with CutMix/RandAugment for 300 counted seconds.

Local history makes the proposal both plausible and risky. EXP010's accepted model uses default fan-in initialization for Conv and Linear. EXP012/015 show that compute-neutral residual/initialization changes can pass structural and short-fit checks yet suppress the long strong phase. EXP024 changed transition width and immediately produced class concentration, but also changed parameter shapes, shortcut ratios, classifier width, and RNG consumption; EXP033 changes none of those. EXP025 and EXP031 show that initial function continuity or an initialization-scale bound does not ensure safe multi-step geometry. Therefore this narrower, fixed initialization requires exact-corpus trajectory gates even though it adds no parameter or per-step operator.

## Exact implementation and RNG preservation

Replace the current shared initializer:

```python
@staticmethod
def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight)
```

with exactly:

```python
@staticmethod
def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(
            m.weight, mode="fan_out", nonlinearity="relu"
        )
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
```

The Linear branch deliberately retains the literal accepted call, including its defaults. Do not change `fc.bias`, Conv bias settings, BN affine/running-state initialization, add zero-gamma, reinitialize after `.to(device)`, or use a model-wide explicit fan-out call that would include Linear. Each module receives the same number and shape of normal draws in the same `self.apply` traversal order, so seed-42 RNG advancement after construction should remain exact. The final Linear weight and bias, all BN tensors, and every unaffected Conv's sampled values must be bitwise accepted; the changed Conv tensors must share the same underlying standard-normal directions with only the derived scale ratios below.

## Exactly which layers and scales change

For a 3x3 Conv `[C_out,C_in,3,3]`, accepted fan-in standard deviation is

```text
sigma_in  = sqrt(2 / (9*C_in))
```

and candidate fan-out standard deviation is

```text
sigma_out = sqrt(2 / (9*C_out)).
```

Their ratio is `c = sigma_out/sigma_in = sqrt(C_in/C_out)`. Of the model's 19 convolutions, **16 are same-width and remain numerically identical** because `C_in=C_out`. Only these three tensors change:

| Layer | Shape | Weights | Accepted std | Candidate std | Candidate/control scale |
|---|---:|---:|---:|---:|---:|
| stem `conv1` | `[32,3,3,3]` | 864 | `sqrt(2/27)=0.272166` | `sqrt(2/288)=0.083333` | `sqrt(3/32)=0.306186` |
| `layer2.0.conv1` | `[64,32,3,3]` | 18,432 | `sqrt(2/288)=0.083333` | `sqrt(2/576)=0.058926` | `1/sqrt(2)=0.707107` |
| `layer3.0.conv1` | `[128,64,3,3]` | 73,728 | `sqrt(2/576)=0.058926` | `sqrt(2/1152)=0.041667` | `1/sqrt(2)=0.707107` |

The candidate changes 93,024 of 1,069,920 Conv weights (**8.694%**) and 8.662% of all 1,073,962 parameters, solely by rescaling their initial samples. For a Kaiming-normal Conv, expected squared Frobenius norm is `2*C_out` under fan-in and `2*C_in` under fan-out. Across the three changed tensors this falls from `64+128+256=448` to `6+64+128=198`, an aggregate squared-norm ratio 0.44196 and norm ratio 0.66480. Same-width Conv weights, the 128x10 Linear weight, its 10-element bias, all 19 BN modules, parameter count, and tensor shapes are exact.

## What BatchNorm makes invariant—and what it does not

All three changed convolutions are immediately followed by BatchNorm before ReLU. In train mode, positive rescaling by `c` approximately cancels:

```text
BN(c*z) = (c*z-c*mean(z)) / sqrt(c^2*var(z)+eps)
        = (z-mean(z)) / sqrt(var(z)+eps/c^2).
```

Thus, if `eps` were zero and arithmetic exact, the normalized forward outputs—and consequently the initial train-mode model function—would be invariant. In reality:

- BN epsilon is effectively multiplied by `1/c^2`: 10.667x for the stem and 2x for each widening Conv, so invariance is approximate rather than exact.
- FP32 reduction/rounding changes when pre-BN values are rescaled.
- BN running variances learn values proportional to roughly `c^2` (0.09375 for the stem and 0.5 at each widening Conv before running-stat interpolation), so initial eval-mode outputs are not invariant while running variance still starts at one.
- The backward gradient of a scale-normalized Conv is approximately proportional to `1/c`. Candidate raw data-gradient norms may therefore be about 3.266x in the stem and 1.414x in each widening Conv.
- Relative data updates can scale approximately as `1/c^2`: about 10.667x for the stem and 2x at the widening Convs. Fixed LR 0.1 and momentum are not invariant to this reparameterization.

Those optimizer effects are the actual intervention. BN is expected to preserve initial feature values while fan-out changes parameter norm, gradient magnitude, momentum state, and normalized-function step size. The proposal must not claim that BN makes the experiment functionally identical beyond initialization.

## Coupled weight decay and optimizer implications

Keep installed ordinary PyTorch SGD exactly unchanged: LR 0.1 in the strong phase, momentum 0.9, and coupled all-parameter decay `1e-4`. For the same underlying sample direction, the absolute decay vector `lambda*w` is initially 0.306x accepted in the stem and 0.707x at each widening Conv. Across all three changed tensors, initial L2 penalty energy is about 44.2% of accepted and decay-vector norm about 66.5%.

Relative pure-decay shrinkage remains `lr*lambda`, but the data-gradient/decay balance does not: BN can amplify the data gradient as weight scale shrinks while the absolute decay term shrinks with the weight. Ordinary momentum then accumulates those larger raw gradients. This is why no layer-specific LR or decay compensation is allowed—the candidate tests torchvision-style fan-out under the accepted optimizer—but also why first/multi-step update gates are mandatory. Do not exclude these tensors from decay, scale their gradients, add warmup/clipping, or separately tune momentum.

## Static and exact-initialization gates

An ignored controller must first instantiate accepted and candidate models from independent resets to seed 42 on CPU, then serialize/fsync evidence before assertions. Require:

- exactly 19 Conv2d modules, three unequal-fan layers with the names/shapes/counts above, 16 equal-fan layers, one Linear, 19 BNs, 1,073,962 parameters, and unchanged graph/module ordering;
- every same-width Conv, Linear weight/bias, BN tensor, and non-parameter buffer bitwise equal between arms;
- each changed Conv equals its accepted tensor times the registered `c` within `1e-7` absolute / `1e-6` relative tolerance, with matching signs/standard-normal directions and empirical norm/std ratios near the analytic values;
- identical CPU and CUDA RNG states immediately after construction, proving the fan argument changed scale rather than draw order;
- finite state and no mutation from the inspection itself.

On exact production-distribution batches in train mode before any update, hook each changed Conv and following BN. Require pre-BN RMS ratios consistent with `c`, post-BN activation RMS ratios in `[0.98,1.02]`, finite logits/loss, initial candidate/control logit RMS-difference ratio at most 0.02, and loss ratio in `[0.95,1.05]`. Record eval-mode differences and running-variance behavior, but do not require eval-mode identity because the initial running statistics intentionally break scale cancellation.

## Immutable-corpus trajectory safety

Reuse the immutable EXP022 200-batch strong post-policy corpus, file SHA-256 `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946`, and EXP028 64-batch weak hard corpus, file SHA-256 `ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032`. Recompute file and tensor digests, validate schema/count/target rank, and never regenerate, reorder, filter, or mutate either corpus.

Train independent accepted and candidate models—initialized as proven above—on byte-identical copies of all 200 strong records at LR 0.1 and then all 64 weak records at the registered low LR. Use deterministic backend settings, identical ordinary SGD, and no evaluator. Record per step and named layer: loss, class histogram, raw gradient norm, data-plus-decay direction, momentum/update/parameter norms, relative update, Conv norm, pre/post-BN RMS, BN running mean/variance/counter, logits, target rank, and finite state.

Mandatory gates are:

- all logits, losses, gradients, parameters, momentum, BN state, and diagnostics finite, with exact BN counters and complete optimizer state;
- no step where candidate maximum predicted-class share exceeds 95% while control is at or below 95%; lower candidate loss cannot waive this gate;
- no candidate whole-model update above 25% of whole-model parameter norm, no candidate/control whole-model update ratio above 5, and no candidate update above 5 times its preceding 16-step median;
- for each changed Conv, no single update above 50% of that tensor's pre-update norm; separately report the first-step raw-gradient and relative-update ratios against the 3.266/1.414 and 10.667/2.0 scale expectations;
- candidate/control terminal debiased loss-EMA ratio at most 1.5 in both strong and weak phases;
- post-BN activations remain finite/nondegenerate, running variances positive, and neither the stem nor transition residual path becomes numerically dominant or silent;
- all source/corpus/RNG declarations remain unchanged after replay.

These are gross-safety gates, not accuracy proxies. EXP015 showed a favorable short loss trajectory can invert over the full strong phase. Passing them authorizes one production run but does not establish better initialization. Failure retires this exact all-Conv fan-out point; it does not authorize fan-out only on widening layers, excluding the stem, interpolating fan modes, changing BN epsilon, or rescaling LR.

## Runtime and production verification

Initialization happens before the training timer and leaves the graph, shapes, kernels, targets, parameter count, and per-step operations unchanged. A paired timing campaign would measure noise rather than candidate overhead and is not required. Expected exposure is the accepted 26,898 steps and 598.7 MiB, with at least 26,629 steps treated as a consistency expectation, not a way to discard an otherwise valid fixed-budget result.

After safety passes, reconfirm the 94.15 baseline at `7c1e7d8`; only `train.py` is tracked-modified; the source diff is limited to the initializer branch; no stale `run.log` exists; syntax/Ruff/format/scope checks pass; exactly one idle 97,871-MiB H20 is visible; and all non-initialization contracts are unchanged. Run seed 42 exactly once:

```bash
timeout --kill-after=5s 595s uv run train.py > run.log 2>&1
```

Require exit zero; one finite ten-field summary; approximately 300.0-301.0 counted seconds; total below 600 seconds; 1,073,962 parameters; one transition near 80% with eight workers stopped; 45-55% CutMix among eligible strong batches; hard weak targets; first weak LR about 0.01; no duplicate evaluation epoch and no more than one evaluation per epoch; normal VRAM/exposure; and no retry.

## Expected trajectory, risks, and verdict

The favorable mechanism is improved backward transport through the stem and stage expansions while BN keeps their initial normalized features close. Expected support is healthy early multiclass geometry, strong-phase fit near or above EXP010's 89.73% switch checkpoint, first-weak recovery near or above 93.16%, final NLL at or below 0.1934, little best-final regression, and `best_test_acc >=94.25%`. Report all early checkpoints, switch/first-weak/best/final accuracy, final NLL, train-loss EMA, CutMix counts, epochs/steps, evaluation count, VRAM/time, plus preflight scale/gradient/update/BN diagnostics.

Risks:

- **Effective-step risk — high:** the stem begins at 30.6% of accepted norm and may receive roughly 10.7x larger relative data updates despite nearly invariant train-mode features.
- **Strong-fit risk — medium-high:** local compute-neutral initialization/identity interventions have repeatedly suppressed the long composite phase after safe short probes.
- **BN-invariance risk — medium:** epsilon, running-stat lag, weight decay, SGD, and momentum break exact scale equivalence; official fan-out precedent does not remove those CIFAR-specific dynamics.
- **Impact risk — medium-high:** only three Conv tensors change and BN may rapidly erase much of the intended representational effect, leaving less than the 0.10-point gate.
- **Single-seed risk — medium:** a 94.25-94.35 result is formally positive but weak evidence relative to CUDA/trajectory noise.
- **Runtime/implementation risk — low:** no recurring operation or shape changes; scientific correctness depends on preserving Linear/RNG state exactly.

Formal verdict:

- **Improvement:** every protocol condition passes and `best_test_acc >=94.25%`.
- **No improvement:** the run is protocol-valid but `best_test_acc <94.25%`, regardless of NLL, exposure, or a favorable short trajectory.
- **Invalid/crash:** initialization/RNG/Linear mismatch, exact-corpus safety veto, scope/hardware/data/evaluator/timer violation, malformed summary, nonzero exit, or runtime at least 600 seconds.

Do not rerun a valid completion. Do not rescue with stem exclusion, transition-only fan-out, an interpolated scale, fan-out Linear, changed BN epsilon, layer-specific LR/decay, clipping, warmup, another seed, extra evaluations, or relaxed gates. Each variant is a new experiment with a new hypothesis and ID.
