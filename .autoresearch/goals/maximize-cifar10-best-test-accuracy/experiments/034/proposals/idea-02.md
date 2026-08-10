# Proposal: Conv2d-Only Kaiming `fan_out` Initialization

## Decision and falsifiable hypothesis

Initialize every `nn.Conv2d.weight` with Kaiming normal `mode="fan_out", nonlinearity="relu"`, the convention used by the installed torchvision ResNet. Preserve the accepted literal `init.kaiming_normal_(m.weight)` call for `nn.Linear`, and preserve all BN defaults, biases, graph structure, optimizer, all-parameter `1e-4` decay, N1/M7 plus p=0.5 alpha-1 CutMix strong phase, weak hard-label tail, elapsed-time LR schedule, evaluator, timer, seed, precision, batch size, and worker lifecycle.

**Falsifiable hypothesis:** backward-variance-oriented scaling in the stem and the two widening convolutions will improve early gradient transport and strong-phase representation learning without recurring runtime cost, raising seed-42 `best_test_acc` from the current 94.15% moving baseline to at least **94.25%**, with switch accuracy near or above the accepted 89.73%, final NLL no worse than 0.1934, and at least 99% of accepted optimizer-step exposure. The plausible accuracy band is deliberately wide (roughly 94.00-94.35%) because BN damps the forward effect and a safe miss is likely. A valid result below the moving-baseline-plus-0.10 threshold falsifies this exact all-Conv fan-out point. A safety veto invalidates it before production; neither outcome authorizes an in-experiment subset or scale rescue.

## Evidence and trajectory-aware rationale

The installed PyTorch 2.9.1 implementation documents that `kaiming_normal_` defaults to `fan_in`, which preserves forward variance, while `fan_out` preserves backward variance. The installed torchvision ResNet applies `kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")` to Conv2d modules. This is implementation precedent rather than direct evidence of a gain for this short CIFAR ResNet-20, CutMix/RandAugment curriculum, and fixed 300-second horizon.

The accepted EXP010 model uses default fan-in initialization for both Conv and Linear. The proposal preserves every accepted runtime/data choice and changes no kernel after construction, so it attacks representation quality orthogonally to the measured 75.46%-of-step backward bottleneck without spending exposure. It is nevertheless high risk:

- EXP012 and EXP015 show compute-neutral residual/initialization changes can pass short checks yet suppress the full strong phase.
- EXP024 shows an early candidate-only class transient can follow a transition representation change, although it also changed widths, shapes, shortcut ratios, and RNG consumption.
- EXP025 and EXP031 show that exact initial function or bounded initialization amplitude does not ensure bounded multi-step recruitment.
- Most recently, EXP033 achieved its intended sparse 1.48% mean-fill dose and lower terminal loss yet still produced candidate-only concentration and 8.93x logit geometry. Therefore lower loss, a small intervention, or an approximately preserved initial function cannot waive class/output/update gates.

Unlike EXP033, this candidate and control consume byte-identical stored inputs and targets, so paired divergence is attributable to the initialization reparameterization rather than distinct augmented images. The immutable-corpus screen is still representative rather than proof of the future scored trajectory.

## Exact production implementation

Replace only the initializer type split:

```python
@staticmethod
def _weights_init(m):
    # Kaiming init normal instead of default uniform per "Delving Deep into Rectifiers" (He et al. 2015), cited as
    # [13] in the ResNet paper
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
```

Do not change the accepted Linear call, `fc.bias`, Conv bias settings, BN affine or running-state initialization, residual ordering, zero-gamma state, optimizer groups, LR/decay, or call initialization again after `.to(device)`. Do not add production diagnostics or timed-path conditionals. The tracked diff must be confined to this initializer branch.

## Exact tensors, scales, and RNG equivalence

For a 3x3 Conv with shape `[C_out,C_in,3,3]`, accepted `fan_in` uses `sigma_in=sqrt(2/(9*C_in))`, candidate `fan_out` uses `sigma_out=sqrt(2/(9*C_out))`, and their ratio is `c=sqrt(C_in/C_out)`. Of 19 convolutions, 16 have equal input/output width and remain bitwise identical. Only three tensors change:

| Layer | Shape | Weights | Accepted std | Candidate std | Candidate/control scale |
|---|---:|---:|---:|---:|---:|
| `conv1` | `[32,3,3,3]` | 864 | 0.272166 | 0.083333 | 0.306186 |
| `layer2.0.conv1` | `[64,32,3,3]` | 18,432 | 0.083333 | 0.058926 | 0.707107 |
| `layer3.0.conv1` | `[128,64,3,3]` | 73,728 | 0.058926 | 0.041667 | 0.707107 |

Thus 93,024 of 1,069,920 Conv weights change, 8.662% of all 1,073,962 parameters. Their aggregate expected squared Frobenius norm falls from 448 under fan-in to 198 under fan-out, a 0.44196 energy ratio and 0.66480 norm ratio. Same-width Conv weights, the 128x10 Linear weight, its bias, all BN tensors, graph, shapes, and parameter count remain exact.

Both calls use one `tensor.normal_` with the same tensor shape and traversal order. In the installed environment, independent seed-42 probes confirm identical post-call CPU RNG state; equal-fan tensors are bitwise identical, and changed tensors equal accepted samples times `c` within `2.98e-8` maximum absolute error. The execution controller must independently establish the same property at full-model scope rather than trusting this exploratory probe:

1. Reset CPU and CUDA RNG to registered states, instantiate the accepted model, and capture its complete state plus post-construction RNG states.
2. Reset to the identical states, instantiate the candidate, and capture the same evidence.
3. Require identical module traversal, 19 Conv2d/19 BN/one Linear, 1,073,962 parameters, bitwise identity for every unaffected parameter/buffer, and exact matching signs/draw directions for each changed Conv with `atol=1e-7, rtol=1e-6` against the analytic rescaling.
4. Require bitwise-identical CPU and CUDA RNG states after construction and no inspection-induced mutation. Serialize and fsync the report before assertions.

## BatchNorm invariance is approximate, not protective

Each changed Conv is immediately followed by BatchNorm. For positive `c`, train-mode normalization approximately cancels the scale:

```text
BN(c*z) = (z-mean(z)) / sqrt(var(z) + eps/c^2).
```

With zero epsilon and exact arithmetic, the initial normalized function would match. In this model it does not remain equivalent: effective BN epsilon rises 10.667x in the stem and 2x at each widening Conv; FP32 reductions differ; running variances accumulate at different scales; coupled decay acts on smaller weights; and SGD momentum integrates larger raw gradients. For a scale-normalized Conv, raw data-gradient norm is expected to grow roughly `1/c` (3.266x stem, 1.414x transitions), while the relative data update can grow roughly `1/c^2` (10.667x stem, 2x transitions). That optimizer reparameterization is the intended mechanism and the dominant safety risk.

No LR, warmup, clipping, gradient scale, layer-specific decay, decoupled decay, or BN-epsilon compensation is allowed. Such compensation would test a different idea and would destroy the clean torchvision-style fan-out comparison.

## Static and initial-function gates

Before any trajectory run, use production-distribution tensors and hooks around the three changed Conv-BN pairs. Require finite tensors/loss, pre-BN RMS ratios within 2% of the registered `c`, post-BN activation RMS ratios in `[0.98,1.02]`, candidate/control initial train-logit RMS difference divided by control logit RMS at most 0.02, and loss ratio in `[0.95,1.05]`. Record rather than gate initial eval-mode differences because unit running variance intentionally prevents exact eval cancellation. Require positive running variances and exact expected BN counters after the probe.

This gate verifies the intended local reparameterization; it does not claim that matching initial logits predicts accuracy or multi-step safety.

## Immutable-corpus trajectory safety

Reuse, never regenerate or filter:

- EXP022 200-batch accepted strong post-policy corpus: `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/022/preflight-corpus.pt`, SHA-256 `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946`.
- EXP028 64-batch accepted weak hard-label corpus: `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/028/weak-corpus.pt`, SHA-256 `ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032`.

Recompute file and tensor digests; validate batch counts, shapes, target ranks, hard/CutMix coverage, and finite tensors. Train independent accepted and candidate models over byte-identical copies of all 200 strong batches at LR 0.1, then all 64 weak batches at the registered weak LR, with deterministic backends, identical ordinary momentum SGD and coupled decay, and no evaluator. Hash corpora and sources before and after. Record per step: loss and EMA, class histogram, logit RMS, whole-model raw-gradient/update/parameter/momentum norms, candidate/control ratios, BN buffers/counters, and for each changed Conv its raw-gradient norm, update norm, parameter norm, relative update, pre/post-BN RMS, and running variance.

Production is authorized only if all of these pass:

- all parameters, logits, losses, gradients, updates, momentum, BN buffers, and diagnostics are finite; optimizer membership/state and BN counters are complete and exact;
- zero steps with candidate maximum predicted-class share above 95% while control is at or below 95%; as EXP033 demonstrated, lower loss cannot override this veto;
- candidate/control whole-model update-norm ratio never exceeds 5, candidate whole-model update never exceeds 25% of pre-update whole-model parameter norm, and no candidate update exceeds 5x its preceding 16-step median;
- no changed Conv receives a single update above 50% of its pre-update tensor norm; report first-step and trajectory quantiles against the anticipated 3.266/1.414 raw-gradient and 10.667/2.0 relative-update multipliers instead of pretending those intentional ratios should be near one;
- candidate/control logit RMS, raw-gradient norm, and update norm remain below 5x at every step, with no silent or dominant changed path and positive, finite BN variances;
- terminal debiased candidate/control loss-EMA ratio is at most 1.5 in both strong and weak phases;
- stored corpus/source/RNG declarations and all immutable inputs remain unchanged.

These are catastrophic-geometry bounds, not accuracy proxies. They deliberately leave room for the candidate's expected larger relative stem step while vetoing the class/output/update pathologies repeatedly observed in EXP020/022/024/028/031/033. Any veto retires this exact all-Conv fan-out point before timing or production. Do not rerun the controller to seek a favorable path.

## Runtime and production integrity

Initialization completes before `t_start_training` and leaves the trained graph, kernels, shapes, memory traffic, batch construction, and per-step operations unchanged. No paired timing campaign is needed; its result would measure run noise rather than recurring candidate work. Exposure remains a post-run integrity check: expect approximately 26.9k steps and 598.7 MiB, require at least 99% of the accepted 26,898 steps absent documented system-wide contention, and never discard a lower-accuracy valid run merely because exposure differs within that bound.

Immediately before the single scored run:

- query `exp-index.sh baseline` rather than hardcode a stale comparator; expected current value is 94.15 at `7c1e7d8`, so the current gate is 94.25;
- require the experiment branch to descend from the integration baseline; only tracked `train.py` may differ, and its diff must be the initializer split above;
- require no stale `run.log`, passing syntax/Ruff/format/scope checks, and exactly one idle H20 with approximately 97,871 MiB visible;
- verify seed 42, parameter count, data transforms, CutMix collator, loaders, schedule, timer, evaluator, and summary code are unchanged.

Run exactly once with output redirected:

```bash
timeout --kill-after=5s 595s uv run train.py > run.log 2>&1
```

Require exit zero; exactly one finite ten-field summary; 300.0-301.0 counted seconds; total under 600 seconds; 1,073,962 parameters; one transition near 80% with all eight strong workers stopped; hard weak targets; first weak LR near 0.01; 45-55% CutMix among strong batches; no duplicate evaluation epoch; at most one evaluation per epoch; and 18-19 total evaluations, never above the accepted 19-look ceiling. Record source/log hashes and all early, switch, first-weak, best, and final metrics. Do not rerun a valid completion.

## Expected signatures, risks, and formal verdict

A supportive trajectory preserves healthy multiclass predictions, keeps switch fit near or above 89.73%, recovers first-weak accuracy near or above 93.16%, ends with NLL at or below 0.1934, has little best-final regression, and reaches the moving-baseline-plus-0.10 gate. These diagnostics explain the result but never alter the formal top-1 verdict.

- **Effective-step risk — high:** the stem starts at 30.6% of accepted norm and can receive roughly 10.7x larger relative data updates despite nearly invariant train features.
- **Long-horizon fit risk — medium-high:** previous compute-neutral initialization/residual ideas passed short probes but suppressed protected strong-phase fit.
- **Trajectory risk — medium-high:** EXP033 reinforces that small, lower-loss perturbations can still trigger early class/logit geometry failures.
- **Impact risk — medium-high:** only three tensors differ and BN may erase much of the representational effect, leaving less than 0.10 point.
- **Single-seed risk — medium:** a narrow formal improvement remains weak evidence about run-to-run variability, but seed rerolls are forbidden.
- **Runtime risk — low:** no recurring production operation changes; correctness depends on exact Linear/RNG/source preservation.

Formal outcomes:

- **Improvement:** every integrity condition passes and `best_test_acc >= current_moving_baseline + 0.10` (currently 94.25%).
- **No improvement:** the production run is valid but misses that threshold, regardless of favorable loss, NLL, exposure, or intermediate accuracy.
- **Invalid/crash:** initialization/RNG/Linear mismatch, immutable-corpus safety veto, tracked-scope/hardware/data/evaluator/timer/evaluation-count violation, malformed/nonfinite summary, nonzero exit, or total runtime at least 600 seconds.

No rescue is allowed: do not exclude the stem, select only widening layers, interpolate fan scales, apply fan-out to Linear, change BN epsilon, compensate LR/decay, clip, warm up, reroll seed/corpus, add evaluations, relax gates, or run production after a veto. Each is a new experiment with a new hypothesis and ID.
