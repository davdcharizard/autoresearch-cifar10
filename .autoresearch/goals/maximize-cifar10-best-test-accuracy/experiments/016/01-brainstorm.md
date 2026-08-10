# Brainstorm EXP-016
**Created**: 2026-08-06

## Web Search & Literature Review

- **Making Convolutional Networks Shift-Invariant Again** (`papers/blurpool.md`; https://proceedings.mlr.press/v97/zhang19a.html): filter before subsampling and transform both strided paths consistently; correct integration can improve classification accuracy.
- **Balanced Mixture of Supernets for Learning the CNN Pooling Architecture** (`papers/resnet20-downsampling-search.md`; https://proceedings.mlr.press/v224/roshtkhari23a.html): direct CIFAR-10 ResNet20 evidence that downsampling placement materially affects accuracy.
- **Amortized Nesterov's Momentum** (https://proceedings.mlr.press/v124/zhou20a.html): stochastic Nesterov trades acceleration against noise robustness; it supports optimizer plausibility but not this exact CIFAR operating point.
- **PyTorch AMP guidance** (official docs referenced in EXP-015): BF16 autocasts eligible convolution/linear work while retaining FP32 master parameters and FP32 evaluation; local H20 kernels/numerics remain decisive.

## Experimental History Review

- EXP-010 remains 94.15% with healthy 89.73% strong fit, 93.16% first weak, and 26,898 steps. Preserve its postactivation graph, p=0.5 CutMix pressure, all-parameter `1e-4`, and 80% transition unless a candidate explicitly funds a different capacity point.
- Decay variants, stronger CutMix, full preactivation, and selective zero-gamma failed. EXP-012/015 establish a recurring rule: identity-oriented residual changes lower strong fit by 2.85-3.25 points even at equal exposure.
- EXP-014 rejects unbounded raw-max readout. Initial functional equality and short-fit wins are not evidence of phase-scale representation quality.
- EXP-013 rejects batch256 at only 1.189x image throughput. The measured systems opportunity remains the 97.57% model forward/backward path, not loader or optimizer overhead.
- Nesterov was bundled with EXP-001's failed 15% LR hold and never isolated on the accepted 80% schedule. Downsampling/anti-aliasing and BF16-funded capacity remain untested.

## Collected Ideas

- **Full-path BlurPool transitions** — turn each stride-2 residual convolution into stride-one convolution followed by a fixed low-pass/downsample, and apply the same blur/subsample to the Option-A shortcut before padding. Targets aliasing and transition information without suppressing branch activity; sequential kernels may cost too many updates.
- **Isolated Nesterov SGD** — set only `nesterov=True` on the accepted optimizer, retaining momentum, LR hold/tail, decay, data, and graph. It tests the optimizer component confounded by EXP-001 at near-zero cost, but its ceiling may be below 0.10.
- **BF16-funded width 3** — use BF16 autocast to pay for a width-3 postactivation model while keeping FP32 master state/evaluation. Width previously delivered the largest architecture gain; the combination has high upside but two mechanisms and a demanding timing/numeric gate.
- **Channels-last width 2** — preserve FP32 semantics but seek faster convolution kernels via memory format, converting model and batches once. It may add exposure without precision drift, though 32x32 kernels can be slower and exposure alone lacks an accuracy mechanism.
- **Anti-aliased shortcut only** — replace Option-A slicing by fixed 2x2 average/blur downsampling while leaving residual stride-2 convolution accepted. It is cheaper than full BlurPool but violates consistent path filtering and may create phase misalignment.
- **Small positive residual gamma** — initialize six ordinary final BN scales to 0.1 instead of zero, retaining first-step residual gradients. EXP-015 suggests this could preserve activity, but it remains in a recurring failed identity-initialization family and is deprioritized.
- **Moonshot: BF16 width 3 plus full-path blur** — combine more capacity with shift-stable transitions while BF16 pays part of both costs. The ceiling is high, but three coupled mechanisms make preflight/attribution too complex for a first test.

## Combinations

- **BF16 + width 3**: lower-precision convolution can fund a capacity increase whose local width-2 predecessor gained 1.25 points; unlike BF16 alone, it supplies a credible accuracy mechanism, but must retain a predeclared exposure floor.
- **BlurPool + channels-last**: layout speed could offset blur kernels while preserving FP32, but the combination should follow evidence that each installed kernel path works rather than rescue a failed timing gate.
- **Nesterov + EMA**: accelerated online updates plus late smoothing might balance noise, but EXP-015's review found little tail variance and the combination weakens attribution.

## Candidate Ideas

### Isolated PyTorch Nesterov Momentum

**Summary**: Add only `nesterov=True` to the accepted SGD optimizer, retaining momentum `0.9`, coupled decay `1e-4`, the long `lr=0.1` hold, the `0.01` transition/cosine tail, and every other accepted model/data/evaluation choice. See `proposals/idea-02.md`.

**Reasoning**: EXP-001 bundled Nesterov with an early-decay schedule, so the accepted lineage has never isolated the optimizer flag. This is the cleanest unresolved ablation and should retain at least 99% of EXP-010's 26,898 updates. It could improve transient response and basin selection around the long noisy strong phase and abrupt weak-tail LR drop. However, it adds no capacity or data signal, its likely upside is close to the 0.10-point threshold, and installed PyTorch semantics make the first parameter update exactly `1.9x` ordinary momentum while also changing the effective coupled-decay trajectory. A real-batch stability gate must rule out high-LR overshoot without tuning or rescue.

**Sources**: https://proceedings.mlr.press/v124/zhou20a.html; installed PyTorch 2.9.1 SGD semantics; EXP-001/002/010/015 analyses; `proposals/idea-02.md`.

**Estimated Effort**: Low implementation effort, medium verification effort because exact recurrence, RNG/state identity, production-batch safety, and paired timing must be proven.

**Risk Assessment**: Medium. Attribution is excellent and compute risk is minimal, but the accuracy mechanism is weak and the first-step/high-LR dynamics may be worse than the already-validated standard momentum recipe.

### BF16-Funded Width-3 Postactivation ResNet-20

**Summary**: Increase the accepted width multiplier from 2 to 3 and run only training forward/loss under CUDA BF16 autocast, retaining FP32 master parameters, gradients, optimizer state, BatchNorm persistent state, and evaluation. This intentionally tests one resource exchange: BF16 must fund enough width-3 exposure for increased capacity to improve accuracy. See `proposals/idea-03.md`.

**Reasoning**: Width 2 delivered the largest local architecture gain, improving 1.25 points despite only 70.76% of width-1 exposure, while the measured forward/backward path consumes 97.57% of GPU-stage time. Width 3 preserves the successful postactivation graph and active residual branches but raises parameters from 1,073,962 to 2,412,730. BF16 is relevant only if measured H20 kernels accelerate this larger graph: a three-arm width2-FP32/width3-FP32/width3-BF16 gate requires at least `1.15x` BF16 funding versus width3 FP32 and at least 22,863 projected steps, plus strict paired numerical and loader/wall-time checks. The result validates the combined operating point, not either component independently.

**Sources**: local EXP-007/010 results and system profile; official PyTorch 2.9 AMP documentation; `proposals/idea-03.md`.

**Estimated Effort**: High. The code diff is small, but dtype/state assertions, a 200-step paired numerical probe, balanced fresh-process three-arm timing, loader checks, and wall projection are substantial.

**Risk Assessment**: High. It has the highest plausible accuracy upside, but tiny CIFAR kernels may not receive enough BF16 speedup, width 3 may be underexposed or saturated, and BF16 plus width couples two mechanisms. All funding and numerical gates are hard no-go conditions with no fallback configuration.

### Full-Path Anti-Aliased Transition Blocks

**Summary**: Replace each learned stride-2 transition convolution by a stride-1 convolution followed by a fixed depthwise stride-2 `3x3` binomial blur, and apply the same blur and sampling phase to the Option-A shortcut before its existing zero pad. This preserves the width-2 postactivation topology, parameter count, initialization, optimizer, data recipe, and active residual branches while testing one coherent full-path anti-aliasing mechanism. See `proposals/idea-01.md`.

**Reasoning**: The accepted model performs unfiltered decimation in both transition paths. ICML 2019 evidence supports low-pass filtering immediately before subsampling, and direct CIFAR-10 ResNet20 evidence shows downsampling operator/placement is accuracy-relevant. Unlike the failed preactivation and zero-gamma experiments, this does not initialize residual branches toward identity. The cost is material: dense transition convolutions plus depthwise blur may reduce fixed-time exposure, and smoothing may erase CIFAR detail or dilute CutMix boundaries. Full execution is conditional on exact impulse/ramp alignment tests, production-distribution stability, and retaining at least 24,016 projected steps.

**Sources**: `papers/blurpool.md`; `papers/resnet20-downsampling-search.md`; https://proceedings.mlr.press/v97/zhang19a.html; https://proceedings.mlr.press/v224/roshtkhari23a.html; `proposals/idea-01.md`.

**Estimated Effort**: Medium. A small module and two transition-path changes, but exact spatial-phase and paired timing gates are required.

**Risk Assessment**: Medium-high. The mechanism is externally supported and architecturally coherent, but can hurt fine-detail learning, shortcut transport, or exposure; the source papers do not establish this exact kernel under the accepted CutMix/RandAugment recipe.

## Review

The mandatory external Claude review completed successfully; no fallback reviewer was used. It selected BF16-funded width 3 because width is the only locally demonstrated multi-point architecture lever and the candidate has the only ceiling plausibly above the ten-image acceptance margin. The review's decisive correction is adopted: the width-3 FP32 timing arm is the actual default PyTorch path, its TF32 flags must be recorded without changing them, and BF16 must still satisfy the existing funding and exposure gates against that real control. A sub-15% BF16 advantage is an expected preflight no-go, not permission to disable TF32, relax the gate, or try another width/precision. The proposal already addresses the other material concerns - width saturation, capacity starvation, coupled-mechanism attribution, numeric drift, and wall time - through conjunctive gates and a one-run rule.

The review lowered the prior for BlurPool because padded random cropping already supplies translation pressure and the cited ResNet20 work does not validate this exact kernel. It judged Nesterov a clean unresolved ablation but too close to the single-seed noise floor. Full review: `01-idea-review.md`.

## Idea Evaluation

- **BF16-funded width 3**: strongest lead. Evidence is incomplete for width 2 to 3 and default TF32 may make it infeasible, but these uncertainties are directly testable before the production run. Its capacity mechanism and potential impact dominate the alternatives if all hard gates pass.
- **Full-path BlurPool**: coherent and literature-grounded, but its expected effect is modest under existing crop augmentation and it risks both lower exposure and the recurring strong-underfit signature through detail suppression.
- **Isolated Nesterov**: best attribution and cheapest execution, but no local evidence supports an effect comfortably above 0.10 points; the first-step `1.9x` update also creates a real high-LR risk without supplying a higher ceiling.

## Chosen Idea
**Selected**: BF16-Funded Width-3 Postactivation ResNet-20

**Why this idea**:
Width is the only locally validated multi-point lever, and BF16 targets the measured forward/backward-dominated cost that otherwise makes width 3 impractical under the fixed timer. The choice is conditional rather than optimistic: the exact width3-BF16 operating point proceeds only if it is numerically faithful and retains at least 22,863 projected updates while beating the actual default width3-FP32/TF32 path by the declared margin. No alternative precision, width, or fallback experiment is allowed inside EXP-016.

**Hypothesis**:
On the H20, BF16 autocast will accelerate width-3 training by at least `1.15x` versus the actual default width3-FP32 path, keep width3-BF16 no slower than `1.17647x` the accepted width2-FP32 step, and preserve at least 22,863 updates. With finite, closely aligned paired numerics and the unchanged EXP-010 recipe, the added postactivation capacity will retain healthy strong-phase fit and raise FP32-evaluated `best_test_acc` from 94.15% to at least 94.25%.
