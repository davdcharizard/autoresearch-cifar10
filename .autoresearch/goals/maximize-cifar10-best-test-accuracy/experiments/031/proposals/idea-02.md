# Proposal: Scale-Controlled 10% Max-Residual Global Pooling

## Decision and hypothesis

Replace only the accepted final global-average descriptor with a bounded residual mixture:

```python
avg = F.adaptive_avg_pool2d(out, 1).flatten(1)
max_ = F.adaptive_max_pool2d(out, 1).flatten(1)
pooled = avg + 0.10 * POOL_RESIDUAL_SCALE * (max_ - avg)
return self.fc(pooled)
```

Keep the accepted `Linear(128,10)` object, weights, bias, parameter count, initialization, and optimizer path. `POOL_RESIDUAL_SCALE` is one frozen scalar obtained by a preregistered training-only calibration below and then hardcoded before safety/timing/production. Because the scale is capped at one, the effective convex max coefficient cannot exceed 0.10; constant spatial maps remain exactly average pooled.

The hypothesis is that a small amount of peak-over-average evidence recovers localized class features that pure averaging dilutes, while average evidence retains at least 90% coefficient and continues to reflect CutMix region extent. The point prediction is 94.30%; formal success is `best_test_acc >=94.25%` versus EXP010's 94.15%, with at least 99% of accepted update exposure.

## Exact fixed scaling rule

Derive the scalar once in an ignored controller before tracked implementation. The rule, corpus, initialization, and cap are fixed before observing safety, timing, or accuracy:

1. Construct an accepted seed-42 width-2 ResNet-20 normally and put a disposable copy in evaluation mode. Do not alter the eventual production model or its RNG/state.
2. Use CIFAR-10 **training** examples with indices 0-1023, in index order, transformed only by the accepted `ToTensor` and per-band normalization. Do not use random crop, flip, RandAugment, CutMix, labels, the test set, evaluator outputs, or a favorable subset.
3. Capture the final post-ReLU `[N,128,8,8]` map before pooling. In float64 accumulators compute `A=spatial_mean(map)`, `R=spatial_max(map)-A`, `rms_A=sqrt(mean(A^2))`, and `rms_R=sqrt(mean(R^2))` over all 1024x128 descriptor elements.
4. Set `POOL_RESIDUAL_SCALE = min(1.0, rms_A / max(rms_R, 1e-12))`, serialize the corpus-index rule, state hash, `rms_A`, `rms_R`, raw ratio, capped value, and source-code hash, then hardcode the value to eight significant decimal digits.

On this calibration corpus, the added descriptor `0.10*s*R` therefore has RMS at most 10% of the average descriptor RMS. The cap prevents amplification if the residual happens to be smaller than the average. This is an activation-scale definition, not a fitted accuracy hyperparameter. Once computed, `s` cannot be changed because of output, gradient, timing, checkpoint, or test behavior; a failed gate aborts the point.

The production formula is algebraically `(1-c)*avg + c*max` with fixed `c=0.10*s` in `[0,0.10]`. Do not normalize per batch/example/channel at runtime, because that would introduce a data-dependent training rule and extra reductions. Do not learn `s`, add a gate, or recalibrate after training begins.

## Why this is distinct from EXP014

EXP014 added an independent zero-initialized raw-max classifier. Its initial output identity hid a 4.10x max/average classifier-gradient ratio; one LR-0.1 update made the new weight norm 1.221x the accepted classifier, increased same-batch loss from 5.678 to 56.362, and collapsed all predictions to one class. This proposal has:

- no extra classifier, parameters, optimizer state, LR, or unconstrained branch;
- no raw `max_fc(max)` logit that can grow independently;
- one shared accepted classifier acting on a convex descriptor whose max coefficient is at most 0.10;
- a residual `(max-avg)`, so the perturbation vanishes on spatially uniform evidence and does not double-count the average component;
- mandatory first-output, gradient, update, and exact-corpus gates rather than relying on initial-function identity.

Unlike EXP014, initial logits intentionally differ slightly. The causal claim is a bounded representation perturbation with unchanged classifier capacity, not exact initialization identity.

## Evidence and expected mechanism

The AISTATS mixed-pooling work supports mixtures of average and max statistics as a way to improve invariance and recognition, but does not validate this exact coefficient or shallow CIFAR endpoint. EXP010 supplies stronger local context: CutMix's class-bearing spatial regions improved accuracy by 0.60 points, suggesting final-map location matters, while pure max would be misaligned with area-proportional targets. A <=10% residual contribution is intended to retain extent-sensitive average evidence while adding limited salience.

EXP014 proves the family is dangerous when max evidence has an independent uncontrolled gradient. EXP029 proves even small helpers can cost 1.97% exposure, so the extra max reduction/backward and elementwise blend require real full-step timing. EXP030 shows more terminal fit pressure worsens generalization; this proposal instead changes the descriptor throughout training while preserving the accepted optimizer and 0.01 weak-tail quench.

Expected diagnostics are a nonzero but small descriptor/logit perturbation at initialization, additional gradients at spatial argmax locations without overwhelming the dense average path, switch accuracy near EXP010's 89.73%, first-weak accuracy near or above 93.16%, and final NLL no worse than 0.1934 if localized evidence genuinely helps. A lower strong switch or worse NLL would indicate that sparse salience amplified N1/M7/CutMix artifacts rather than useful regions.

## Exact implementation and invariants

Tracked `train.py` may change only by adding the frozen scalar constant and replacing the three accepted average-pool/view lines with the formula above. Preserve width-2 postactivation ResNet-20, Option-A shortcuts, all 1,073,962 parameters and their seed-42 values, ordinary momentum 0.9, all-parameter coupled decay `1e-4`, batch 128, N1/M7 plus p=0.5 alpha-1 CutMix through the simultaneous 80% boundary, LR 0.01-to-`1e-4` weak hard tail, FP32/default backend flags, timer, evaluator cadence, worker lifecycle, and summary.

Require bitwise-equal candidate/control state dicts and post-construction CPU/CUDA RNG hashes. Pooling must consume no RNG. Every accepted parameter must appear exactly once in the unchanged optimizer group. Hard `[N]` and probability `[N,10]` targets must both produce finite scalar CE. Do not add new production diagnostics inside the counted step; print the fixed `s` once at startup if provenance logging is desired.

Excluded variants include a separate max classifier, learned or per-channel scale, per-batch normalization, absolute raw max addition, GeM, logit normalization, gradient clipping, separate LR, coefficient warmup, tail-only activation, and any coefficient other than the calibration-defined `0.10*s`.

## Calibration and semantic gates

Before training, require:

1. Repeated calibration from the same accepted state/corpus produces the identical eight-digit scalar and hashes. `0 < s <= 1`, `rms_R > 0`, and measured calibration `RMS(0.10*s*R)/RMS(A) <=0.100001`.
2. A synthetic constant final map gives `max=avg`, candidate descriptor/logits equal accepted exactly, and gradients remain finite. Hand-computed random maps match the implementation to FP64-reference tolerance.
3. Candidate parameter/RNG/optimizer identities pass; logits have shape `[128,10]`; the accepted classifier object and initialization are untouched.
4. On identical production-distribution hard and CutMix batches at initialization, record descriptor-delta/average RMS, per-example relative descriptor norms, logit relative-L2, cosine, CE ratio, classifier-gradient norm, backbone-gradient norm, and total first-update norm. Require aggregate descriptor RMS ratio <=0.12, logit cosine >=0.995, candidate/control classifier-gradient and total-update ratios each in `[0.80,1.25]`, and no candidate-only >95% predicted-class concentration after the first update. These gates explicitly exclude EXP014-like scale behavior.

No test images, labels, or `Eval.evaluate()` may enter calibration or safety. Calibration metrics select only the predetermined scalar; safety metrics are pass/fail and cannot revise it.

## Exact-corpus trajectory safety

Materialize and hash once 200 exact post-transform strong batches, balanced near 50/50 hard and resolved alpha-1 CutMix targets, plus 64 exact weak hard batches. Shut down all eight workers. In fresh processes, start accepted/candidate arms from identical logical model/optimizer/RNG states and replay byte-identical tensors in byte-identical order.

Serialize per-step loss, prediction histogram, BN counters, gradient/update norms, descriptor RMS ratios, and finiteness before assertions. Require no corpus mutation/skips, RNG drift, BN-counter mismatch, nonfinite state, or candidate-only class concentration above 95%. Require strong and weak terminal loss-EMA ratios <=1.10, candidate/control total-update p95 <=1.25 and maximum <=1.50, classifier-gradient p95 <=1.30, and maximum per-example descriptor perturbation/average norm <=0.75. The latter tolerates rare sparse peaks while blocking a supposedly 10% aggregate perturbation from hiding gross sample-level domination. Failure vetoes timing and production; do not lower the coefficient as a rescue.

## Fresh paired timing gate

After semantic safety passes, confirm one idle H20 and run five counterbalanced alternating fresh-process control/candidate pairs using the persisted strong-hard, strong-soft, and weak-hard paths. Each arm restores identical state/backend flags, warms 100 steps, then measures at least 1,000 complete synchronized production steps including H2D, forward, CE, backward, SGD, and synchronize. Use the production 40/40/20 path weighting; record mean/median/p95, trial CV, peak memory, and actual steps.

Authorize production only if weighted aggregate candidate/control mean <=1.0100, every pair <=1.04, per-arm trial-mean CV <=2%, candidate weighted p95 <=1.04x control, `floor(26_898 * control_mean/candidate_mean) >=26_629`, peak allocation <620 MiB, and all values remain finite. Benchmark evaluator-like batch-256 inference separately; require candidate/control mean <=1.05, CV <=2%, and conservative total wall projection below 540 seconds with no more than EXP010's 19 evaluations.

The 99% exposure gate is intentionally stricter than EXP014's old 97% gate after EXP029 demonstrated that tiny reductions can erase hundreds of updates. If timing misses, retire this implementation; do not reduce measurement length, move pooling outside the timer, enable fusion/compile, or change `s`.

## Production verification

After all gates pass, run exactly once at seed 42 on one 97,871-MiB H20 with `uv run train.py > run.log 2>&1`; no reroll. Require exit zero, ten unique finite summary fields, 300.0 counted seconds, total below 600 seconds, exactly 1,073,962 parameters, at least 26,629 steps, memory within the timing gate, one switch near 80%, eight stopped strong workers, 45-55% strong CutMix, hard weak targets, and 18-19 unique evaluation epochs, never more than EXP010's 19 and including terminal evaluation.

Compare switch accuracy with 89.73% and the 87.08% strong-underfit marker, first weak with 93.16%, final/best with 94.15%, NLL with 0.1934, steps with 26,898, and best-final gap. Also report the frozen `s`, effective coefficient `c`, calibration RMS values, initial descriptor/logit/gradient ratios, and exact-corpus safety statistics. These mechanism diagnostics cannot substitute for the primary metric.

## Risks

- Hard max is area-insensitive and its gradient is sparse. Even at <=10% coefficient it may overvalue a tiny CutMix donor, augmentation artifact, or one hot location.
- RMS calibration bounds aggregate initialization scale, not every example, later feature distributions, logits, or gradients. The exact-corpus sample/update gates are therefore mandatory.
- Calibration uses one fixed initial model and unaugmented training subset; production training-mode BN and N1/M7 distributions can shift the residual scale. It is a reproducible anchor, not universal normalization.
- Max indices are nonsmooth and ties choose a discrete location. This can introduce trajectory divergence and worse calibration even when all arithmetic is finite.
- Extra adaptive max backward plus elementwise operations may exceed the 1% exposure budget despite EXP014's old near-neutral timing.
- The same classifier loses up to 10% average coefficient when `s=1`; the method may weaken distributed texture/area evidence more than peak evidence helps.
- One fixed seed and a ten-image acceptance margin limit causal precision; a bare threshold pass remains weak evidence.

## Verdict and no-rescue rules

- **Improvement:** all protocol/integrity gates pass and `best_test_acc >=94.25%`. Accept the bounded pooling change. A healthy switch, NLL, and nontrivial calibrated perturbation strengthen but do not define the verdict.
- **No improvement:** a valid production run is below 94.25%. Revert without reroll even if NLL, first-weak accuracy, or one checkpoint improves.
- **Invalid/no-go:** calibration, scale, semantic, safety, timing, scope, hardware, lifecycle, evaluator-count, summary, finiteness, or wall-limit failure. Fix only a demonstrable controller/implementation defect without changing the rule or scalar.

Do not rescue with 5%/20%, a new calibration corpus, uncapped scaling, per-channel normalization, learned gating, EXP014's branch, GeM, gradient clipping, a second seed-42 run, or combination with another candidate. Each is a new experiment.

## Evidence consulted

- `knowledge/papers/mixed-pooling.md` and the AISTATS 2016 source it summarizes.
- `experiments/010/04-analysis.md`, `experiments/014/04-analysis.md`, `experiments/029/04-analysis.md`, and `experiments/030/04-analysis.md`.
- Goal definition, system understanding, learnings/results through EXP030, current `train.py`, and EXP031 brainstorm.
