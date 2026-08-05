# Adversarial Idea Review EXP-046

## Reviewer Provenance

I developed `proposals/idea-03.md` before receiving this critic assignment. I
therefore give label smoothing no credit for familiarity, proposal length, or
verification detail, and judge it only from the accepted source, saved
literature, and experimental record used for all three randomized finalists.

## Prioritized Feedback

1. **Advance CIFAR-mean constant crop fill; it has the strongest source-level diagnosis and preserves every accepted learning mechanism.** Accepted `train.py` uses raw-black constant crop padding but subtracts `(0.4914,0.4822,0.4465)` with unit standard deviation, while accepted RandAugment already fills out-of-bounds RGB with `(125,123,114)`. The proposal's arithmetic is sound: only crop offset `(4,4)` avoids padding, so `80/81` crops touch synthetic pixels; independent uniform offsets give about `13.41%` synthetic area on average. Quantized mean fill maps to approximately `(-0.001204,+0.000153,+0.000559)` rather than `(-0.4914,-0.4822,-0.4465)`. This is a genuine train-distribution change aimed at the diagnosed generalization gap, not evaluator manipulation or a seed reroll.

2. **Mean fill is distinct from failed EXP032 reflection, but “inconsistency” is not proof that black borders are harmful.** EXP032 changed boundary geometry and entered a NumPy reflection path; it failed worker-timing stability before any score. This candidate retains constant PIL padding, crop geometry, transform order, and the same two crop RNG draws, changing only pixel value. Its likely failure mechanism is that black padding supplies useful structured occlusion, crop-offset evidence, or contrast regularization, while mean fill remains an unnatural textureless block that RandAugment can spread. Keep the fixed asymmetric-pixel oracle, exact crop/flip/private-RandAugment decision replay, and report normalized pad fraction and input-moment deltas only as non-tuning diagnostics. Do not respond to them with a different color, schedule, padding width, or mode.

3. **The mean-fill state/RNG and cost claims are credible, but active persistent-worker timing must still decide feasibility.** Scalar-black and tuple-mean constant PIL padding should share the accepted `ImageOps.expand` route and draw no RNG; different pixels may change RandAugment outputs but must not change its sampled operation/sign/magnitude or worker-private pre/post states. Model construction, mixup draws, parameters, and optimizer state are independent of fill. GPU shapes and work remain exact, but loader wait lies outside the counted per-step timer and EXP032 exposed host outliers. Preserve the proposal's one-worker-pool-at-a-time active/inactive loader gate and print measurements before assertions. A stable timing failure closes this exact implementation without an accuracy score; it does not authorize reflection or alternate-fill rescue.

4. **Pooled-feature manifold mixup is rigorous but bundles two inseparable mechanisms while deleting the strongest relevant positive treatment.** EXP002's early input mixup gained `+0.69` points, and EXP004/005/020/035 protect its 65%/alpha-0.2 operating point. The candidate simultaneously removes convex pixel inputs and their spatial-BN statistics, then adds post-GAP interpolation immediately before the nonlinear pooled MLP. A result cannot distinguish cleaner BN statistics from decision-manifold interpolation, and the saved knowledge contains no manifold-mixup evidence. The most likely failure is that a 128-dimensional late mixture is too easy or off-manifold and cannot replace low-level invariance learned by the complete spatial backbone. The implementation correctly keeps one Beta draw, one permutation, paired targets, one forward, the sole refined-path CE, and exact hard tail. If selected in a future loop, interpret it only as the complete fixed replacement and retain its no-placement/no-compound closure; do not attempt to “isolate” the confound with a second score.

5. **Early-only label smoothing is algebraically correct but has the weakest causal case; my prior authorship does not change that.** PyTorch's all-class convention makes the paired smoothed CEs exactly `q=0.95*y_mix+0.05*u`: different-class masses are `0.95*lambda+0.005`, `0.95*(1-lambda)+0.005`, and `0.005` elsewhere, while a same-class target has mass `0.955`. The proposal correctly distinguishes accepted-identical hard-tail code from the necessarily different incoming learned state and makes only step-aligned RNG claims. Nevertheless, mixup already provides soft targets, alpha and duration are locally bracketed, no training-only calibration diagnostic supports uniform mass, epsilon 0.05 is conventional, and EXP041 warns that another CE-derived constraint can weaken the accepted pooled-head learner. This candidate was already deprioritized in EXP044/045 review without any new supporting evidence. It is valid, but it should not consume the score ahead of a diagnosed input artifact.

6. **All three candidates satisfy the literal hard constraints and correctly fail closed.** They modify only `train.py`, add no dependency, preserve one H20 and the frozen 300-second budget, validate no more than once per epoch, retain seed 42, use the 600-second kill limit, and require one redirected `run.log`. Treat “reconfirm baseline” as a source/provenance audit, not another scored baseline run. Success must require both `best_test_acc >=94.58%` and realized exposure `>=127`; final accuracy/loss cannot rescue a miss, a valid score is never rerun, and invalid repair is limited to independently demonstrated harness/infrastructure defects with production semantics frozen.

## Scored Verdict

### CIFAR-Mean Constant Fill for Random Crops

- **Strength of evidence and reasoning: 4/5.** The normalized border mismatch, affected-crop frequency, constant-path distinction from EXP032, RNG argument, and accepted RandAugment fill provide unusually concrete local support; only evidence that black borders actually cause residual errors is missing.
- **Potential impact: 3/5.** The change reaches about 13.4% of pixels in almost every training crop at no GPU cost and can materially alter learned invariance, but it removes one simple artifact rather than adding capacity or a new optimization signal.

### Early Pooled-Feature Manifold Mixup

- **Strength of evidence and reasoning: 3/5.** Placement, Jacobian, RNG, cost, and closure are coherent, and EXP036 supports downstream nonlinearity, but there is no direct manifold-mixup source and the treatment removes EXP002's strongest validated input regularizer while confounding BN-distribution and feature-linearity effects.
- **Potential impact: 4/5.** Relocating the complete early convex-label prior can reshape the backbone/head decision geometry throughout 65% of training, so the upside is substantial even though the direction is poorly supported.

### Early-Only Epsilon-0.05 Uniform Label Smoothing

- **Strength of evidence and reasoning: 2/5.** Dense-target semantics, temporal gating, state/RNG controls, and closure are exact, but no calibration diagnosis or local epsilon evidence overcomes redundancy with accepted mixup.
- **Potential impact: 2/5.** It is nearly free and could modestly temper confidence, but adds no information and mostly weakens an already bracketed early target trajectory.

## Selected Lead

**CIFAR-Mean Constant Fill for Random Crops** is the single strongest idea. It wins over manifold mixup because it removes a quantified, code-demonstrable artificial boundary while preserving the accepted input mixup, spatial BN regime, pooled head, optimizer, and all stochastic decisions; manifold mixup instead replaces a `+0.69`-point validated mechanism with an unsupported late relocation. It wins over label smoothing because it changes actual input evidence rather than redundantly adding target entropy without a calibration diagnosis. Advance exactly constant `(125,123,114)` fill with the proposal's semantic, active/inactive loader, sole-score, and no-rescue gates.
