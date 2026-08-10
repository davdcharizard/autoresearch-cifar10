# Report EXP-038: Output-RMS-Matched Cosine Classifier
- **Created**: 2026-08-06

## Goal

Test whether a bounded angular classifier can raise seed-42 CIFAR-10 `best_test_acc` from the moving 94.15% baseline to at least 94.25%, while changing only `train.py` and preserving the one-H20, 300-second training and accepted evaluator protocol.

## Idea & Hypothesis

The candidate normalized pooled features and each classifier row, then applied frozen scale `22.786916732788086`. The scale came from output-RMS matching on a pre-existing immutable accepted batch rather than labels, accuracy, or a sweep. The hypothesis was that removing radial feature/weight shortcuts would bound confidence and improve strong-view/CutMix generalization without destabilizing shared SGD/decay geometry.

## Approach

`train.py` retained the existing Linear module, bias, parameter order, global average pooling, backbone, optimizer, decay, data, schedule, timer, evaluator, and summary. Only its final function changed from affine logits to `scale * F.linear(normalize(features), normalize(fc.weight))` with epsilon `1e-6`; the stored bias was unused in-function. Adversarial plan review removed a proposed evaluation-cadence rewrite, kept switch fit informational after production, and expanded slow-drift verification. Construction/formula/calibration and short exact-corpus gates preceded a planned long replay, timing, and production run.

## Execution

Static and construction checks passed after two controller fixes. The first diagnostic hook accidentally returned a tensor and replaced `layer3` output; it was changed to a named `None`-returning hook. The second controller revision replaced an undefined scalar-square-root zero-vector derivative with `torch.linalg.vector_norm` and allowed 50 ppm for default-cuDNN RMS reproducibility. The corrected controller passed. One 200-strong/64-weak replay of two accepted controls and the candidate then completed in finite state. Its global gates passed, but the candidate had already violated the authoritative lifetime row-dispersion bound, making the conditional long replay unable to pass. Timing and production were not run, with no parameter rescue or reroll.

## Results

- **Primary metric**: NaN (baseline: 94.15%, delta: N/A; production was not authorized)
- **Observations**: Construction/RNG/state and the 1,073,962-parameter inventory matched exactly. Accepted/candidate CutMix logit RMS was `2.760024/2.760048`; hard-view RMS ratio was `0.998027`. FP64 random/tiny/zero formula and VJPs matched at numerical precision, logits were bounded, and actual SGD kept the unused bias bitwise fixed. In the 264-step replay the candidate had no specific concentration or update spike; its maxima were `0.6226x` control logits, `0.9894x` gradients, and `0.9079x` updates, with strong/weak loss EMA ratios `0.9811/0.9587`. Nevertheless classifier row max/min crossed3 at step4, peaked at4.5122 on step15, and exceeded3 at261/264 looks, ending at3.1028.
- **Analysis**: Output-scale calibration controlled initial global logits and the angular head remained globally finite and concentration-free. It did not control per-class raw-row optimizer geometry: shared LR, momentum, and coupled decay produced sharply different row norms almost immediately. Because normalized logits divide by each row norm, that dispersion implies different effective tangent learning rates across classes even while raw logits stay bounded. The evidence does not show an accuracy regression or exhaust angular classification; it rejects this exact fixed-scale, epsilon, unchanged-SGD/decay operating point under the preregistered lifetime invariant. An intrinsically fixed-row-norm or projected optimizer formulation would be a separate intervention, not a post-hoc repair.
- **Key Learning**: Output-RMS parity bounded global logits but not per-class optimizer geometry; cosine rows exceeded 3x raw-norm dispersion from step4 onward.

## Verification

- **Conditions**: Corpus integrity, scope/static checks, construction/RNG/inventory, FP64 formula/VJPs, zero-vector finiteness, logit bound, calibration, bias immutability, exact BN counters, global ratios, updates, and concentration passed. The preregistered lifetime classifier row max/min ceiling failed at261/264 looks. Long replay, timing, and production were skipped by plan.
- **Review Notes**: The failure is not a control or infrastructure artifact: both accepted arms qualified the global gates, controller/source/corpus hashes were durable, and the row statistic is direct and denominator-safe. The `<=3` ceiling may be conservative for a functionally normalized head, but changing it after observing persistent 4.51x dispersion would invalidate the prospective protocol.
- **Verdict**: invalid
- **Verdict Basis**: A prospective candidate-specific safety gate failed before production, leaving no trustworthy primary metric. No goal hard constraint was violated and no infrastructure crash occurred.

## Unexplored Avenues

- Explicit unit-row projection after each optimizer step would remove raw-norm dispersion and equalize tangent scaling, but it changes momentum/decay semantics and needs a new independently reviewed hypothesis.
- A weight-normalization parameterization with controlled learned scales could preserve angular geometry while making class-wise scale explicit; it would reintroduce tunable radial degrees of freedom and is not an adjacent retry.
- Logit normalization without classifier-row normalization may retain ordinary classifier optimizer geometry, but it has different translation/gradient behavior and requires its own non-accuracy temperature calibration.

## Next Steps

- **High confidence**: return to a clean untried mechanism such as FP32 channels-last or another bounded representation change rather than tuning this failed head.
- **Medium confidence**: retain raw-row norm dispersion as a mandatory diagnostic for any future normalized classifier or weight-reparameterized head.
- **Low confidence**: revisit angular classification only with an intrinsic norm-control mechanism justified before results, not a scale/epsilon/decay adjustment to EXP038.
