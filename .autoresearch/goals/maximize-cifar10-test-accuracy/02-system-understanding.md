# System Understanding: Maximize CIFAR-10 Test Accuracy
**Last verified**: 2026-07-27 @ a7c42dc (baseline: 94.48%)

## Problem Decomposition

- **Setup** - cost: 1.1 s, 0.3% of 345.3 s wall; bound by CPU/process initialization. Evidence: EXP-027 scored summary in `experiments/027/03-execute.md`.
- **Training step** - cost: exactly 300.0 s counted, 87.2% of wall, 25,450 steps / 130.304 data passes. Evidence: EXP-036 scored summary in `experiments/036/03-execute.md`.
- **Forward/backward** - cost: 24.1%/73.7% of a 13.914 ms isolated mixup step and 24.3%/74.2% of a 13.769 ms hard step. Evidence: balanced 3x50-step CUDA-event probe at 67c8e98; all other measured components were individually below 0.8%.
- **Forward stages** - cost: stem 1.9%, stage 1 39.6%, stage 2 27.6%, stage 3 29.5%, head 1.4% of 3.224 ms. Evidence: 150-forward CUDA-event probe at batch 256 on the H20.
- **Late prefix gradients** - cost: freezing stem/stage 1 reduced complete hard steps from 11.213 to 7.191 ms, a 35.9% saving. Evidence: EXP-028 balanced timing; the scored tail rose from about 22k to 35.5k images/s.
- **Parameter placement** - stage 3 owns 820,608 of 987,098 parameters (83.1%) but 29.5% of forward time; stages 1/2 own 32,992/131,520 parameters. Evidence: direct accepted-model parameter enumeration and stage probe.
- **Data path** - cost: H2D 0.6%; early mixup plus zeroing 0.6%, hard-tail zeroing <0.1% in the isolated step probe. Eight persistent workers overlap early RandAugment; EXP-027 loader medians were 2.794/2.894 s per 195-batch base/composed epoch.
- **Evaluation and epoch boundaries** - cost: 42.8 s after subtracting setup and counted training, 12.4% of wall over 27 unique evaluations. This is outside the scored 300 s and leaves ample room under the 600 s hard limit. Evidence: EXP-036 343.9 s total and accepted-cadence audit.

## Current Bottleneck

The metric is limited by the accuracy/exposure tradeoff inside GPU training, not I/O, wall limit, or memory. The model nearly interpolates the training tail (smoothed loss about 0.0028) yet finishes at 0.2456 test loss, so generalization and boundary quality remain limiting. Backpropagation consumes about 74% of the prior counted step; extra spatial compute directly reduces the new 130-pass operating regime. EXP-036 shows that 16,384 post-pooling parameters can improve decision quality without spatial cost, while EXP-028 shows high-resolution adaptation remains essential.

## Headroom Assessment

- Memory is not binding: EXP-027 peaked at 1096.3 MiB on a 97,871 MiB H20, leaving about 98.9% allocation headroom.
- Wall time is not binding at the accepted cadence: 343.9 s leaves 256.1 s under the hard timeout, while counted training remains fixed at 300 s.
- Data augmentation is successfully overlapped: composed loader timing added only 0.100 s per synthetic epoch and did not reduce realized counted exposure below the 130-pass gate.
- Compute is binding: optimizer, H2D, mixup, and loss together were only about 1.8% of the pre-head isolated mixup step, so micro-optimizing them cannot create material exposure. Changes must improve generalization at near-zero spatial cost or buy more accuracy per backward pass.
- Theoretical compute utilization was not measured; absolute CUDA-event steps (13.8-13.9 ms) differ from the scored mean (11.55 ms), so use the probe for component shares and direct scored runs for exposure.

## Open Questions

- Can the pooled residual head compose with a genuinely orthogonal regularization or optimization mechanism without dropping below roughly 127 passes?
- Does classifier-specific decay allocation improve the new nonlinear pooled representation, or merely inflate logits and worsen confidence?
- Can a training-derived normalized-logit scale improve angular geometry without a scale sweep or frozen-test feedback?
- Which CIFAR-10 error classes account for the remaining 5.52%, given that evaluator access cannot be used for experiment design?
