# EfficientNetV2: Smaller Models and Faster Training — progressive learning (arXiv 2104.00298, ICML 2021)
- **Authors**: Tan, Le (Google)
- **URL**: https://arxiv.org/abs/2104.00298
- **Status in project**: FULLY measured, both halves negative here — image-size half ZERO conversion (EXP-031); regularization-ramp half (TA+RE gated off for the 21-epoch warmup) read 96.38 = mean −1.2σ (EXP-065) despite the light phase training visibly faster (loss 0.76 vs ~1.1) — banked easy-distribution progress inverts when full pressure arrives. Progressive learning does not transfer to short-budget heavy-aug CIFAR in either component.

## Claim

Training speeds up substantially (up to 11×) with no accuracy loss when BOTH image size AND
regularization strength (dropout, RandAugment magnitude, mixup) ramp from low to high over
training — and the pairing is load-bearing: small images + weak reg early, large images +
strong reg late. Ablation shows ramping reg with FIXED image size still helps modestly;
applying strong reg to small images HURTS.

## Project-relevant readings

- The image-size half does not transfer to CIFAR (EXP-031: 24px discards signal, not
  redundancy — zero conversion). The reg-ramp half at FIXED 32px is the residual unmeasured
  claim: early training at lower aug pressure converges faster per step.
- Maps onto this project as the HEAD-side quadrant of the tail-pressure law: EXP-025/033
  measured that the TAIL must keep full pressure (lightening it loses both ways); whether the
  HEAD must too is unmeasured. The warmup phase (first 15% of budget) is the principled
  boundary — heat ramps while aug is light, full aug for the entire anneal.
- Inversion risk precedent: EXP-018 (zero-γ init) showed mechanisms that "turn on during peak
  heat" lose — TA+RE switching on at p=0.15 (peak LR) is exactly that signature, on the data
  side. The experiment decides which precedent governs.
- dt unchanged (augmentation is CPU-worker-side; charged step is GPU-bound) — zero toll,
  launch-certain, byte-identical run signatures expected.
- Mid-run transform switching with persistent workers requires the EXP-041 shared-memory
  tensor pattern (workers hold forked transform copies; a `torch` shared-memory flag read
  inside the transform propagates).
