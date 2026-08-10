# EXP-014 Paired Timing Gate

**Date**: 2026-08-06
**Hardware**: one idle NVIDIA H20, 97,871 MiB
**Protocol**: five alternating fresh-process accepted/candidate pairs; 100 warmups and 500 synchronized measurements per arm; training alternated hard and probability targets and included H2D, forward, CE, backward, SGD, and synchronization.

## Training

- accepted median-of-trial means: 10.9357 ms
- candidate median-of-trial means: 10.9508 ms
- candidate/control ratio: 1.001381
- projected candidate steps from 26,898: 26,860
- accepted/candidate trial-mean CV: 0.136% / 0.360%
- candidate/control p95 ratio: 0.998952
- accepted/candidate peak allocation: 598.671 / 598.686 MiB

All training gates passed: ratio <=1.025, projected steps >=26,242, CV <=2%, p95 ratio <=1.07, candidate allocation <620 MiB, and all losses finite.

## Inference

- accepted median-of-trial means: 1.97871 ms
- candidate median-of-trial means: 2.03113 ms
- candidate/control ratio: 1.026495
- accepted/candidate trial-mean CV: 0.477% / 0.547%
- conservative total projection across 19 evaluator passes: 330.779 s

All inference gates passed: ratio <=1.05, CV <=2%, and projected total <540 seconds. Batch size and loader length remain 128 and 390; evaluator call count and elapsed-progress cadence are unchanged, so the full run must still verify 18-19 unique evaluation epochs and never more than 19.

**Verdict**: PASS. The single seed-42 accuracy run may launch.
