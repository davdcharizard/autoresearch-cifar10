# EXP-015 Paired Timing Gate

**Attempt**: exact retry after attempt 1's isolated cold control-inference outlier
**Hardware**: one idle NVIDIA H20, 97,871 MiB
**Protocol**: five alternating fresh control/candidate processes; 100 warmups and 500 synchronized training or inference measurements per arm; no script, threshold, order, or candidate change from attempt 1.

## Training

- accepted/candidate median trial means: 10.9335 / 10.9250 ms
- candidate/control ratio: 0.999219
- projected candidate steps from 26,898: 26,919
- accepted/candidate trial-mean CV: 0.496% / 0.823%
- candidate/control median p95 ratio: 0.990778
- accepted/candidate peak allocation: 598.671 / 598.671 MiB

All gates passed: ratio <=1.01, projected steps >=26,629, CV <=2%, p95 ratio <=1.02, finite losses, candidate allocation <650 MiB, and zero allocation delta.

## Inference

- accepted/candidate median trial means: 1.98742 / 1.98648 ms
- candidate/control ratio: 0.999527
- accepted/candidate trial-mean CV: 0.817% / 0.910%
- conservative projected total: 330.700 seconds

All gates passed: ratio <=1.01, CV <=2%, and projected total <540 seconds. Batch size/loader length remain 128/390, so 18-19 unique evaluations are expected and more than 19 remains forbidden.

**Verdict**: PASS. The single fixed-seed accuracy run may launch.
