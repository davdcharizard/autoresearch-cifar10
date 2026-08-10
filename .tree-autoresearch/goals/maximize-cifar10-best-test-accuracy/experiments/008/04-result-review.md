# Claude Adversarial Result Review: EXP-008

## Stop Decision

Claude judged the stop justified, but found the preregistered 1.20x absolute loader-headroom floor miscalibrated because the parent's worst 29,485.7 images/s also falls below 30,720. The decision remains robust on the raw comparison: candidate best 18,601 is below parent worst 29,485.7, and every candidate epoch is below the roughly 25,600-image/s early GPU demand. Even discounting that demand estimate by 25% leaves 19,200, still above the candidate's best.

## Integrity Findings

- The 1.20x absolute gate must not be reused; future gates should be relative to a parent measured by the same harness and should pass the parent by construction.
- The planned under-550-second total projection was not measured and must not be claimed.
- Loader-only throughput cannot be converted directly into full-run wall time, but its no-contention bias favors the candidate; production contention would not erase this deficit.
- The measured slowdown belongs to the complete paired-view package. No ablation separates RandAugment operation cost from dual-view normalization, serialization, IPC, and pinning.
- Correctness evidence is strong and orthogonal: the transform is live, parent clean/RNG semantics are preserved, and private worker streams are distinct/replayable for the tested configuration.
- No metric, GPU job, retry, or charged training time was consumed.

## Safe Learning

- The parent CPU pipeline has only about 15-35% loader headroom over early GPU demand, so this magnitude of extra per-sample CPU work is unaffordable.
- The fixed paired clean-FP32 plus augmented-uint8 package costs about 2.1x parent loader time.
- Exact paired-view phase control with bitwise parent parity and isolated private RNG is technically achievable and reusable.
- Nothing was learned about RandAugment's effect on CIFAR-10 accuracy.

## Verdict

Claude preferred a dedicated `preflight-reject` state and suggested `invalid` if such a label is unavailable, while acknowledging `crash/NaN` as the schema's mechanical encoding of last resort. The tree skill's frozen classification guide explicitly maps “no results produced” to `crash` and reserves `invalid` for hard-constraint or trust failures. This analysis therefore records `crash/NaN`, with the report and key learning explicitly stating that the code did not crash and the augmentation accuracy hypothesis remains untested.

## Next Direction

Keep the loader parent-identical and move augmentation work on-device, or first ablate transport-only versus operation-only CPU cost. Any future feasibility gate must use a parent-relative same-harness comparison and a measured GPU-demand distribution.
